from email import header
import enum
from functools import reduce
from cv2 import norm
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import SpectralClustering

from pathlib import Path
import re
from sys import argv

from utils.tools import ordered_cmat, correct_predictions


DATA_ROOT = "physionet.org/files/mimic-cxr/2.0.0/"
REPLACE_TABLE = {
    r"\n+": " ", # Multiple new lines
    r"\.+": "", # ...
    r"\,+": "",
    r"\?+": "",
    r"/+": "",
    r":+": "",
    r";+": "",
    r"-+": "",
    r"_+": "",
    r"\(": "",
    r"\)": "",
    r"[A-Z][A-Z]+": "", # Replace capital words due to the structure of radiology reports
    r"[0-9]": "",
}



def normalise(input):
    if type(input) == pd.DataFrame:
        return (input.fillna(0.) - input.mean())/input.std()
    elif type(input) == np.ndarray:
        return (np.nan_to_num(input, 0.) - np.nanmean(input, 0))/np.nanstd(input, 0)


def locate(df, column, column_value, out_column, require_unique = True):
    out = df[df[f"{column}"] == column_value][f"{out_column}"]
    if require_unique:
        assert len(out.unique()) == 1, "More than one value"
    return out.iloc[0]


def clean_text(text):
    for k, v in REPLACE_TABLE.items():
        text = re.sub(k, v, text)
    return re.sub(r"\s+", " ", text).strip()


def read_text(path):
    try:
        with open(f"{DATA_ROOT}{path}") as f:
            lines = " ".join(f.readlines())
            return clean_text(lines)
    except FileNotFoundError:
        try:
            with open(f"/home/erland/Desktop/temp_save/{DATA_ROOT}{path}") as f:
                lines = " ".join(f.readlines())
                return clean_text(lines)
        except FileNotFoundError:
            pass


def get_multiclass_diff(counts, one_hot_diff = False):
    counts = pd.DataFrame(counts)
    diff = counts[~counts.isnull().any(axis=1)]
    diff = diff.append(pd.DataFrame(0, index=counts.index[~counts.index.isin(diff.index)], columns=diff.columns))
    if one_hot_diff: return diff
    out_weights = pd.DataFrame(0, index=diff.index, columns=diff.columns)
    for diag, col in diff.iteritems():
        for _, col2 in diff.loc[:,~diff.columns.isin([diag])].iteritems():
            out_weights[diag] = out_weights[diag] + (col - col2)
    return out_weights    


def get_binary_diff(A, B):
    a_in_b = A.index.isin(B.index)
    b_in_a = B.index.isin(A.index)
    a = A[a_in_b]
    a = a.append(pd.Series(0., index=A[~a_in_b].index)).sort_index()
    a = a.append(pd.Series(0., index=B[~b_in_a].index)).sort_index()
    b = B[b_in_a]
    b = b.append(pd.Series(0., index=A[~a_in_b].index)).sort_index()
    b = b.append(pd.Series(0., index=B[~b_in_a].index)).sort_index()
    return (a - b).sort_values(ascending = False)
    

def weight_sentence(series_format, text, weight):
    words, counts = np.unique(text.lower().split(" "), return_counts = True)
    series_format[words] = counts
    return series_format


def get_text_weights(data, weights, words, out_format = "sum"):
    words_format = pd.Series(0, index = words)
    out_weights = data["text"].apply(lambda x: weight_sentence(words_format.copy(), x, weights))
    out_weights = out_weights.groupby(level=0).mean()
    if out_format == "sum":
        return (out_weights * weights.values.reshape(1,-1)).sum(1)
    elif out_format == "array":
        return out_weights * weights.values.reshape(1,-1)


def get_multiclass_weights(text, diff):
    weighted_reports = []
    words_format = pd.Series(0, index = diff.index)
    for diag in diff.columns:
        diff_ = text.apply(lambda x: weight_sentence(words_format.copy(), x, diff[diag]))
        diff_ = diff_.groupby(level=0).mean()
        weighted_reports.append((diff_ * diff[diag].values.reshape(1,-1)).sum(1))
    weighted_reports = pd.concat(weighted_reports, axis=1)
    weighted_reports.columns = diff.columns
    return weighted_reports


def plot(X, labels, fname, title = None, **sns_kwargs):
    if (X.shape[1] > 2):
        from sklearn.manifold import TSNE
        X = TSNE(perplexity=15).fit_transform(X)
        # shuffle
    plot_ = sns.scatterplot(
        x=X[:,0],
        y=X[:,1],
        hue=labels,
        **sns_kwargs
    )
    if title is not None: plot_.set_title(title)
    plt.savefig(fname, bbox_inches='tight')
    plt.clf()


def affinity_plot(X, labels, fname, title = None, log=False, **sns_kwargs):
    assert X.shape[0] == X.shape[1], "not graph structure"
    idxs_ = []
    i = 0
    ticks = []
    for diag in np.unique(labels):
        idx_ = np.argwhere(labels == diag)[:,0]
        idxs_.append(idx_)
        if i != 0: ticks.append((i, ""))
        ticks.append((i + len(idx_)/2, diag))
        i += len(idx_)
        if i != len(labels): ticks.append((i, ""))
    idxs_ = np.concatenate(idxs_)
    X_ = X[np.ix_(idxs_,idxs_)]
    labels_ = labels[idxs_]

    if log:
        if X_.all():
            X_ = np.log(X_)
        else:
            X_ = np.log1p(X_)
    plot_ = sns.heatmap(X_, **sns_kwargs)
    if title is not None: plot_.set_title(title)
    plt.xticks(*zip(*ticks))
    plt.yticks(*zip(*ticks))
    plt.savefig(fname, bbox_inches='tight')
    plt.clf()


def spectral_clustering_accuracy(index, feature_matrix, labels, one_class = "None", **spectral_clustering_kwargs):
    pred = SpectralClustering(n_clusters = len(labels.unique()), **spectral_clustering_kwargs).fit_predict(feature_matrix)
    acc, _ = ordered_cmat(labels[index], pred)
    # max(
    #     ((labels[index] != one_class).values.astype(float) == pred).mean(),
    #     ((labels[index] == one_class).values.astype(float) == pred).mean()
    # )
    return acc, pred


def wordcloudify(words, fname, title=None, exceptions=None):
    with plt.style.context("dark_background"):
        plt.figure(dpi=1200)

        wc = WordCloud(
            width=960,
            height=720,
            collocations=False,
            stopwords=exceptions,
        ).generate(words)
        
        if title is not None: plt.title(title)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")

    plt.savefig(fname, bbox_inches='tight')
    plt.close()


def count_split(labels, images):
    classes = labels.unique()
    counts = {}
    texts = pd.DataFrame()
    for cl in classes:
        idx_ = images["hadm_id"].isin(labels[labels == cl].index)
        paths_ = pd.DataFrame(images[idx_]["study_path"].unique(), columns=["path"])
        paths_["text"] = paths_["path"].apply(lambda path: read_text(path))
        paths_ = paths_[paths_["text"].notna()]
        paths_["unique text"] = paths_["text"].apply(lambda x: " ".join(set(x.lower().split(" "))))

        texts = texts.append(paths_)
        counts[cl] = pd.Series(" ".join(paths_["unique text"].values).split(" ")).value_counts() / len(paths_)
    texts.index = texts["path"].apply(lambda x: locate(images, "study_path", x, "hadm_id"))
    return counts, texts


def _get_acc(pred, labels, label_assignment = True, return_pred=True):
    if type(labels) == pd.Series:
        factorised_labels, diag = pd.factorize(labels)
    else:
        factorised_labels = labels
    acc, cmat, (ri, ci) = ordered_cmat(factorised_labels, pred, label_assignment = label_assignment, return_ri_ci=True)
    if label_assignment:
        if type(labels) == pd.Series:
            pred_str = pd.Series(diag[correct_predictions(pred, ri, ci)], index=labels.index)
        else:
            pred_str = correct_predictions(pred, ri, ci)
    else:
        if type(labels) == pd.Series:
            pred_str = pd.Series(diag[pred], index=labels.index)
        else:
            pred_str = pred
    if return_pred: return acc, cmat, pred_str
    else: acc, cmat



def kmeans(X, y, print_out = True, return_acc = False, **kmeans_kwargs):
    from sklearn.cluster import KMeans
    pred = KMeans(n_clusters = len(np.unique(y)), **kmeans_kwargs).fit_predict(X)
    acc, _, pred = _get_acc(pred, y)
    if print_out: print(f"Unsupervised k-means accuracy : {acc}")
    if not return_acc: return pred
    else: return pred, acc


def spectral_clustering(X, y, print_out = True, return_acc = False, **sc_kwargs):
    sc = SpectralClustering(n_clusters = len(np.unique(y)), **sc_kwargs)
    pred = sc.fit_predict(X)
    acc, _, pred = _get_acc(pred, y)
    if print_out: print(f"Unsupervised spectral clustering accuracy : {acc}")
    if not return_acc: return (pred, sc.affinity_matrix_)
    else: return (pred, sc.affinity_matrix_), acc


def gmm(X, y, **gmm_kwargs):
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=len(np.unique(y)), **gmm_kwargs).fit(X)
    pred = gmm.predict(X)
    acc, _, pred = _get_acc(pred, y)
    print(f"Unsupervised GMM accuracy : {acc}")
    return pred, acc


def selftrain(X, y, idx, **kwargs):
    from sklearn.svm import SVC
    from sklearn.semi_supervised import SelfTrainingClassifier
    labels, diag = pd.factorize(y.copy())
    labels_ss = labels.copy()
    labels_ss[list(set(range(len(labels_ss))) - set(idx))] = -1

    svc = SVC(probability=True, kernel="linear")
    lp = SelfTrainingClassifier(svc, **kwargs).fit(X.values, labels_ss.astype(float))
    pred = lp.predict(X)
    acc, _, pred = _get_acc(pred, labels, label_assignment = False)
    pred_str = pd.Series(diag[pred.astype(int)], index=y.index)
    print(f"Self-training accuracy : {acc}")
    return pred_str


def pca(features, dim, **kwargs):
    from sklearn.decomposition import PCA
    return PCA(n_components=dim, **kwargs).fit_transform(features)


def unsupervised_diff(
    labels,
    texts,
    index,
    words,
    images,
    binary = False,
    semi_supervised_index = None,
    PCA=False,
    PCA_dim = 250,
    clustering_method = "kmeans",
    **clustering_kwargs
):
    features = pd.DataFrame(0, index=index, columns=words)
    for idx, row in features.iterrows():
        rowtext = texts["unique text"].loc[idx]
        if type(rowtext) != str:
            rowtext = rowtext.values
        else:
            rowtext = [rowtext]
        row.loc[" ".join(rowtext).split(" ")] += 1

    cl = labels.unique()
    if PCA: features = pca(features, PCA_dim)
    if clustering_method == "kmeans":
        pred = kmeans(features, labels, **clustering_kwargs)
        affinity_matrix = None
    elif clustering_method == "spectral":
        # Features shape?
        pred, affinity_matrix = spectral_clustering(features, labels, affinity = "rbf", **clustering_kwargs)
    elif clustering_method == "gmm":
        pred, acc = gmm(features, labels, **clustering_kwargs)
        affinity_matrix = None
    elif clustering_method == "semi_supervised":
        assert semi_supervised_index is not None
        pred, set_pos = selftrain(features, labels, semi_supervised_index, **clustering_kwargs)
        affinity_matrix = None

    if affinity_matrix is None:
        pred = pred.astype("object")
        counts, _ = count_split(pred, images)
        diff = get_binary_diff(counts[cl[0]], counts[cl[1]])
        weights = get_text_weights(texts.loc[index], diff, words)
        return pred, weights
    else:
        return pred, affinity_matrix


def unsupervised_multiclass_diff(
    labels,
    texts,
    index,
    words,
    images,
    binary = False,
    semi_supervised_index = None,
    PCA=False,
    PCA_dim = 250,
    clustering_method = "kmeans",
    **clustering_kwargs
):
    features = pd.DataFrame(0, index=index, columns=words)
    for idx, row in features.iterrows():
        rowtext = texts["unique text"].loc[idx]
        if type(rowtext) != str:
            rowtext = rowtext.values
        else:
            rowtext = [rowtext]
        row.loc[" ".join(rowtext).split(" ")] += 1

    cl = labels.unique()
    if PCA: features = pca(features, PCA_dim)
    if clustering_method == "kmeans":
        pred = kmeans(features, labels, **clustering_kwargs)
        affinity_matrix = None
    elif clustering_method == "spectral":
        pred, affinity_matrix = spectral_clustering(features, labels, affinity = "rbf", assign_labels = "discretize", **clustering_kwargs)
    elif clustering_method == "gmm":
        pred, acc = gmm(features, labels, **clustering_kwargs)
        affinity_matrix = None
    elif clustering_method == "semi_supervised":
        assert semi_supervised_index is not None
        pred = selftrain(features, labels, semi_supervised_index, **clustering_kwargs)
        affinity_matrix = None

    if affinity_matrix is None:
        pred = pred.astype("object")
        counts, _ = count_split(pred, images)
        diff = get_multiclass_diff(counts, one_hot_diff = True)
        weights = get_multiclass_weights(texts["text"].drop_duplicates(), diff)
        return pred, weights
    else:
        return pred, affinity_matrix


def match_hadm_id(df, idx):
    try:
        out = df.loc[idx]["hadm_id"]
        if type(out) == pd.Series:
            assert len(out.unique()) == 1
            out = out.iloc[0]
        return out
    except KeyError:
        pass


def W_to_A(W, epsilon = 0., noise_sd = 0., transform=None):
    A = W @ W.T
    A = (A - A.mean()) / A.std()
    if epsilon > 0.:
        A[A < epsilon] = 0.
    if noise_sd > 0.:
        A = A + np.random.normal(scale=noise_sd, size=A.shape)
    A = A - A.min()
    if transform == "knn":
        from sklearn.neighbors import KNeighborsTransformer
        A = KNeighborsTransformer(n_neighbors = 50, metric="precomputed").fit_transform(A).todense()
    return (A + A.T)/2


def get_kmeans_accs(index, W, labels, sd, N):
    accs = []
    for i in range(N):
        A = W_to_A(W, epsilon=0, noise_sd=sd)
        SE = SpectralEmbedding(affinity="precomputed").fit(A)
        A_trans = SE.embedding_
        _, sup_acc = kmeans(A_trans, labels[index], print_out=False, return_acc=True)
        accs.append(sup_acc)
    return accs


def optimise_sd(weights, labels, N = 30, SD = np.arange(0., 1., .1)):
    W = weights.values.reshape(-1,1)
    accs = {}
    means = {}
    sds = {}
    for sd in SD:
        accs[sd] = get_kmeans_accs(weights.index, W, labels, sd, N)
        means[sd] = np.mean(accs[sd])
        sds[sd] = np.std(accs[sd])
    return accs, means, sds


def plot_distributions(input_dict, fname):
    for k, v in input_dict.items():
        if k == 0.: continue
        plt.hist(v, bins=10, alpha=.3, label=f"SD {round(k,1)}")
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.title("k-means accuracy on eigenmap by SD")
    plt.legend(loc="upper left")
    plt.savefig("kmeans_accuracy_by_sd.png", bbox_inches="tight")



def main(SELECTION=None):
    # SELECTION = ["diaphragmatic hernia", "pleural effusion", "pneumothorax", "pulmonary nodule"]
    # SELECTION = ["diaphragmatic hernia", "pneumothorax", "pulmonary nodule"]
    # SELECTION = ["pneumonia", "pleural effusion"]

    assert len(argv) > 1
    dir_path = Path(argv[1])
    labels = pd.read_csv(dir_path / "labels.csv")
    labels = labels.set_index("hadm_id")["long_title"]

    images = pd.read_csv(dir_path / "images.csv")
    images = images.set_index("dicom_id")[["hadm_id", "study_path"]]

    labevents = pd.read_csv(dir_path / "labevents.csv")
    lab_hadm_id = labevents["dicom_id"].apply(lambda id: match_hadm_id(images, id))
    labevents = labevents[lab_hadm_id.notna()].drop(columns="dicom_id")
    labevents.index = lab_hadm_id[lab_hadm_id.notna()].astype(int)
    labevents = labevents.groupby(level=0).mean()
    labevents = normalise(labevents)


    vitalsigns = pd.read_csv(dir_path / "vital_signs.csv")
    chart_hadm_id = vitalsigns["dicom_id"].apply(lambda id: match_hadm_id(images, id))
    vitalsigns = vitalsigns[chart_hadm_id.notna()].drop(columns="dicom_id")
    vitalsigns.index = chart_hadm_id[chart_hadm_id.notna()].astype(int)
    vitalsigns = vitalsigns.groupby(level=0).mean()
    vitalsigns = normalise(vitalsigns)

    if SELECTION is not None:
        selection_idx = labels.isin(SELECTION)
        labels = labels[selection_idx]
        images = images[images["hadm_id"].isin(selection_idx.index)]
        labevents = labevents[labevents.index.isin(selection_idx.index)]
        vitalsigns = vitalsigns[vitalsigns.index.isin(selection_idx.index)]


    classes = labels.unique()
    assert len(classes) > 1
    binary = len(classes) == 2


    counts, texts = count_split(labels, images)


    # def plot_pairs(n_classes, *labels_, SD=0.2):
    #     if n_classes == 2: label, other_label = labels_
    #     elif n_classes == 3: label, other_label, third_label = labels_
    #     elif n_classes == 4: label, other_label, third_label, fourth_label = labels_
    #     idx__ = (labels == label) | (labels == other_label)

    #     if n_classes == 2:
    #         idx__ = idx__[idx__.index.isin(texts.index)]
    #         idx__ = idx__[idx__].index
    #         diff = get_binary_diff(counts[label], counts[other_label])
    #         words = diff.index.values
    #         W = get_text_weights(texts.loc[idx__], diff, words, out_format="sum").values.reshape(-1,1)
    #         plot(
    #             SpectralEmbedding(affinity="precomputed").fit_transform(
    #                 W_to_A(
    #                     W,
    #                     epsilon=0,
    #                     noise_sd=SD
    #                 )
    #             ),
    #             labels.loc[idx__].values,
    #             fname = f"experiments/{label} - {other_label}.png",
    #             title = f"{label} - {other_label}, SD = {SD}",
    #             alpha = 0.3,
    #         )
    #     elif n_classes == 3:
    #         idx___ = idx__ | (labels == third_label)
    #         idx___ = idx___[idx___.index.isin(texts.index)]
    #         idx___ = idx___[idx___].index
    #         diff = get_multiclass_diff({k: v for k, v in counts.items() if k in [label, other_label, third_label]})
    #         words = diff.index.values
    #         W = get_multiclass_weights(
    #             texts.loc[idx___],
    #             diff
    #         ).values
    #         plot(
    #             SpectralEmbedding(affinity="precomputed").fit_transform(
    #                 W_to_A(
    #                     W,
    #                     epsilon=0,
    #                     noise_sd=SD
    #                 )
    #             ),
    #             labels.loc[idx___].values,
    #             fname = f"experiments/{label} - {other_label} - {third_label}.png",
    #             title = f"{label} - {other_label} - {third_label}, SD = {SD}",
    #             alpha = 0.3,
    #         )
    #     elif n_classes == 4:
    #         idx___ = idx__ | (labels == third_label) | (labels == fourth_label)
    #         idx___ = idx___[idx___.index.isin(texts.index)]
    #         idx___ = idx___[idx___].index
    #         diff = get_multiclass_diff({k: v for k, v in counts.items() if k in [label, other_label, third_label, fourth_label]})
    #         words = diff.index.values
    #         W = get_multiclass_weights(
    #             texts.loc[idx___],
    #             diff
    #         ).values
    #         plot(
    #             SpectralEmbedding(affinity="precomputed").fit_transform(
    #                 W_to_A(
    #                     W,
    #                     epsilon=0,
    #                     noise_sd=SD
    #                 )
    #             ),
    #             labels.loc[idx___].values,
    #             fname = f"experiments/{label} - {other_label} - {third_label} - {fourth_label}.png",
    #             title = f"{label} - {other_label} - {third_label} - {fourth_label}, SD = {SD}",
    #             alpha = 0.3,
    #         )
    #     else:
    #         raise NotImplementedError

    # for i, label in enumerate(classes):
    #     for j, other_label in enumerate(classes):
    #         if j <= i: continue

    #         # plot_pairs(2, label, other_label)
    #         for k, third_label in enumerate(classes):
    #             if k <= j: continue
    #             # plot_pairs(3, label, other_label, third_label)
    #             for l, fourth_label in enumerate(classes):
    #                 if l <= k: continue
    #                 plot_pairs(4, label, other_label, third_label, fourth_label)
    

    idx_ = texts.index.unique()
    if binary:
        # Assuming binary differences - Supervised
        diff = get_binary_diff(counts[classes[0]], counts[classes[1]])
        words = diff.index.values
    elif not binary:
        diff = get_multiclass_diff(counts)
        words = diff.index.values



    # Supervised

    # if binary:
    #    weights = get_text_weights(texts, diff, words, out_format="sum")
    # elif not binary:
    #    weights = get_multiclass_weights(texts, diff)
    # W = weights.values.reshape(-1,1)
    # SD = 0.4
    # A = W_to_A(W, epsilon=0, noise_sd=SD)
    # no_noise_A = W_to_A(W, epsilon=0, noise_sd=0.)
    # SE = SpectralEmbedding(affinity="precomputed").fit(A)
    # A_trans = SE.embedding_
    # # sup_acc = get_kmeans_accs(idx_, W, labels, .4, 30)
    # # _, sup_no_noise_acc = kmeans(no_noise_A, labels[idx_], return_acc=True)
    # plot(
    #     no_noise_A,
    #     labels[idx_].values,
    #     fname = f"fully_labelled_eigenmap_no_noise.png",
    #     title = f"Spectral embedding, labelled",
    #     alpha = 0.3,
    # )
    # np.save(f"{str(dir_path)}/graph_supervised_nonoise.npy", SE.affinity_matrix_, allow_pickle=True)
    # # ~70% accuracy
    # print(kmeans(A_trans, labels[idx_], print_out=False, return_acc=True)[-1])
    # plot(
    #     A_trans,
    #     labels[idx_].values,
    #     fname = f"fully_labelled_eigenmap_noise.png",
    #     title = f"Spectral embedding, SD = {SD}",
    #     alpha = 0.3,
    # )
    # np.save(f"{str(dir_path)}/graph_supervised.npy", SE.affinity_matrix_, allow_pickle=True)
    # print()




    # Unsupervised

    N = 50
    semi_sup_idx = np.concatenate(tuple(
        np.random.choice(np.where(labels[idx_] == cls_)[0], size=int(N), replace=False) for cls_ in classes
    ))

    if binary:
        unsup_pred, unsup_weights = unsupervised_diff(
            labels[idx_],
            texts.loc[idx_],
            words,
            images,
            PCA = False,
        )
        unsup_W = unsup_weights.values.reshape(-1,1)
    elif not binary:
        unsup_pred, unsup_weights = unsupervised_multiclass_diff(
            labels[idx_],
            texts.loc[idx_],
            idx_,
            words,
            images,
            PCA = False,
            # clustering_method="spectral",
            semi_supervised_index = semi_sup_idx,
        )

        if type(unsup_weights) == np.ndarray:
            assert unsup_weights.shape[0] == unsup_weights.shape[1], "not graph output"
            unsup_A = unsup_weights
            del unsup_weights

            pd.Series(idx_).to_csv(f"experiments/graph_hadm.csv", header=False, index=False)
            np.save(f"experiments/graph_unsupervised.npy", unsup_A, allow_pickle=True)
            plot(
                SpectralEmbedding(n_components=len(classes), affinity="precomputed").fit_transform(unsup_A),
                labels[idx_].values,
                f"experiments/rbf_kernel.png",
                title="RBF kernel"
            )
            affinity_plot(
                unsup_A,
                labels[idx_].values,
                f"experiments/rbf_kernel_affinity_mat.png",
                title = "RBF kernel affinity matrix",
                log = True
            )
            print()
        else:
            unsup_W = unsup_weights.values
            unsup_A = None

            unsup_SE = SpectralEmbedding(n_components=len(classes), affinity="precomputed").fit(W_to_A(unsup_W, epsilon=0., noise_sd=0.))
            unsup_A_trans = unsup_SE.embedding_

            pd.Series(unsup_weights.index).to_csv(f"experiments/graph_hadm.csv", header=False, index=False)
            np.save(f"experiments/graph_unsupervised.npy", unsup_SE.affinity_matrix_, allow_pickle=True)
            plot(
                unsup_A_trans,
                labels[idx_].values,
                f"experiments/kmeans_kernel.png",
                title="RBF kernel"
            )
            affinity_plot(
                unsup_SE.affinity_matrix_,
                labels[idx_].values,
                f"experiments/kmeans_kernel_affinity_mat.png",
                title = "RBF kernel affinity matrix",
                log = False
            )
            print()

        
    # unsup_acc, _ = spectral_clustering_accuracy(
    #     idx_,
    #     W_to_A(unsup_W, epsilon=.5, noise_sd=.4),
    #     labels[idx_],
    #     affinity = 'precomputed',
    # )
    # print(f"Unsupervised spectral clustering accuracy : {unsup_acc}")

    if unsup_A is None:
        unsup_SD = 0.
        unsup_A = W_to_A(unsup_W, epsilon=0, noise_sd=unsup_SD, transform="knn")
    unsup_SE = SpectralEmbedding(n_components=len(classes), affinity="precomputed").fit(unsup_A)
    unsup_A_trans = unsup_SE.embedding_
    # unsup_acc = get_kmeans_accs(idx_, unsup_W, labels, .4, 30)
    # ~70% accuracy
    print(f"Clustering on affinity matrix accuracy: {kmeans(unsup_A_trans, labels[idx_], print_out=False, return_acc=True)[-1]}")
    if SELECTION is not None:
        plot(
            unsup_A_trans,
            labels[idx_].values,
            fname = f"experiments/{SELECTION}, unlabelled graph.png",
            title = f"Spectral embedding, unlabelled, SD = {unsup_SD}",
            alpha = 0.3,
        )
        np.save(f"{str(dir_path)}/{SELECTION}, graph_unsupervised.npy", unsup_SE.affinity_matrix_, allow_pickle=True)
        pd.Series(unsup_weights.index).to_csv(f"{str(dir_path)}/{SELECTION}, graph_hadm.csv", header=False, index=False)
    else:
        plot(
            unsup_A_trans,
            labels[idx_].values,
            fname = f"experiments/unlabelled graph.png",
            title = f"Spectral embedding, unlabelled, SD = {unsup_SD}",
            alpha = 0.3,
        )
        np.save(f"{str(dir_path)}/graph_unsupervised.npy", unsup_SE.affinity_matrix_, allow_pickle=True)
        pd.Series(unsup_weights.index).to_csv(f"{str(dir_path)}/graph_hadm.csv", header=False, index=False)
    print()



    # Semi-supervised

    # N1, N2 = 100, 500
    # semi_sup_idx_1 = np.concatenate((
    #     np.random.choice(np.where(labels[idx_] == classes[0])[0], size=int(N1/2), replace=False),
    #     np.random.choice(np.where(labels[idx_] == classes[1])[0], size=int(N1/2), replace=False),
    # ))
    # semi_sup_idx_2 = np.concatenate((
    #     np.random.choice(np.where(labels[idx_] == classes[0])[0], size=int(N2/2), replace=False),
    #     np.random.choice(np.where(labels[idx_] == classes[1])[0], size=int(N2/2), replace=False),
    # ))
    # semi_pred_1, semi_W_1 = unsupervised_diff(
    #     labels[idx_],
    #     texts.loc[idx_],
    #     idx_,
    #     words,
    #     images,
    #     semi_supervised_index = semi_sup_idx_1,
    #     PCA = False,
    #     clustering_method="semi_supervised"
    # )
    # semisup_SD_1 = 0.4
    # semisup_A_1 = W_to_A(semi_W_1, epsilon=0, noise_sd=semisup_SD_1)
    # semisup_SE_1 = SpectralEmbedding(affinity="precomputed").fit(semisup_A_1)
    # semisup_A_trans_1 = semisup_SE_1.embedding_
    # # N1_acc = get_kmeans_accs(idx_, semi_W_1, labels, .4, 30)
    # # ~70% accuracy
    # print(kmeans(semisup_A_trans_1, labels[idx_], print_out=False, return_acc=True)[-1])
    # plot(
    #     semisup_A_trans_1,
    #     labels[idx_].values,
    #     fname = f"N1_labelled_eigenmap.png",
    #     title = f"Spectral embedding - N1, SD = {semisup_SD_1}",
    #     alpha = 0.3,
    # )
    # np.save(f"{str(dir_path)}/graph_semisup_N1.npy", semisup_SE_1.affinity_matrix_, allow_pickle=True)
    # print()

    # semi_pred_2, semi_W_2 = unsupervised_diff(
    #     labels[idx_],
    #     texts.loc[idx_],
    #     idx_,
    #     words,
    #     images,
    #     semi_supervised_index = semi_sup_idx_2,
    #     PCA = False,
    #     clustering_method="semi_supervised"
    # )

    # semisup_SD_2 = 0.4
    # semisup_A_2 = W_to_A(semi_W_2, epsilon=0, noise_sd=semisup_SD_2)
    # semisup_SE_2 = SpectralEmbedding(affinity="precomputed").fit(semisup_A_2)
    # semisup_A_trans_2 = semisup_SE_2.embedding_
    # # N2_acc = get_kmeans_accs(idx_, semi_W_2, labels, .4, 30)
    # # ~70% accuracy
    # print(kmeans(semisup_A_trans_2, labels[idx_], print_out=False, return_acc=True)[-1])
    # plot(
    #     semisup_A_trans_2,
    #     labels[idx_].values,
    #     fname = f"N2_labelled_eigenmap.png",
    #     title = f"Spectral embedding - N2, SD = {semisup_SD_2}",
    #     alpha = 0.3,
    # )
    # np.save(f"{str(dir_path)}/graph_semisup_N2.npy", semisup_SE_2.affinity_matrix_, allow_pickle=True)
    # print()



    # print(f"Max k-means accuracies:\nSup., noise: {max(sup_acc)}\nUnsup.: {max(unsup_acc)}\nSemisup., N1: {max(N1_acc)}\nSemisup., N2: {max(N2_acc)}")
    # print(f"Mean k-means accuracies:\nSup., noise: {np.mean(sup_acc)}\nUnsup.: {np.mean(unsup_acc)}\nSemisup., N1: {np.mean(N1_acc)}\nSemisup., N2: {np.mean(N2_acc)}")
    # print(f"SD k-means accuracies:\nSup., noise: {np.std(sup_acc)}\nUnsup.: {np.std(unsup_acc)}\nSemisup., N1: {np.std(N1_acc)}\nSemisup., N2: {np.std(N2_acc)}")


    # for cl, diff_ in {classes[0]: diff[diff > 0], classes[1]: diff[diff < 0].sort_values()}.items():
    #     wordcloudify(" ".join(diff_.index), f"wordcloud-{cl}.png", title = cl)




if __name__ == "__main__":
    main()
