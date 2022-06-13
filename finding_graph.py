from email import header
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
    r"[A-Z][A-Z]+": "",
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


# def weight_sentence(text, weights):
#     text = text.split(" ")
#     w = []
#     for t in text:
#         try:
#             w.append(weights[t.lower()])
#         except KeyError:
#             print("hmm..")
#             pass
#     return sum(w) / max(len(w), 1)
    

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


def plot(X, labels, fname, title = None, **sns_kwargs):
    plot = sns.scatterplot(
        x=X[:,0],
        y=X[:,1],
        hue=labels,
        **sns_kwargs
    )
    if title is not None: plot.set_title(title)
    plt.savefig(fname, bbox_inches='tight')
    plt.clf()


def spectral_clustering_accuracy(index, feature_matrix, labels, one_class = "None", **spectral_clustering_kwargs):
    pred = SpectralClustering(n_clusters = 2, **spectral_clustering_kwargs).fit_predict(feature_matrix)
    acc = max(
        ((labels[index] != one_class).values.astype(float) == pred).mean(),
        ((labels[index] == one_class).values.astype(float) == pred).mean()
    )
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


def _get_acc(pred, labels, set_pos = None):
    if set_pos is None:
        classes = labels.unique()
        acc = {
            classes[0]: ((labels.values == classes[0]).astype(float) == pred).mean(),
            classes[1]: ((labels.values == classes[1]).astype(float) == pred).mean()
        }
        return acc, max(acc, key=acc.get)
    else:
        return ((labels.values == set_pos).astype(float) == pred).mean()



def kmeans(X, y, print_out = True, return_acc = False, **kmeans_kwargs):
    from sklearn.cluster import KMeans
    pred = KMeans(n_clusters = 2, **kmeans_kwargs).fit_predict(X)
    acc, set_pos = _get_acc(pred, y)
    if print_out: print(f"Unsupervised k-means accuracy : {acc[set_pos]}")
    if not return_acc: return pred, set_pos
    else: return acc[set_pos]


def gmm(X, y, **gmm_kwargs):
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=2, **gmm_kwargs).fit(X)
    pred = gmm.predict(X)
    acc, set_pos = _get_acc(pred, y)
    print(f"Unsupervised GMM accuracy : {acc[set_pos]}")
    return pred, set_pos


def selftrain(X, y, idx, **kwargs):
    from sklearn.svm import SVC
    from sklearn.semi_supervised import SelfTrainingClassifier
    set_pos = y.unique()[0]
    labels = y.copy()
    labels[labels != set_pos] = 0
    labels[labels == set_pos] = 1
    labels.iloc[list(set(range(len(labels))) - set(idx))] = -1

    svc = SVC(probability=True, kernel="linear")
    lp = SelfTrainingClassifier(svc, **kwargs).fit(X.values, labels.values.astype(float))
    pred = lp.predict(X)
    acc = _get_acc(pred, y, set_pos)
    print(f"Self-training accuracy : {acc}")
    return pred, set_pos


def pca(features, dim, **kwargs):
    from sklearn.decomposition import PCA
    return PCA(n_components=dim, **kwargs).fit_transform(features)


def unsupervised_diff(
    labels,
    texts,
    index,
    words,
    images,
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
        pred, set_pos = kmeans(features, labels, **clustering_kwargs)
    elif clustering_method == "gmm":
        pred, set_pos = gmm(features, labels, **clustering_kwargs)
    elif clustering_method == "semi_supervised":
        assert semi_supervised_index is not None
        pred, set_pos = selftrain(features, labels, semi_supervised_index, **clustering_kwargs)

    pred = pred.astype("object")
    idx_pos = pred >= .5
    pred[pred < .5] = cl[cl != set_pos][0]
    pred[idx_pos] = set_pos
    counts, _ = count_split(pd.Series(pred, index = index), images)
    diff = get_binary_diff(counts[cl[0]], counts[cl[1]])
    weights = get_text_weights(texts.loc[index], diff, words)
    return pred, weights.values.reshape(-1,1)


def match_hadm_id(df, idx):
    try:
        out = df.loc[idx]["hadm_id"]
        if type(out) == pd.Series:
            assert len(out.unique()) == 1
            out = out.iloc[0]
        return out
    except KeyError:
        pass


def W_to_A(W, epsilon = 0., noise_sd = 0.):
    A = W @ W.T
    A = (A - A.mean()) / A.std()
    if epsilon > 0.:
        A[A < epsilon] = 0.
    if noise_sd > 0.:
        A = A + np.random.normal(scale=noise_sd, size=A.shape)
    return A - A.min()


def get_kmeans_accs(index, W, labels, sd, N):
    accs = []
    for i in range(N):
        A = W_to_A(W, epsilon=0, noise_sd=sd)
        SE = SpectralEmbedding(affinity="precomputed").fit(A)
        A_trans = SE.embedding_
        sup_acc = kmeans(A_trans, labels[index], print_out=False, return_acc=True)
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



def main():
    dir_path = Path(argv[1])
    labels = pd.read_csv(dir_path / "labels.csv")
    labels = labels.set_index("hadm_id")["long_title"]

    images = pd.read_csv(dir_path / "images.csv")
    images = images.set_index("dicom_id")[["hadm_id", "study_path"]]

    # hadm_order = pd.read_csv(dir_path / "hadm_order.csv", header=None)[0]

    labevents = pd.read_csv(dir_path / "labevents.csv")
    lab_hadm_id = labevents["dicom_id"].apply(lambda id: match_hadm_id(images, id))
    labevents = labevents[lab_hadm_id.notna()].drop(columns="dicom_id")
    labevents.index = lab_hadm_id[lab_hadm_id.notna()].astype(int)
    labevents = labevents.groupby(level=0).mean()
    labevents = normalise(labevents)
    # lab_embedding = SpectralEmbedding(n_components=1, affinity="nearest_neighbors").fit_transform(labevents.values)
    lab_embedding = pca(labevents.values, 1)


    vitalsigns = pd.read_csv(dir_path / "vitalsigns.csv")
    chart_hadm_id = vitalsigns["dicom_id"].apply(lambda id: match_hadm_id(images, id))
    vitalsigns = vitalsigns[chart_hadm_id.notna()].drop(columns="dicom_id")
    vitalsigns.index = chart_hadm_id[chart_hadm_id.notna()].astype(int)
    vitalsigns = vitalsigns.groupby(level=0).mean()
    vitalsigns = normalise(vitalsigns)
    # chart_embedding = SpectralEmbedding(n_components=1, affinity="nearest_neighbors").fit_transform(vitalsigns.values)
    chart_embedding = pca(vitalsigns.values, 1)

    classes = labels.unique()


    counts, texts = count_split(labels, images)


    # Assuming class differences - Supervised
    diff = get_binary_diff(counts[classes[0]], counts[classes[1]])
    words = diff.index.values
    cl_diff = {
        classes[0]: diff[diff > 0],
        classes[1]: diff[diff < 0].sort_values()
    }
    weights = get_text_weights(texts, diff, words, out_format="sum")
    # keep_ratio = .9
    # importance = weights.sum().abs().sort_values(ascending=False).cumsum() / weights.sum().abs().sum()
    # imp_cols = importance.iloc[:(importance - keep_ratio).abs().argmin()].index
    # print(f"Filtered to {len(imp_cols)}/{len(weights.columns)} ({100 * len(imp_cols)/len(weights.columns)}%) columns.")
    # weights = weights.loc[:,imp_cols]

    # accs, means, sds = optimise_sd(weights, labels)
    # plot_distributions(accs, f"distributions_sd_fully-labeled.png")

    W = weights.values.reshape(-1,1)
    idx_ = weights.index
    SD = 0.4
    A = W_to_A(W, epsilon=0, noise_sd=SD)
    no_noise_A = W_to_A(W, epsilon=0, noise_sd=0.)
    SE = SpectralEmbedding(affinity="precomputed").fit(A)
    A_trans = SE.embedding_
    # sup_acc = get_kmeans_accs(idx_, W, labels, .4, 30)
    # sup_no_noise_acc = kmeans(no_noise_A, labels[idx_], return_acc=True)
    plot(
        no_noise_A,
        labels[idx_].values,
        fname = f"fully_labelled_eigenmap_no_noise.png",
        title = f"Spectral embedding, labelled",
        alpha = 0.3,
    )
    np.save(f"{str(dir_path)}/graph_supervised_nonoise.npy", SE.affinity_matrix_, allow_pickle=True)
    # ~70% accuracy
    print(kmeans(A_trans, labels[idx_], print_out=False, return_acc=True))
    plot(
        A_trans,
        labels[idx_].values,
        fname = f"fully_labelled_eigenmap_noise.png",
        title = f"Spectral embedding, SD = {SD}",
        alpha = 0.3,
    )
    np.save(f"{str(dir_path)}/graph_supervised.npy", SE.affinity_matrix_, allow_pickle=True)
    print()




    # Unsupervised
    unsup_pred, unsup_W = unsupervised_diff(
        labels[idx_],
        texts.loc[idx_],
        idx_,
        words,
        images,
        PCA = False,
    )
    # unsup_acc, _ = spectral_clustering_accuracy(
    #     idx_,
    #     W_to_A(unsup_W, epsilon=.5, noise_sd=.4),
    #     labels[idx_],
    #     affinity = 'precomputed',
    # )
    # print(f"Unsupervised spectral clustering accuracy : {unsup_acc}")

    
    unsup_SD = 0.4
    unsup_A = W_to_A(unsup_W, epsilon=0, noise_sd=unsup_SD)
    unsup_SE = SpectralEmbedding(affinity="precomputed").fit(unsup_A)
    unsup_A_trans = unsup_SE.embedding_
    # unsup_acc = get_kmeans_accs(idx_, unsup_W, labels, .4, 30)
    # ~70% accuracy
    print(kmeans(unsup_A_trans, labels[idx_], print_out=False, return_acc=True))
    plot(
        unsup_A_trans,
        labels[idx_].values,
        fname = f"unlabelled_eigenmap.png",
        title = f"Spectral embedding, unlabelled, SD = {unsup_SD}",
        alpha = 0.3,
    )
    np.save(f"{str(dir_path)}/graph_unsupervised.npy", unsup_SE.affinity_matrix_, allow_pickle=True)
    print()



    # Semi-supervised
    N1, N2 = 100, 500
    semi_sup_idx_1 = np.concatenate((
        np.random.choice(np.where(labels[idx_] == classes[0])[0], size=int(N1/2), replace=False),
        np.random.choice(np.where(labels[idx_] == classes[1])[0], size=int(N1/2), replace=False),
    ))
    semi_sup_idx_2 = np.concatenate((
        np.random.choice(np.where(labels[idx_] == classes[0])[0], size=int(N2/2), replace=False),
        np.random.choice(np.where(labels[idx_] == classes[1])[0], size=int(N2/2), replace=False),
    ))
    semi_pred_1, semi_W_1 = unsupervised_diff(
        labels[idx_],
        texts.loc[idx_],
        idx_,
        words,
        images,
        semi_supervised_index = semi_sup_idx_1,
        PCA = False,
        clustering_method="semi_supervised"
    )
    semisup_SD_1 = 0.4
    semisup_A_1 = W_to_A(semi_W_1, epsilon=0, noise_sd=semisup_SD_1)
    semisup_SE_1 = SpectralEmbedding(affinity="precomputed").fit(semisup_A_1)
    semisup_A_trans_1 = semisup_SE_1.embedding_
    # N1_acc = get_kmeans_accs(idx_, semi_W_1, labels, .4, 30)
    # ~70% accuracy
    print(kmeans(semisup_A_trans_1, labels[idx_], print_out=False, return_acc=True))
    plot(
        semisup_A_trans_1,
        labels[idx_].values,
        fname = f"N1_labelled_eigenmap.png",
        title = f"Spectral embedding - N1, SD = {semisup_SD_1}",
        alpha = 0.3,
    )
    np.save(f"{str(dir_path)}/graph_semisup_N1.npy", semisup_SE_1.affinity_matrix_, allow_pickle=True)
    print()

    semi_pred_2, semi_W_2 = unsupervised_diff(
        labels[idx_],
        texts.loc[idx_],
        idx_,
        words,
        images,
        semi_supervised_index = semi_sup_idx_2,
        PCA = False,
        clustering_method="semi_supervised"
    )

    semisup_SD_2 = 0.4
    semisup_A_2 = W_to_A(semi_W_2, epsilon=0, noise_sd=semisup_SD_2)
    semisup_SE_2 = SpectralEmbedding(affinity="precomputed").fit(semisup_A_2)
    semisup_A_trans_2 = semisup_SE_2.embedding_
    # N2_acc = get_kmeans_accs(idx_, semi_W_2, labels, .4, 30)
    # ~70% accuracy
    print(kmeans(semisup_A_trans_2, labels[idx_], print_out=False, return_acc=True))
    plot(
        semisup_A_trans_2,
        labels[idx_].values,
        fname = f"N2_labelled_eigenmap.png",
        title = f"Spectral embedding - N2, SD = {semisup_SD_2}",
        alpha = 0.3,
    )
    np.save(f"{str(dir_path)}/graph_semisup_N2.npy", semisup_SE_2.affinity_matrix_, allow_pickle=True)
    pd.Series(idx_).to_csv(f"{str(dir_path)}/graph_hadm.csv", header=False, index=False)
    print()



    # print(f"Max k-means accuracies:\nSup., noise: {max(sup_acc)}\nUnsup.: {max(unsup_acc)}\nSemisup., N1: {max(N1_acc)}\nSemisup., N2: {max(N2_acc)}")
    # print(f"Mean k-means accuracies:\nSup., noise: {np.mean(sup_acc)}\nUnsup.: {np.mean(unsup_acc)}\nSemisup., N1: {np.mean(N1_acc)}\nSemisup., N2: {np.mean(N2_acc)}")
    # print(f"SD k-means accuracies:\nSup., noise: {np.std(sup_acc)}\nUnsup.: {np.std(unsup_acc)}\nSemisup., N1: {np.std(N1_acc)}\nSemisup., N2: {np.std(N2_acc)}")
    print()

    # for cl, diff_ in cl_diff.items():
    #     wordcloudify(" ".join(diff_.index), f"wordcloud-{cl}.png", title = cl)




if __name__ == "__main__":
    main()
