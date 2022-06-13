from os import replace
from numpy.lib.npyio import save
import pydicom as dicom
import pandas as pd
import numpy as np
import sys

from skimage.transform import resize
from pathlib import Path
from math import ceil

from utils.dicom_processing import preprocess_dicom
import cv2


EQUAL_LABEL_COUNT = True
SAVE_IMAGE_PICKLES = False


def preprocess_static_data(df, replace_zero=True, index=None):
    exceptions = ["subject_id", "hadm_id", "dicom_id"]
    vals = df.loc[:,~df.columns.isin(exceptions)]
    vals = (vals - vals.mean()) / vals.std()
    if replace_zero:
        vals = vals.fillna(0)
    df.loc[:,~df.columns.isin(exceptions)] = vals
    if index is not None:
        df.index = index
    return df


def get_graph(graph_fn, hadm_ids, graph_hadm = None):
    if graph_fn.split(".")[-1] == "csv":
        graph = pd.read_csv(graph_fn, index_col=0, header=0).loc[hadm_ids, hadm_ids.astype(str)].values
    elif graph_fn.split(".")[-1] == "npy":
        assert graph_hadm is not None
        assert hadm_ids.isin(graph_hadm).all()
        graph = np.load(graph_fn, allow_pickle = True)
        graph = sort_graph(graph, hadm_ids, graph_hadm)
    return graph


def sort_graph(graph, hadm_id_to, hadm_id_from):
    a = hadm_id_to.apply(lambda x: np.argwhere((x == hadm_id_from).values)[0,0]).values
    return graph[np.ix_(a, a)]



def get_labels(labels, images, dicom_ids):
    labels_combined = images.merge(labels[["hadm_id", "long_title"]], on='hadm_id')
    dicom_ids = dicom_ids[dicom_ids.isin(labels_combined.dicom_id)]
    labels_combined = labels_combined[labels_combined.dicom_id.isin(dicom_ids)].drop_duplicates(subset=["dicom_id"]).sort_values("dicom_id")
    classes = labels_combined.long_title.unique()
    return classes, labels_combined[["subject_id", "hadm_id", "stay_id", "dicom_id", "long_title"]], dicom_ids


def sort_dfs(dfs, ids, sort_by):
    return [df[df.dicom_id.isin(ids)].drop_duplicates(subset=[sort_by]).sort_values(sort_by) for df in dfs]


def import_images(image_df, batch_size = 500, try_import=False):
    import_req = True
    if Path("dicom_extract/complete_dicom.pickle").exists() and try_import:
        images = pd.read_pickle("dicom_extract/complete_dicom.pickle")
        import_req = False
        print("Imported complete DICOM dataset")

    if import_req:
        base_path = Path.cwd() / "physionet.org" / "files" / "mimic-cxr" / "2.0.0"
        images = pd.Series(index=image_df.index)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        for i in range(ceil(len(images)/batch_size)):
            idx_start = int((i)*batch_size)
            idx_end = min(int((i+1)*batch_size), len(images))
            partition_fname = f"dicom_extract/dicom_partition_{i}.pickle"
            idx = images.index[idx_start : idx_end]

            if Path(partition_fname).exists() and try_import:
                import_ = pd.read_pickle(partition_fname)
                if len(idx) == len(import_.index):
                    if (idx == import_.index).all():
                        images.loc[idx] = import_
                        print(f"Imported {partition_fname}: {100 * idx_end/len(images)}% finished")
                        continue
                
            # Image pre-processing
            images.loc[idx] = image_df.loc[idx].dicom_path.apply(lambda x: preprocess_dicom(
                base_path / Path(x),
                clahe,
                bits=8
            ))
            if SAVE_IMAGE_PICKLES: images.loc[idx].to_pickle(partition_fname)
            print("Extracted {} images: {:.2f}% finished".format(
                idx_end - idx_start,
                100 * idx_end / len(images)
            ))

        if SAVE_IMAGE_PICKLES: images.to_pickle("dicom_extract/complete_dicom.pickle")
    imported_index = images[~images.isna()].index
    
    return images, imported_index


def export_cols(fname):
    fname = ".".join(fname.split(".")[:-1])
    pd.Series(pd.read_csv(f"{fname}.csv", index_col=0).columns).to_csv(f"{fname}_cols.csv", header=False, index=False)



def main(fnames = sys.argv[1:]):
    # fnames : images - graph - [static_data ...] - labels
    dfs = [pd.read_csv(fnames[0])]
    init_N = len(dfs[0].dicom_id.unique())
    dicom_ids = dfs[0].dicom_id
    folder_name = "/".join(fnames[0].split("/")[:-1])
    print(f"Folder name: {folder_name}")

    graph_fn = fnames[1]
    try:
        graph_hadm = pd.read_csv(f"{folder_name}/graph_hadm.csv", header=None)[0]
        dicom_ids = dicom_ids[dicom_ids.isin(dfs[0][dfs[0]["hadm_id"].isin(graph_hadm)]["dicom_id"])]
    except FileNotFoundError:
        if graph_fn.split(".")[-1] == "npy":
            raise FileNotFoundError
        else:
            graph_hadm = None

    for i, fname in enumerate(fnames[2:-1]):
        # Static data tables
        dfs.append(pd.read_csv(fname))
        dicom_ids = dicom_ids[dicom_ids.isin(dfs[-1].dicom_id)]


    # Get classes, labels and ensure complete overlap
    classes, labels_combined, dicom_ids = get_labels(pd.read_csv(fnames[-1]), dfs[0], dicom_ids)


    if EQUAL_LABEL_COUNT:
        # Equal number of samples of each class
        sample_num = min(labels_combined[labels_combined.dicom_id.isin(dicom_ids)].long_title.value_counts())
        dicom_ids = pd.concat(
            [labels_combined[labels_combined.long_title == lt_].dicom_id.sample(n=sample_num,replace=False) for lt_ in labels_combined.long_title.unique()],
            axis=0
        )
        labels_combined = labels_combined[labels_combined.dicom_id.isin(dicom_ids)].drop_duplicates(subset=["dicom_id"]).sort_values("dicom_id")
        print(f"Sample size reduced from {init_N} to {len(np.unique(labels_combined.long_title))*sample_num}")

    # Ensure same order
    dfs = sort_dfs(dfs, dicom_ids, "dicom_id")

    assert len(np.unique([len(df) for df in dfs])) == 1, "The tables are not equal in length"
    print(f"""{dfs[0].shape[0]} overlapping samples. Label-wise : {
        tuple([f'{label_} : {len(labels_combined[labels_combined["long_title"] == label_])}' for label_ in labels_combined.long_title.unique()])
    }""")

    # Numeric labels
    labels_combined["label"] = 0
    for i, title in enumerate(classes):
        idx_ = labels_combined.long_title.str.contains(title)
        labels_combined["label"][idx_] = i
    numeric_labels = labels_combined["label"]
    numeric_labels.index = dfs[0].dicom_id
    assert not numeric_labels.isna().any(), "There should not be any unlabelled data points"
    dfs.append(numeric_labels)

    assert len(np.unique([len(df) for df in dfs])) == 1, "Not all hospital admissions are accounted for in the datasets"

    # Import images from paths
    _, imported_index = import_images(dfs[0])
    dfs[0] = dfs[0].loc[imported_index]
    dfs[0]["dicom"] = _
    del _
    dfs[0] = dfs[0].sort_values("dicom_id")

    dfs[-1] = dfs[-1].loc[dfs[0].dicom_id]
    dfs[:-1] = sort_dfs(dfs[:-1], dicom_ids, "dicom_id")
    for i in range(1,len(dfs)-1):
        dfs[i] = dfs[i].set_index("dicom_id")

    static_data = [preprocess_static_data(
        df.loc[dfs[0].dicom_id]
    ) for df in dfs[1:-1]]


    for df in static_data:
        assert (df.index.values == dfs[0].dicom_id.values).all(), "The order of datasets are not equal"
    assert (dfs[-1].index.values == dfs[0].dicom_id.values).all(), "The order of labels are not correct"
    assert len(np.unique([len(df) for df in [dfs[0], *static_data, dfs[-1]]])) == 1, "Not equal number of samples"

    graph = get_graph(graph_fn, dfs[0].hadm_id, graph_hadm)
    view_0 = np.stack(dfs[0].dicom.values)[:,np.newaxis,:,:]
    view_1 = static_data[0].values
    view_2 = static_data[-1].values

    dfs[0].hadm_id.to_csv(f"{folder_name}/hadm_order.csv", index=False, header=False)
    try:
        hadm_diags = pd.read_csv(f"{folder_name}/hadm_diags.csv", index_col=0)
        diagnoses = dfs[0].hadm_id.apply(lambda x: hadm_diags.loc[x])
        diagnoses.index = dfs[0].dicom_id
        diagnoses.to_csv(f"{folder_name}/diagnoses.csv")
    except FileNotFoundError:
        print(f"No file hadm_diags.csv found in {folder_name}. Diagnoses table not created. Use hadm_order.csv to match hadm_ids to the view order.")

    for fname in ["chartevents.csv", "labevents.csv"]:
        try:
            export_cols(f"{folder_name}/fname")
        except FileNotFoundError:
            pass

    print(f"Shapes: Graph: {graph.shape}, View 0: {view_0.shape}, View 1: {view_1.shape}, View 2: {view_2.shape}")
    # Put together .npz
    np.savez(
        f"{folder_name}/dataset.npz",
        n_views = 3,
        labels = dfs[-1].values,
        graph = (graph + graph.T)/2,
        view_0 = view_0,
        view_1 = view_1,
        view_2 = view_2
    )
    np.savez(
        f"{folder_name}/dataset_no_images.npz",
        n_views = 2,
        labels = dfs[-1].values,
        graph = (graph + graph.T)/2,
        view_0 = view_1,
        view_1 = view_2
    )
    np.savez(
        f"{folder_name}/dataset_images.npz",
        n_views = 1,
        labels = dfs[-1].values,
        graph = (graph + graph.T)/2,
        view_0 = view_0,
    )
    print()




if __name__ == "__main__":
    main()
