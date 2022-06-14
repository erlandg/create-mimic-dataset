# Combining Databases

Combining the databases MIMIC-IV and MIMIC-CXR on common DICOM IDs. Requires access to both databases through Google Cloud.

## Accessing the databases

### Accessing the data

The MIMIC databases are publicly available but requires some certification to access.  

1. Register a CITI account selecting "Data or Specimens only Research" : <https://www.citiprogram.org/index.cfm?pageID=154&icat=0&ac=0>
2. Create a Physionet account <https://physionet.org/register/>
3. Apply for access to :
    * <https://physionet.org/content/mimiciv/2.0/>
    * <https://physionet.org/content/mimic-cxr/2.0.0/>

Wait for access to the databases (this might take a few days).

### Getting access in Google Cloud

1. Access MIMIC-IV and MIMIC-CXR in Google Cloud : <https://mimic.mit.edu/docs/gettingstarted/cloud/>
2. Create a project (<https://console.cloud.google.com/>) which is to be used in the Colab project in the following section.


# Creating the Graph Multi-modal Dataset

## Extracting raw data (`create_db.ipynb`)

Google Colab link : <https://colab.research.google.com/drive/1tkHdUwXzzp8UD3vQT9T__Zlhl0VFdzML?usp=sharing>

The above code should provide the following files:  
* `most_common_diagnoses.csv`  
    Lists the number of cases of the `N_diagnoses` most common unique diagnosis names. Diagnosis name is used rather than ICD codes as both ICD-9 and ICD-10 are used.
* `labels.csv`  
    Provides labels of `subject_id` and `hadm_id` and their associated labels.
* `hadm_diags.csv`  
    Provides BoW representations of patient diagnoses by `hadm_id`.
* `images.csv`  
    Associates `subject_id`, `hadm_id`, `stay_id`, and `dicom_id` with DICOM date and time, as well as paths to images and radiology reports.
* `vital_signs.csv`  
    Associates `subject_id`, `hadm_id`, `stay_id`, and `dicom_id` with vital signs properties (max, min, median, mean, first, last) of *heart_rate, sbp, dbp, mbp, sbp_ni, dbp_ni, mbp_ni, resp_rate, temperature, spo2, glucose*.
* `labevents.csv`  
    Associates `subject_id`, `hadm_id`, `stay_id`, and `dicom_id` with lab test properties (max, min, median, mean, first, last) of filtered lab results (names listed in the thesis).
    
### Downloading CXR data

Now, using the paths provided in `images.csv` we may download image and radiology data to local storage from <https://physionet.org/content/mimic-cxr/2.0.0/>. This is done locally — rather than in Colab — due to the storage space required.  

1. Running `python3 extract_table.py images.csv` will yield a .txt file of all download paths, *.paths.txt*.
2. [Optional] Run `python3 confirm_database.py .paths.txt` to find the paths, in *.err_paths.txt*, that do not already exist locally.
3. Create file `credentials` defined as such (using your physionet account details):  
  ```
  USER=<INSERT_USERNAME_HERE>  
  PASS=<INSERT_PASSWORD_HERE>  
  ```
4. Run `bash download.sh .err_paths.txt` or `bash download.sh .paths.txt` to download files from PhysioNet.org.

## Preprocessing and combining the data

### Preparation

The folder structure should be as follows:  

```
create-mimic-dataset
│   combine_data.py
│   finding_graph.py
│  
└───physionet.org
│   │
│   └───mimic-cxr
│       │   
│       └───2.0.0  
│           │   
│           └───...  
│           │   │  *image_path*.dcm  
│   
└───tables
│   │   images.csv
│   │   graph.npy
│   │   graph_hadm.csv
│   │   vital_signs.csv
│   │   labevents.csv
│   │   labels.csv
│   │   ...
│   
│   ...
```

### Extracting graph data

To extract graphs we run :
```bash
python3 finding_graph.py tables
```
yielding multiple graphs and one `graph_hadm.csv` associating the indices of the graph to their `hadm_id`. We rename the desired graph (for the unsupervised case, *./tables/graph_unsupervised.npy*) as `graph.npy`. The folder structure should now comply to the one previously described.

### Combining

Having files `vital_signs.csv`, `labevents.csv`, `graph.npy`, `graph_hadm.csv`, and the CXR images, we may pre-process and combine the graph multi-modal dataset by running (from folder *create-mimic-dataset/*) :  

```bash
python3 combine_data.py tables/images.csv tables/graph.npy tables/vital_signs.csv tables/labevents.csv tables/labels.csv
```

The above yields :
* *./tables/dataset.npz*
* *./tables/dataset_images.npz*
* *./tables/dataset_no_images.npz*  

Each containing elements
```
n_views: The number of views, V
labels: Integer labels. Shape (n,)  
graph: Dense affinity matrix. Shape (n, n)
view_0: Data for first view. Shape (n, ...)  
  .  
  .  
  .  
view_V: Data for view V. Shape (n, ...)  
```





