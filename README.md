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

## Preprocessing and combining the data

...





