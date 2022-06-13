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

Wait for access (this might take a few days).

### Getting access in Google Cloud

1. Access MIMIC-IV and MIMIC-CXR in Google Cloud : <https://mimic.mit.edu/docs/gettingstarted/cloud/>
2. Create a project at : <https://console.cloud.google.com/>


## Google Colab link (`create_db.ipynb`)

<https://colab.research.google.com/drive/1tkHdUwXzzp8UD3vQT9T__Zlhl0VFdzML?usp=sharing>


