import pandas as pd
import numpy as np
import sys

file = pd.read_csv(sys.argv[1])
file = file["dicom_path"].append(file["study_path"])
paths = file.to_numpy().flatten()
np.savetxt('.paths.txt', np.unique(paths), '%s')
