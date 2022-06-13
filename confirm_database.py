import pandas as pd
import numpy as np
import sys
from pathlib import Path

base_path = Path.cwd() / "physionet.org" / "files" / "mimic-cxr" / "2.0.0"
paths = pd.read_csv(sys.argv[1], header=None)[0]
path_test = paths.apply(lambda x: (base_path / Path(x)).exists())
if path_test.all():
    print(f"All {path_test.sum()} paths in paths.txt correctly identified")
else:
    print(f"{path_test.sum()} / {len(path_test)} ({100*path_test.sum()/len(path_test)}%) of the paths in paths.txt correctly identified")
    error_paths = paths[path_test == False]
    np.savetxt('.err_paths.txt', error_paths.unique(), '%s')
