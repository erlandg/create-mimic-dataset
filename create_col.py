import pandas as pd
from sys import argv

assert len(argv) >= 3, "Requires two arguments: input and output name, e.g. python create_col.py chartevents.csv chartcols.csv."
assert argv[1] != argv[2], "Won't override data file."
pd.Series(pd.read_csv(argv[1], index_col=0, header=0).columns).to_csv(argv[2], header=None, index=False)
