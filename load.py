import pandas as pd 
import numpy as np
from numpy import isnan
import os 



def dataloader(datadir):
    data = pd.read_csv(os.path.join(datadir, f'LUADLUSC_float32.tsv'),sep = '\t')
    data_copy = data.copy()
    data_copy.set_index('sample', inplace = True)
    X = data_copy.iloc[:,:-1]
    y_target = data_copy.iloc[:,-1]
    
    return X, y_target

