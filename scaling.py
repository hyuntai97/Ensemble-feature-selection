from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np 

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier


# 표준화
def standardize_select(standardize,x_train = None, x_val = None):
    x_train_index = x_train.index
    x_val_index = x_val.index

    if standardize == 'Standard':
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_new = scaler.transform(x_train)
        x_val_new = scaler.transform(x_val)
        x_train = pd.DataFrame(x_train_new, columns = x_train.columns, index = x_train_index)
        x_val = pd.DataFrame(x_val_new, columns = x_val.columns, index = x_val_index)

    if standardize == "Robust":     
        scaler = RobustScaler()
        scaler.fit(x_train)
        x_train_new = scaler.transform(x_train)
        x_val_new = scaler.transform(x_val)
        x_train = pd.DataFrame(x_train_new, columns = x_train.columns, index = x_train_index)
        x_val = pd.DataFrame(x_val_new, columns = x_val.columns, index = x_val_index)

    return x_train , x_val

# 정규화 
def normalize_select(normalize,x_train = None , x_val= None):
    x_train_index = x_train.index
    x_val_index = x_val.index

    if normalize == "Normalizer":   
      scaler = Normalizer()
      scaler.fit(x_train)
      x_train_new = scaler.transform(x_train)
      x_val_new = scaler.transform(x_val)
      x_train = pd.DataFrame (x_train_new, columns= x_train.columns , index = x_train_index)
      x_val = pd.DataFrame (x_val_new , columns= x_val.columns , index = x_val_index)

    if normalize == "Minmax":       
      scaler = MinMaxScaler()
      scaler.fit(x_train)
      x_train_new = scaler.transform(x_train)
      x_val_new = scaler.transform(x_val)
      x_train = pd.DataFrame (x_train_new, columns= x_train.columns , index = x_train_index)
      x_val = pd.DataFrame (x_val_new , columns= x_val.columns , index = x_val_index)



    return x_train, x_val