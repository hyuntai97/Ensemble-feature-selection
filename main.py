import argparse
import json 
import pickle

import os
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')
import shap
from tqdm import tqdm, notebook

import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import model_selection

from preprocess import feature_select, sample_generate , replace_outlier
from scaling import standardize_select, normalize_select
from load import dataloader

from sklearn.preprocessing import StandardScaler

if __name__=='__main__':
    # config
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline")
    parser.add_argument('--seed', type=int, default=1, help='Set seed')
    parser.add_argument('--datadir', type=str, default='./data', help='Set data directory')
    parser.add_argument('--logdir', type=str, default='./logs', help='Set log directory')
    parser.add_argument('--savedir', type=str, default='./save', help='Set arguments directory')
    parser.add_argument('--paramdir', type=str, default='./params', help='Set parameters directory')
    parser.add_argument('--kfold', type=int, default=5, help='Number of cross validation')
    parser.add_argument('--fsmethod', type=str, default ='SelectKBest_f', help ='Choice feature select method')
    parser.add_argument('--featurenum', type=int, default =100, help ='Set feature selected number')
    parser.add_argument('--rfestep', type=float, default = 0.005, help ='Set rfe step')
    parser.add_argument('--standardize', type=str, default = 'Standard', help ='Choice standardize method')
    parser.add_argument('--normalize', type=str, default = 'Normalizer', help ='Choice normalize method')
    parser.add_argument('--dataload', type=int, default = 1, help = 'Choice data loading method')
    parser.add_argument('--n_estimators', type=int, default=100, help = 'Set The number of trees in the forest')
    parser.add_argument('--colsdir', type=str, default='./cols', help='Set Directory in which the selected column is stored')
    parser.add_argument('--n_resampling', type=int, default=10, help='Set resampling count')
    args = parser.parse_args()

models = {
   #"Ran":RandomForestClassifier(),
   #"KNN":KNeighborsClassifier(),
   #"Log":LogisticRegression(),
   #"SVC":SVC(probability=True),
   #"Ada":AdaBoostClassifier(),
   #"GNB":GaussianNB(),
   "Bag":BaggingClassifier(),
   #"XGB":XGBClassifier(),
   #"LGB":LGBMClassifier()
}

for model_name in models.keys():
  model_ = model_name

# save argument
json.dump(vars(args), open(os.path.join(args.savedir,f'arguments{args.seed}_{args.fsmethod}_{model_}.json'),'w'))

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


results_val = []
selected_cols_dict = {}
skf = StratifiedKFold(n_splits = args.kfold, shuffle = True, random_state = args.seed)

i = 0
X, y_target = dataloader(args.datadir, args.dataload)

for train_idx,val_idx in skf.split(X,y_target):
  for model_name, model in tqdm(models.items()):

    x_train = X.iloc[train_idx, :]
    y_train = y_target.iloc[train_idx]
    
    x_val = X.iloc[val_idx, :]
    y_val = y_target.iloc[val_idx]

    # feature standardize
    #x_train, x_val = standardize_select(args.standardize,x_train, x_val)

    # feature select
    x_train, x_val, selected_columns = feature_select(args.fsmethod ,args.featurenum , args.rfestep, args.seed, args.n_estimators , args.n_resampling, x_train, y_train, x_val)
    
    # outlier replace
    #x_train, x_val = replace_outlier(x_train, x_val)

    # feature normalize
    #x_train, x_val = normalize_select(args.normalize,x_train, x_val)

    # sample generate
    #x_train, y_train = sample_generate(x_train, y_train)

    # model training
    model = model.fit(x_train, y_train)

    # validation score
    y_pred_val = model.predict(x_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    results_val.append([i,accuracy_val, model_name])
    
    # selected columns
    selected_cols_dict[i] = selected_columns
  
  i += 1

df_cols = pd.DataFrame(selected_cols_dict)
pickle.dump(df_cols, open(os.path.join(args.colsdir, f'selected_cols{args.seed}_{args.fsmethod}_{model_}.pkl'),'wb'))

df_results = pd.DataFrame(data = results_val, columns = ['iter','val_acc','model'])
pickle.dump(df_results, open(os.path.join(args.logdir, f'validation_results{args.seed}_{args.fsmethod}_{model_}.pkl'),'wb'))

