import argparse
import json 
import pickle

import os
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')

from tqdm import tqdm_notebook

import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
    #parser.add_argument('--preprocess', type=int, default=1, help='Select preprocessing method')
    #parser.add_argument('--val_size', type=float, default=None, help='Set validation size')
    parser.add_argument('--kfold', type=int, default=5, help='Number of cross validation')
    #parser.add_argument('--params', type=str, default='default', help='Model hyperparmeter')
    #parser.add_argument('--modelname', type=str, 
    #                     choices=['OLS','Ridge','Lasso','ElasticNet','DT','RF','ADA','GT','SVM','KNN','LGB'],
    #                     help='Choice machine learning model')
    parser.add_argument('--fsmethod', type=str, default ='SelectKBest', help ='Choice feature select method')
    parser.add_argument('--featurenum', type=int, default =100, help ='Set feature selected number')
    parser.add_argument('--rfestep', type=int, default = 0.005, help ='Set rfe step')
    parser.add_argument('--standardize', type=str, default = 'Standard', help ='Choice standardize method')
    parser.add_argument('--normalize', type=str, default = 'Nomalizer', help ='Choice normalize method')
    
    args = parser.parse_args()

# save argument
json.dump(vars(args), open(os.path.join(args.savedir,'arguments.json'),'w'))

models = {
   "Ran":RandomForestClassifier(),
   #"KNN":KNeighborsClassifier(),
   #"Log":LogisticRegression(),
   #"SVC":SVC(probability=True),
   #"Ada":AdaBoostClassifier(),
   #"GNB":GaussianNB(),
   #"Bag":BaggingClassifier(),
   "XGB":XGBClassifier(),
   #"LGB":LGBMClassifier()
}


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


results_val = []

skf = StratifiedKFold(n_splits = args.kfold, shuffle = True, random_state = args.seed)

i = 0
X, y_target = dataloader(args.datadir)

for train_idx,val_idx in skf.split(X,y_target):
  for model_name, model in tqdm_notebook(models.items()):

    x_train = X.iloc[train_idx, :]
    y_train = y_target.iloc[train_idx]
    
    x_val = X.iloc[val_idx, :]
    y_val = y_target.iloc[val_idx]

    # feature standardize
    x_train, x_val = standardize_select(args.standardize,x_train, x_val)

    # feature select
    x_train, x_val = feature_select(args.fsmethod ,args.featurenum , args.rfestep, x_train, y_train, x_val)
    
    # outlier replace
    x_train, x_val = replace_outlier(x_train, x_val)

    # feature normalize
    x_train, x_val = normalize_select(args.normalize,x_train, x_val)


    # sample generate
    x_train, y_train = sample_generate(x_train, y_train)

    # model training
    model = model.fit(x_train, y_train)

    # validation score
    y_pred_val = model.predict(x_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    results_val.append([i,accuracy_val, model_name])
   
  
  i += 1

df_results = pd.DataFrame(data = results_val, columns = ['iter','val_acc','model'])
pickle.dump(df_results, open(os.path.join(args.savedir, f'validation_results.pkl'),'wb'))
