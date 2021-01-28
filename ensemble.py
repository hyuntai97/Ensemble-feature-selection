from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression, f_classif
from sklearn import preprocessing

import pandas as pd
import numpy as np 
from sklearn.ensemble import ExtraTreesClassifier
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
from sklearn.tree import DecisionTreeClassifier

def ensemble_model(featurenum, seed, x_train = None, y_train = None):
    rank_lst = []
    fs_model_tree = [DecisionTreeClassifier(random_state = seed), RandomForestClassifier(random_state = seed), ExtraTreesClassifier(random_state = seed)]
    fs_model_kb = [SelectKBest(chi2, k = 'all'), SelectKBest(f_classif, k = 'all')]

    for fs in fs_model_tree:
        model = fs
        model.fit(x_train, y_train)
        importances = model.feature_importances_
        importances_series = pd.Series(importances)
        rank = importances_series.rank(ascending = False, method = 'min')
        rank_lst.append(rank)

    for fs in fs_model_kb:
        model = fs
        model.fit(x_train, y_train)
        importances = model.scores_
        importances_series = pd.Series(importances)
        rank = importances_series.rank(ascending = False, method = 'min')
        rank_lst.append(rank)

    rank_df = pd.concat(rank_lst, axis = 1)
    rank_df['mean'] = rank_df.mean(axis = 1)
    rank_sort = np.sort(rank_df['mean'])
            
    rank_high_index = []
    for i in rank_df.index:
        if rank_df.loc[i,'mean'] < rank_sort[featurenum]:
            rank_high_index.append(i)
        
    rank_high_idx = x_train.columns[rank_high_index]
    
    return rank_high_idx


    


       