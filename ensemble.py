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

def ensemble_model(x_train = None, y_train = None):
    dt = DecisionTreeClassifier(random_state = 0)
    dt.fit(x_train, y_train)
    importances_dt = dt.feature_importances_
    importances_series_dt = pd.Series(importances_dt)
    rank_dt = importances_series_dt.rank(ascending = False, method = 'min')

    rf = RandomForestClassifier(random_state = 0)
    rf.fit(x_train, y_train)
    importances = rf.feature_importances_
    importances_series_rf = pd.Series(importances)
    rank_rf = importances_series_rf.rank(ascending = False, method = 'min')

    extra = ExtraTreesClassifier(random_state = 0)
    extra.fit(x_train, y_train)
    importances = extra.feature_importances_
    importances_series_extra = pd.Series(importances)
    rank_extra = importances_series_extra.rank(ascending = False, method = 'min')

    # xgb = XGBClassifier(random_state = 0)
    # xgb.fit(x_train, y_train)
    # importances = xgb.feature_importances_
    # importances_series_xgb = pd.Series(importances)
    # rank_xgb = importances_series_xgb.rank(ascending = False, method = 'min')

    selector = SelectKBest(chi2, k = 'all')
    selector.fit(x_train ,y_train)
    importances = selector.scores_
    importances_series_chi2 = pd.Series(importances)
    rank_selectkb_chi2 = importances_series_chi2.rank(ascending = False, method = 'min')

    selector = SelectKBest(f_classif, k = 'all')
    selector.fit(x_train ,y_train)
    importances = selector.scores_
    importances_series_f = pd.Series(importances)
    rank_selectkb_f = importances_series_f.rank(ascending = False, method = 'min')

    rank_df = pd.concat([rank_dt, rank_rf , rank_extra, rank_selectkb_chi2, rank_selectkb_f], axis = 1)
    rank_df['mean'] = rank_df.mean(axis = 1)
    rank_sort = np.sort(rank_df['mean'])
        
    rank_high_index = []
    for i in rank_df.index:
        if rank_df.loc[i,'mean'] < rank_sort[1000]:
            rank_high_index.append(i)
    
    rank_high_idx = x_train.columns[rank_high_index]
    return rank_high_idx   



       