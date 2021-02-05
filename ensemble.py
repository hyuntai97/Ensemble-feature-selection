from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression, f_classif
from sklearn import preprocessing
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np 
import shap
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
from sklearn.feature_selection import SelectFromModel

def ensemble_model(n_estimators, featurenum, seed, x_train = None, y_train = None):
    rank_lst = []
    fs_model_tree = [DecisionTreeClassifier(random_state = seed), 
                    RandomForestClassifier(random_state = seed, n_estimators = n_estimators),
                    ExtraTreesClassifier(random_state = seed, n_estimators = n_estimators)]

    fs_model_kb = [SelectKBest(chi2, k = 'all'), SelectKBest(f_classif, k = 'all')]

    fs_model_rfe = [RFE(RandomForestClassifier(random_state = seed), n_features_to_select=1000, step=0.1),
                    RFE(LogisticRegression(random_state = seed), n_features_to_select=1000, step = 0.1), 
                    RFE(DecisionTreeClassifier(random_state = seed), n_features_to_select=1000, step=0.1)]

    fs_model_shap = [RandomForestClassifier(random_state = seed, n_estimators = n_estimators),
                    DecisionTreeClassifier(random_state = seed)]

    fs_model_regular = [SelectFromModel(LogisticRegression(C = 1,penalty = 'l1',solver = 'liblinear'), max_features = featurenum),
                        SelectFromModel(LogisticRegression(C = 1,penalty = 'l2',solver = 'liblinear'), max_features = featurenum),
                        SelectFromModel(LogisticRegression( penalty = 'elasticnet',solver = 'saga', l1_ratio = 0.5), max_features = featurenum)]

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

    for fs in fs_model_rfe:
        rfe = fs
        rfe.fit(x_train, y_train)
        rank = rfe.ranking_
        rank_series = pd.Series(rank)
        rank_lst.append(rank_series)
        
    for fs in fs_model_shap:
        model = fs
        model.fit(x_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_train)
        shap_values_mat = np.abs(shap_values[1])
        shap_mean = np.mean(shap_values_mat, axis = 0)
        importances_series = pd.Series(shap_mean)
        rank = importances_series.rank(ascending = False, method = 'min')
        rank_lst.append(rank)

    for fs in fs_model_regular:
        select_model = fs
        select_model.fit(x_train, y_train)
        importances = np.abs(select_model.estimator_.coef_)[0]
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


    


       