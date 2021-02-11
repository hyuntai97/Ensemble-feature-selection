from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn import preprocessing
from sklearn.decomposition import PCA
import shap
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from ensemble2 import ensemble_model2
from ensemble import ensemble_model
from ensemble2_resampling import ensemble2_model_resampling
from ensemble1_resampling import ensemble1_model_resampling

# feature select
def feature_select(fsmethod , featurenum , rfestep, seed, n_estimators, n_resampling, x_train = None, y_train =None, x_val = None):
    if fsmethod == 'SelectKBest_f':
        selector = SelectKBest(score_func = f_classif, k = featurenum)
        selector.fit(x_train, y_train)
        
        importances = selector.scores_
        importances2 = np.nan_to_num(importances) # importance score nan 값 제거
        importances_sort = np.sort(importances2)
        importances_high_lst = []
        for i in range(len(importances2)):
            if importances2[i] > importances_sort[-featurenum-1]:
                importances_high_lst.append(i)
        importances_high_sorted = sorted(importances_high_lst, key = lambda x: importances2[x], reverse = True)
        selected_columns = x_train.columns[importances_high_sorted]

        x_train = x_train.loc[:, selected_columns]
        x_val = x_val.loc[:, selected_columns]

    if fsmethod == 'SelectKBest_chi2':
        selector = SelectKBest(score_func = chi2, k = featurenum)
        selector.fit(x_train, y_train)
        
        importances = selector.scores_
        importances2 = np.nan_to_num(importances) # importance score nan 값 제거
        importances_sort = np.sort(importances2)
        importances_high_lst = []
        for i in range(len(importances2)):
            if importances2[i] > importances_sort[-featurenum-1]:
                importances_high_lst.append(i)
        importances_high_sorted = sorted(importances_high_lst, key = lambda x: importances2[x], reverse = True)
        selected_columns = x_train.columns[importances_high_sorted]

        x_train = x_train.loc[:, selected_columns]
        x_val = x_val.loc[:, selected_columns]

    if fsmethod == 'feature_importances_rf':
        model = RandomForestClassifier(n_estimators = n_estimators, random_state = seed)
        model.fit(x_train, y_train)
        importances = model.feature_importances_
        importances_sort = np.sort(importances)
        importances_high_lst = []
        for i in range(len(importances)):
            if importances[i] > importances_sort[-featurenum-1]:
                importances_high_lst.append(i)
        selected_columns_lst = sorted(importances_high_lst, key = lambda x: importances[x], reverse = True)
        selected_columns = x_train.columns[selected_columns_lst]
        x_train = x_train.loc[:, selected_columns]
        x_val = x_val.loc[:, selected_columns]

    if fsmethod == 'feature_importances_ext':
        model = ExtraTreesClassifier(random_state = seed)
        model.fit(x_train, y_train)
        importances = model.feature_importances_
        importances_sort = np.sort(importances)
        importances_high_lst = []
        for i in range(len(importances)):
            if importances[i] > importances_sort[-featurenum-1]:
                importances_high_lst.append(i)
        selected_columns_lst = sorted(importances_high_lst, key = lambda x: importances[x], reverse = True)
        selected_columns = x_train.columns[selected_columns_lst]
        x_train = x_train.loc[:, selected_columns]
        x_val = x_val.loc[:, selected_columns]
     

        
    if fsmethod == 'rfe_log':      # feature importance로 selected feature 정렬 불가능
        model_log = LogisticRegression(random_state = seed)
        select_rfe = RFE(model_log, n_features_to_select = featurenum, step = rfestep)
        select_rfe.fit(x_train, y_train)
        selected_mask = select_rfe.support_
        selected_columns = x_train.columns[selected_mask]
        x_train = x_train.loc[:, selected_columns]
        x_val = x_val.loc[:, selected_columns]

    if fsmethod == 'rfe_rf':      # feature importance로 selected feature 정렬 불가능
        model_lr = RandomForestClassifier(random_state = seed)
        select_rfe = RFE(model_lr, n_features_to_select = featurenum, step = rfestep)
        select_rfe.fit(x_train, y_train)
        selected_mask = select_rfe.support_
        selected_columns = x_train.columns[selected_mask]
        x_train = x_train.loc[:, selected_columns]
        x_val = x_val.loc[:, selected_columns]
    
    if fsmethod == 'rfe_dt':      # feature importance로 selected feature 정렬 불가능
        model_dt = DecisionTreeClassifier(random_state = seed)
        select_rfe = RFE(model_dt, n_features_to_select = featurenum, step = rfestep)
        select_rfe.fit(x_train, y_train)
        selected_mask = select_rfe.support_
        selected_columns = x_train.columns[selected_mask]
        x_train = x_train.loc[:, selected_columns]
        x_val = x_val.loc[:, selected_columns]
    

    if fsmethod == 'pca':
        # pca 전 데이터 정규화   
        x_train_index = x_train.index
        x_val_index = x_val.index
        scaler = preprocessing.StandardScaler()
        scaler.fit(x_train)
        x_train_new = scaler.transform(x_train)
        x_val_new = scaler.transform(x_val)
        x_train = pd.DataFrame(x_train, columns = x_train.columns, index = x_train_index)
        x_val = pd.DataFrame(x_val, columns = x_val.columns, index = x_val_index)

        # pca
        pca = PCA(n_components = x_train.shape[0])
        principal = pca.fit(x_train)
        x_train_new = principal.transform(x_train)
        x_val_new = principal.transform(x_val)
        x_train = pd.DataFrame(x_train_new, index = x_train_index)
        x_val = pd.DataFrame(x_val_new, index = x_val_index)

    # shap feature selection method
    if fsmethod == 'shap1':
        model = RandomForestClassifier(n_estimators = n_estimators, random_state = seed)
        model.fit(x_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_train)
        shap_values_mat = np.abs(shap_values[1])
        shap_mean = np.mean(shap_values_mat, axis = 0)
        shap_mean_sort = np.sort(shap_mean)
        importances_high_lst = []
        for i in range(len(shap_mean)):
            if shap_mean[i] > shap_mean_sort[-featurenum-1]:
                importances_high_lst.append(i)
        selected_columns_lst = sorted(importances_high_lst, key = lambda x: shap_mean[x], reverse = True)
        selected_columns = x_train.columns[selected_columns_lst]
        x_train = x_train.loc[:, selected_columns]
        x_val = x_val.loc[:, selected_columns]

    if fsmethod == 'shap2':
        model = DecisionTreeClassifier(random_state = seed)
        model.fit(x_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_train)
        shap_values_mat = np.abs(shap_values[1])
        shap_mean = np.mean(shap_values_mat, axis = 0)
        shap_mean_sort = np.sort(shap_mean)
        importances_high_lst = []
        for i in range(len(shap_mean)):
            if shap_mean[i] > shap_mean_sort[-featurenum-1]:
                importances_high_lst.append(i)
        selected_columns_lst = sorted(importances_high_lst, key = lambda x: shap_mean[x], reverse = True)
        selected_columns = x_train.columns[selected_columns_lst]
        x_train = x_train.loc[:, selected_columns]
        x_val = x_val.loc[:, selected_columns]

    
    # feature select regularization method       
    if fsmethod == 'regularization_l1':
        select_model = SelectFromModel(LogisticRegression(C = 1,penalty = 'l1',solver = 'liblinear'), max_features = featurenum)
        select_model.fit(x_train, y_train)
        importances = np.sum(np.abs(select_model.estimator_.coef_), axis = 0)
        importances_sort = np.sort(importances)
        importances_high_lst = []
        for i in range(len(importances)):
            if importances[i] > importances_sort[-featurenum-1]:
                importances_high_lst.append(i)
        importances_high_sorted = sorted(importances_high_lst, key = lambda x: importances[x], reverse = True)
        selected_columns = x_train.columns[importances_high_sorted]

        x_train = x_train.loc[:, selected_columns]
        x_val = x_val.loc[:, selected_columns]

    if fsmethod == 'regularization_l2':
        select_model = SelectFromModel(LogisticRegression(C = 1,penalty = 'l2',solver = 'liblinear'), max_features = featurenum)
        select_model.fit(x_train, y_train)
        importances = np.sum(np.abs(select_model.estimator_.coef_), axis = 0)
        importances_sort = np.sort(importances)
        importances_high_lst = []
        for i in range(len(importances)):
            if importances[i] > importances_sort[-featurenum-1]:
                importances_high_lst.append(i)
        importances_high_sorted = sorted(importances_high_lst, key = lambda x: importances[x], reverse = True)
        selected_columns = x_train.columns[importances_high_sorted]

        x_train = x_train.loc[:, selected_columns]
        x_val = x_val.loc[:, selected_columns]

    if fsmethod == 'regularization_elastic':
        select_model = SelectFromModel(LogisticRegression( penalty = 'elasticnet',solver = 'saga',l1_ratio = 0.5), max_features = featurenum)
        select_model.fit(x_train, y_train)
        importances = np.sum(np.abs(select_model.estimator_.coef_), axis = 0)
        importances_sort = np.sort(importances)
        importances_high_lst = []
        for i in range(len(importances)):
            if importances[i] > importances_sort[-featurenum-1]:
                importances_high_lst.append(i)
        importances_high_sorted = sorted(importances_high_lst, key = lambda x: importances[x], reverse = True)
        selected_columns = x_train.columns[importances_high_sorted]

        x_train = x_train.loc[:, selected_columns]
        x_val = x_val.loc[:, selected_columns]

    
    # feature select ensemble method
    if fsmethod == 'ensemble1':
        rank_high_idx = ensemble_model(n_estimators, featurenum, seed, x_train, y_train)
        x_train = x_train.loc[:, rank_high_idx]
        x_val = x_val.loc[:, rank_high_idx]

        selected_columns = rank_high_idx

    if fsmethod == 'ensemble2':
        frequent_high_idx = ensemble_model2(n_estimators, featurenum, seed, x_train, y_train)
        x_train = x_train.loc[:, frequent_high_idx]
        x_val = x_val.loc[:, frequent_high_idx]

        selected_columns = frequent_high_idx
    
    # feature select ensemble & resampling method
    if fsmethod == 'ensemble1_resampling':
        rank_high_idx = ensemble1_model_resampling(n_resampling, n_estimators, featurenum, seed, x_train, y_train)
        x_train = x_train.loc[:, rank_high_idx]
        x_val = x_val.loc[:, rank_high_idx]
        
        selected_columns = rank_high_idx

    if fsmethod == 'ensemble2_resampling':
        frequent_high_idx = ensemble2_model_resampling(n_resampling, n_estimators, featurenum, seed, x_train, y_train)
        x_train = x_train.loc[:, frequent_high_idx]
        x_val = x_val.loc[:, frequent_high_idx]

        selected_columns = frequent_high_idx
    

    return x_train, x_val , selected_columns


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

# replace outlier

def get_outlier(df = None, column = None, weight = 1.5):
    a = df.loc[:,column]
    quantile_25 = np.percentile(a.values, 25)
    quantile_75 = np.percentile(a.values, 75)
    
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    #lowest_val = quantile_25 - iqr_weight  # 최소 제한선 (대체 값)
    highest_val = quantile_75 + iqr_weight  # 최대 제한선 (대체 값)
    
    #low_outlier_index = a[(a < lowest_val)].index  # 최소 제한선보다 낮은 이상치들의 인덱스
    high_outlier_index = a[(a > highest_val)].index  # 최대 제한선보다 높은 이상치들의 인덱스
    
    return high_outlier_index,highest_val

# test set 에서 이상치 인덱스 찾는 함수 
def get_outlier_test2(df = None, column = None , outlier_dict = None):
    a = df.loc[:,column]
    outlier_index = a[(a > outlier_dict[column])].index

    return outlier_index

def replace_outlier(x_train = None, x_val = None ):
    # 높은 이상치 값만 대체 (낮은 이상치 값은 대체하지 않음)
    outlier_dict = {}
    for i in x_train.columns:
        high_outlier_index,highest_val = get_outlier(x_train, i)
        outlier_dict[i] = highest_val
        # x_train.loc[low_outlier_index,i] = lowest_val
        x_train.loc[high_outlier_index,i] = highest_val   # train set 이상치 대체 

    # test set의 이상치 대체 (X_train에서 얻은 값으로 대체시킴)
    for i in x_val.columns:
        outlier_index = get_outlier_test2(x_val, i ,outlier_dict = outlier_dict)
        x_val.loc[outlier_index,i] = outlier_dict[i]

    return x_train, x_val 


# sample generate

def sample_generate(x_train = None, y_train = None):
    x_train_new = x_train.copy()
    x_train_new['target,LUAD:0,LUSC:1'] = y_train.values
    x_train0 = x_train_new.loc[x_train_new['target,LUAD:0,LUSC:1']==0]
    x_train1 = x_train_new.loc[x_train_new['target,LUAD:0,LUSC:1']==1]

    idx_lst0 = np.random.choice(x_train0.shape[0],3000)
    new_sample_lst = []
    for i in range(1000):
        new_sample = (x_train0.iloc[idx_lst0[i]] + x_train0.iloc[idx_lst0[i*2]] + x_train0.iloc[idx_lst0[i*3]] ) / 3
        new_sample_lst.append(new_sample)

    idx_lst1 = np.random.choice(x_train1.shape[0],3000)
    for i in range(1000):
        new_sample = (x_train1.iloc[idx_lst1[i]] + x_train1.iloc[idx_lst1[i*2]] + x_train1.iloc[idx_lst1[i*3]] ) / 3
        new_sample_lst.append(new_sample)

    for i in range(len(new_sample_lst)):
        x_train_new.loc[f"gen_sample{i}"] = new_sample_lst[i].values
    
    y_train = x_train_new.iloc[:,-1]
    x_train = x_train_new.drop('target,LUAD:0,LUSC:1', axis = 1)

    return x_train, y_train


  







  
