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
from sklearn.metrics import accuracy_score

def ensemble1_resampling_weight(n_resampling, n_estimators, featurenum, seed, x_train = None, y_train = None):
    model = RandomForestClassifier()
    weighted_rank_lst = []
    for i in range(n_resampling):
        # trainset에서 200개 데이터 resampling
        sample = np.random.choice(x_train.shape[0],200,replace = False)
        x_train_sample = x_train.iloc[sample, :]
        y_train_sample = y_train.iloc[sample]
        n = x_train_sample.shape[1]
        # trainset에서 200개 resampling 데이터를 제외하고 나머지는 train 안의 validation set
        not_sample = [i for i in range(x_train.shape[0]) if i not in sample]
        x_train_val = x_train.iloc[not_sample, :]
        y_train_val = y_train.iloc[not_sample]
        
        # ensemble에는 tree_based, selectkbest, shap 사용 
        fs_model_tree = [RandomForestClassifier(random_state = seed, n_estimators = n_estimators),
                         ExtraTreesClassifier(random_state = seed, n_estimators = n_estimators)]

        fs_model_kb = [SelectKBest(chi2, k = 'all'), SelectKBest(f_classif, k = 'all')]

        fs_model_shap = [RandomForestClassifier(random_state = seed, n_estimators = n_estimators)]

        # tree based (weighted rank)
        for fs in fs_model_tree:
            fs_model = fs
            fs_model.fit(x_train_sample, y_train_sample)
            importances = fs_model.feature_importances_
            importances_series = pd.Series(importances)
            # 가장 큰 rank 값을 가진 10개의 feature를 사용하여 1번 학습
            rank = importances_series.rank(ascending = True, method = 'max') # 순위 역순
            rank_high_idx = rank[rank > n - 10].index
            x_train_sample_high = x_train_sample.iloc[:, rank_high_idx]
            x_train_val_high = x_train_val.iloc[:, rank_high_idx]
            model = model.fit(x_train_sample_high, y_train_sample)
            # 학습된 데이터로 acc 계산
            y_pred_val = model.predict(x_train_val_high)
            accuracy_val = accuracy_score(y_train_val, y_pred_val)
            # weighted rank (클수록 높은 importance)
            weighted_rank = rank**2 * accuracy_val**2

            weighted_rank_lst.append(weighted_rank)

        # selectkbest
        for fs in fs_model_kb:
            fs_model = fs
            fs_model.fit(x_train_sample, y_train_sample)
            importances = fs_model.scores_
            importances2 = np.nan_to_num(importances)
            importances_series = pd.Series(importances2)
            # 가장 큰 rank 값을 가진 10개의 feature를 사용하여 1번 학습
            rank = importances_series.rank(ascending = True, method = 'max') # 순위 역순
            rank_high_idx = rank[rank > n - 10].index
            x_train_sample_high = x_train_sample.iloc[:, rank_high_idx]
            x_train_val_high = x_train_val.iloc[:, rank_high_idx]
            model = model.fit(x_train_sample_high, y_train_sample)
            # 학습된 데이터로 acc 계산
            y_pred_val = model.predict(x_train_val_high)
            accuracy_val = accuracy_score(y_train_val, y_pred_val)
            # weighted rank (클수록 높은 importance)
            weighted_rank = rank**2 * accuracy_val**2

            weighted_rank_lst.append(weighted_rank)

        for fs in fs_model_shap:
            fs_model = fs
            fs_model.fit(x_train_sample, y_train_sample)
            explainer = shap.TreeExplainer(fs_model)
            shap_values = explainer.shap_values(x_train_sample)
            shap_values_mat = np.abs(shap_values[1])
            shap_mean = np.mean(shap_values_mat, axis = 0)
            importances_series = pd.Series(shap_mean)
            # 가장 큰 rank 값을 가진 10개의 feature를 사용하여 1번 학습
            rank = importances_series.rank(ascending = True, method = 'max') # 순위 역순
            rank_high_idx = rank[rank > n - 10].index
            x_train_sample_high = x_train_sample.iloc[:, rank_high_idx]
            x_train_val_high = x_train_val.iloc[:, rank_high_idx]
            model = model.fit(x_train_sample_high, y_train_sample)
            # 학습된 데이터로 acc계산
            y_pred_val = model.predict(x_train_val_high)
            accuracy_val = accuracy_score(y_train_val, y_pred_val)
            # weighted rank (클수록 높은 importance)
            weighted_rank = rank * accuracy_val**2
            
            weighted_rank_lst.append(weighted_rank)

    rank_df = pd.concat(weighted_rank_lst, axis = 1)
    
    rank_df['sum'] = rank_df.sum(axis = 1)  # 열 별 weighted rank 의 sum을 구하기
    rank_sort = np.sort(rank_df['sum'])[::-1]

    # 가장 순위 합이 높은 feature들을 featurenum만큼 Select !! 
    rank_high_index = []
    for i in rank_df.index:
        if rank_df.loc[i, 'sum'] > rank_sort[featurenum]:
            rank_high_index.append(i)

    selected_columns_lst = sorted(rank_high_index, key = lambda x : rank_df.loc[x, 'sum'], reverse = True)
    rank_high_idx = x_train.columns[selected_columns_lst]

    return rank_high_idx    


            

        
       

