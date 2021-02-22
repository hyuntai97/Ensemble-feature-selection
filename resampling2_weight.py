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

def ensemble2_resampling_weight(n_resampling, n_estimators, featurenum, seed, x_train = None, y_train = None):
    np.random.seed(seed = seed) # seed값 설정
    model = RandomForestClassifier() # 모델 변경 가능 (가중치 성능 모델)
    selected_cols_dict = {}    
    for x in range(n_resampling):
        sample = np.random.choice(x_train.shape[0],200 , replace = False)
        x_train_sample = x_train.iloc[sample, :]
        y_train_sample = y_train.iloc[sample]
        n = x_train_sample.shape[1]
        # trainset에서 200개 resampling 데이터를 제외하고 나머지는 train 안의 validation set
        not_sample = [i for i in range(x_train.shape[0]) if i not in sample]
        x_train_val = x_train.iloc[not_sample, :]
        y_train_val = y_train.iloc[not_sample]


        # 총 8개의 fs_method ensemble (rfe는 시간이 너무 오래 걸려서 제외)
        fs_model_tree = [RandomForestClassifier(random_state = seed, n_estimators = n_estimators),
                          ExtraTreesClassifier(random_state = seed, n_estimators = n_estimators)]

        fs_model_kb = [SelectKBest(chi2, k = 'all'), SelectKBest(f_classif, k = 'all')]

        fs_model_shap = [RandomForestClassifier(random_state = seed, n_estimators = n_estimators)]



        # tree importance
        for j,fs in enumerate(fs_model_tree):
            fs_model = fs
            fs_model.fit(x_train_sample, y_train_sample)
            importances = fs_model.feature_importances_
            importances_series = pd.Series(importances)
            
            # 가장 작은 rank 값을 가진 10개의 high_importance feature를 사용하여 1번 학습
            rank = importances_series.rank(ascending = False, method = 'min') 
            rank_high_idx = rank[rank <= 10].index
            x_train_sample_high = x_train_sample.iloc[:, rank_high_idx]
            x_train_val_high = x_train_val.iloc[:, rank_high_idx]
            model = model.fit(x_train_sample_high, y_train_sample)
            
            # 학습된 데이터로 acc계산
            y_pred_val = model.predict(x_train_val_high)
            accuracy_val = accuracy_score(y_train_val, y_pred_val) # 가중치 (1차적인 feature selection의 성능)

            # 각 feature selection method 별로 feature 선택 
            rank_high_index = rank[rank <= 400].index 
            selected_columns = x_train.columns[rank_high_index]
            
            # 가중치(1차적인 feature selection의 성능) 을 포함시킨 selected_columns
            selected_columns_acc = []
            for col in selected_columns:
                selected_columns_acc.append(col + f'_{accuracy_val**2}')
            
            selected_cols_dict[f'tree_importances_{j}_{x}'] = selected_columns_acc

        for j,fs in enumerate(fs_model_kb):
            fs_model = fs
            fs_model.fit(x_train_sample, y_train_sample)
            importances = fs_model.scores_
            importances2 = np.nan_to_num(importances)
            importances_series = pd.Series(importances2)

            # 가장 작은 rank 값을 가진 10개의 high_importance feature를 사용하여 1번 학습
            rank = importances_series.rank(ascending = False, method = 'min') 
            rank_high_idx = rank[rank <= 10].index
            x_train_sample_high = x_train_sample.iloc[:, rank_high_idx]
            x_train_val_high = x_train_val.iloc[:, rank_high_idx]
            model = model.fit(x_train_sample_high, y_train_sample)
            
            # 학습된 데이터로 acc계산
            y_pred_val = model.predict(x_train_val_high)
            accuracy_val = accuracy_score(y_train_val, y_pred_val) # 가중치 (1차적인 feature selection의 성능)

            # 각 feature selection method 별로 feature 선택 
            rank_high_index = rank[rank <= 400].index 
            selected_columns = x_train.columns[rank_high_index]
            
            # 가중치(1차적인 feature selection의 성능) 을 포함시킨 selected_columns
            selected_columns_acc = []
            for col in selected_columns:
                selected_columns_acc.append(col + f'_{accuracy_val**2}')
            
            selected_cols_dict[f'selectkbest_{j}_{x}'] = selected_columns_acc

        for j,fs in enumerate(fs_model_shap):
            fs_model = fs
            fs_model.fit(x_train_sample, y_train_sample)
            explainer = shap.TreeExplainer(fs_model)
            shap_values = explainer.shap_values(x_train_sample)
            shap_values_mat = np.abs(shap_values[1])
            shap_mean = np.mean(shap_values_mat, axis = 0)
            importances_series = pd.Series(shap_mean)
            
            # 가장 작은 rank 값을 가진 10개의 high_importance feature를 사용하여 1번 학습
            rank = importances_series.rank(ascending = False, method = 'min')
            rank_high_idx = rank[rank <= 10].index
            x_train_sample_high = x_train_sample.iloc[:, rank_high_idx]
            x_train_val_high = x_train_val.iloc[:, rank_high_idx]
            model = model.fit(x_train_sample_high, y_train_sample)
            
            # 학습된 데이터로 acc계산
            y_pred_val = model.predict(x_train_val_high)
            accuracy_val = accuracy_score(y_train_val, y_pred_val) # 가중치 (1차적인 feature selection의 성능)

            # 각 feature selection method 별로 feature 선택 
            rank_high_index = rank[rank <= 400].index 
            selected_columns = x_train.columns[rank_high_index]
            
            # 가중치(1차적인 feature selection의 성능) 을 포함시킨 selected_columns
            selected_columns_acc = []
            for col in selected_columns:
                selected_columns_acc.append(col + f'_{accuracy_val**2}')
            
            selected_cols_dict[f'shap_{j}_{x}'] = selected_columns_acc

    df_cols = pd.DataFrame(selected_cols_dict)

    columns = []
    for i in df_cols.values:
        for j in i:
            columns.append(j)
    
    # selected feature들의 가중치를 다 합하기 
    counts = dict()
    for col in columns:
        key = col.split('_')[0]
        value = col.split('_')[1]
        if key not in counts:
            counts[key] = np.float(value)
        else:
            counts[key] += np.float(value)
    
    sorted_counts = sorted(counts.items(), key = lambda x:x[1], reverse = True)
    selected_columns_lst = []
    for i in range(featurenum):    # 지정 된 featurenum 개수만큼 상위 빈출 컬럼 뽑기  
        selected_columns_lst.append(sorted_counts[i][0])
    
    return selected_columns_lst



            
            


