import pandas as pd 
import numpy as np
from numpy import isnan
import os 

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

def dataloader(datadir,dataload):
    data = pd.read_csv(os.path.join(datadir, f'LUADLUSC_float32.tsv'),sep = '\t')
    data_copy = data.copy()
    data_copy.set_index('sample', inplace = True)

    # dataload시에 전체데이터 로그변환 -> 전체데이터 scaling
    if dataload == 1:
        X = data_copy.iloc[:,:-1]
        y_target = data_copy.iloc[:,-1]

        X_columns = X.columns
        X_index = X.index

        # log 변환
        X = X.apply(np.log1p)

        # Standard scaling
        scaler = Normalizer()
        scaler.fit(X)
        X = scaler.transform(X)
        X = pd.DataFrame(X, columns = X_columns, index = X_index)

    elif dataload == 2:
        X = data_copy.iloc[:,:-1]
        y_target = data_copy.iloc[:,-1]

    elif dataload == 3:  # c4 computational gene set 사용해서 feature 개수 줄이는 방법 
        data_gene = pd.read_csv(os.path.join(datadir, f'c4_entrez.gmt.txt'),sep = '\t', engine = 'python' , header = None, index_col = 0, error_bad_lines = False)
        data_gene.drop(1, axis = 1, inplace = True)
        data_gene.fillna(0, inplace = True)

        gene_idx = []
        for idx in data_gene.index:
            gene = data_gene.loc[idx].values.astype(int)
            gene_idx.append(gene)
        idx = []
        for i in range(len(gene_idx)):
            for j in gene_idx[i]:
                idx.append(j)
        set_idx = set(idx)
        lst_idx = list(set_idx)

        # new feature dataframe 생성
        overlap_idx = [] # computational dataset에 있으면서 기존 데이터에도 있는 유전자의 entrez번호 
        for col in data_copy.columns[:-1]:
            if int(col.split('|')[0].split(':')[1]) in lst_idx:
                overlap_idx.append(col)
        data_copy2 = data_copy.loc[:,overlap_idx] # 새롭게 만든 데이터 프레임
        data_copy2['target,LUAD:0,LUSC:1'] = data_copy['target,LUAD:0,LUSC:1']
        X = data_copy.iloc[:,:-1]
        y_target = data_copy.iloc[:,-1]
        
        
    return X, y_target