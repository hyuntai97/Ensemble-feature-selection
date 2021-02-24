# Lab intern Machine-Learning-Pipeline
Suwon university DSML center
# Environments

**Requirment**

```bash
scikit-learn == 0.23.2
```

**ML task**

- Classification

# ML in scikit-learn

## Models

**Linear Models**

- Logistic Regression


**Tree-based Models**

- Decision Tree
- Random Forest
- Adaboost

**Support Vector Machine**

- SVC

**Neighbors**

- K-Nearest Neighbors

**Naive Bayes**

- Gaussian Naive Bayes

## Evaluation

**Classification**

- accuracy_score

## Save & Load

- pickle

# Process

1. Load Data
2. Preprocessing Data
3. Model Training
4. Evaluation
5. Save

# Pipeline

**Scripts**

```bash
.
├── load.py
├── preprocess.py
├── main.py
├── scaling.py
├── ensemble.py
├── ensemble2.py
├── ensemble1_resampling.py
├── ensemble2_resampling.py
├── resampling1_weight.py
├── resampling2_weight.py
└── 
```

## main.py

**Argument**

```
Machine Learning Pipeline

optional arguments:
  --seed          Set seed
  --datadir       Set data directory
  --logdir        Set log directory
  --val_size      Set validation size
  --kfold         Number of cross validation
  --fsmethod      Choice feature select method       
  --featurenum    Set feature selected number
  --rfestep       Set rfe step
  --standardize   Choice standardize method
  --normalize     Choice nomalize method
  --dataload      Choice data loading method
  --n_estimators  Set the number of trees in the forest
  --colsdir       Set Directory in which the selected column is stored
  --n_resampling  Set resampling count
```

**Code**

```python
# 1. load data
train, test = dataloader(datadir=args.datadir, dataload = args.dataload)

# 2. feature standardize
x_train, x_val = standardize_select(args.standardize,x_train, x_val)

# 3. feature select
x_train, x_val = feature_select(args.fsmethod ,args.featurenum , args.rfestep, args.seed, args.n_estimators , args.n_resampling, x_train, y_train, x_val)

# 4. replace outlier
x_train, x_val = replace_outlier(x_train, x_val)

# 5. feature normalize
x_train, x_val = normalize_select(args.normalize,x_train, x_val)

# 6. sample generate
x_train, y_train = sample_generate(x_train, y_train)

# 7. model training
model = model.fit(x_train, y_train)
    
```


