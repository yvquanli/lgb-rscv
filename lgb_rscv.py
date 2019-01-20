import gc
import copy
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from random import choice
from tqdm import trange
from lgb_rscv_params import C_leaves
from lgb_rscv_params import C_001_009_0001
from lgb_rscv_params import C_01_09_001
from lgb_rscv_params import C_05_099_001
from lgb_rscv_params import C_1_9_01
from lgb_rscv_params import C_10_99_1
from lgb_rscv_params import C_100_990_10
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# objective  classification?
# skipped params: continue
# pred
# score
# score std?


class RandomizedSearchCV:
    def __init__(self, objective='regression', param_dist=None, cv=5, n_iter=200, 
                 kf=KFold(n_splits=5), score_function=None, random_state=999):
        self.objective = objective
        self.param_dist = param_dist
        self.cv = cv
        self.n_iter = n_iter
        self.kf = kf
        self.kf.n_splits = cv
        self.kf.shuffle = True
        self.kf.random_state = random_state
        
        if score_function is None:
            if self.objective == 'regression': self.score_function = [r2_score]
            if self.objective == 'multiclass': self.score_function = [accuracy_score]
        else:
            if isinstance(score_function, list) or isinstance(score_function, tuple):
                self.score_function = score_function
            else:
                self.score_function = [score_function]  # cast to list
            
        self.random_state = random_state
        random.seed(random_state)
        
        self.results = []
        self.tried_params = []
        
    def fit(self, X, y): 
        print("Random search start...\n")
        for i in trange(self.n_iter):
            params_light = {
                'num_leaves': choice([15, 31, 63, 81, 127, 197, 231, 275, 302, 511]),
                'bagging_fraction': choice([0.5, 0.7, 0.8, 0.9]),
                'learning_rate': choice([0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5]),
                'min_data': choice([300, 400, 450, 500, 550, 650]),
                'is_unbalance': choice([True, False]),
                'max_bin': choice([3, 5, 10, 12, 18, 20, 22]),
                'boosting_type' : choice(['gbdt', 'dart']),
                'bagging_freq': choice([3, 9, 11, 15, 17, 23, 31]),
                # 'max_depth': choice([3, 4, 5, 6, 7, 9, 11]),
                'feature_fraction': choice([0.5, 0.7, 0.8, 0.9]),
                'lambda_l1': choice([0, 10, 20, 30, 40]),
            }
            
            params_default = {
                'boosting': choice(['gbdt', 'dart', 'rf']),
                'learning_rate': choice(C_001_009_0001 + C_01_09_001),
                'num_leaves': choice([ 7, 15, 31, 63, 127, 255, 511, 1023, 2047,]),   # 如果设置了max_depth，无需设置
                'tree_learner': choice(['data', 'feature', 'serial', 'voting' ]),
                # 'max_depth': choice(list(range(3, 50))),
                'max_bin': choice(C_10_99_1 + C_100_990_10), 
                'min_data_in_leaf': choice(C_10_99_1),
                'feature_fraction': choice(C_05_099_001 +[0.8]*20) ,   # C_01_09_001
                'bagging_fraction': choice(C_05_099_001 +[0.8]*20) ,   # C_01_09_001
                'bagging_freq': choice(list(range(1, 11))) ,
                'lambda_l1': choice(C_001_009_0001 + C_01_09_001 + C_1_9_01 + [0]*100),
                'lambda_l2': choice(C_001_009_0001 + C_01_09_001 + C_1_9_01 + [0]*100),
                'min_split_gain': choice(C_001_009_0001 + C_01_09_001 + C_1_9_01 + [0]*100),
            }
            
            params = params_default
            if self.objective == 'multiclass':
                params['num_class'] = len(pd.unique(y))
            
            if params in self.tried_params: print('params exist!'); continue

            score_temp = {}
            for fun in self.score_function:
                score_temp['train_' + fun.__qualname__] = 0
                score_temp['val_' + fun.__qualname__] = 0
            score_temp['iter'] = i
            score_temp['params'] = params

            # cross validation
            for train_index, val_index in self.kf.split(X, y):
                X_train = X.iloc[train_index]
                y_train = y.iloc[train_index]
                X_val = X.iloc[val_index]
                y_val = y.iloc[val_index]
                gbm = lgb.LGBMModel(objective=self.objective, n_estimators=20000, **params)
                gbm.fit(X=X_train, y=y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=100,
                        verbose=False)
                y_train_pred = gbm.predict(X_train, )   # num_iteration=gbm.best_iteration_
                y_val_pred = gbm.predict(X_val, )   # num_iteration=gbm.best_iteration_
                
                if self.objective == 'multiclass':
                    y_train_pred = np.argmax(y_train_pred, axis=1)
                    y_val_pred = np.argmax(y_val_pred, axis=1)
                
                # add info such as val_score
                for fun in self.score_function:
                    score_temp['train_'+fun.__qualname__] += fun(y_train, y_train_pred)/self.cv
                    score_temp['val_'+fun.__qualname__] += fun(y_val, y_val_pred)/self.cv
                    
            self.tried_params.append(copy.deepcopy(params))
            self.results.append(copy.deepcopy(score_temp))
            gc.collect()

    def predict(self, X):
        pass

    def score(self, X, y):
        pass

    def results_store(self, path=None):
        if path is None:
            if os.path.exists('./output') is False: os.makedirs('./output')
            pd.DataFrame(self.results).to_excel('./output/rscv_results.xls')
            return
        pd.DataFrame(self.results).to_excel(path)
        
        
        
        

        
#    warm start
#         if more_iter_warm_start == 0:
#             tqdm_iterator = trange(self.n_iter)
#         else:
#             tqdm_iterator = trange(self.n_iter, self.n_iter + more_iter_warm_start)
#             self.n_iter += more_iter_warm_start
            