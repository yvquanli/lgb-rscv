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
                 kf=KFold(n_splits=5), score_function=None, random_state=999, params_type='params_default'):
        self.objective = objective
        self.param_dist = param_dist
        self.cv = cv
        self.n_iter = n_iter
        self.kf = kf
        self.kf.n_splits = cv
        self.kf.shuffle = True

        if score_function is None:
            if self.objective == 'regression': self.score_function = [r2_score]
            if self.objective == 'multiclass': self.score_function = [accuracy_score]
        else:
            if isinstance(score_function, list) or isinstance(score_function, tuple):
                self.score_function = score_function
            else:
                self.score_function = [score_function]  # cast to list
            
        self.kf.random_state = random_state
        self.random_state = random_state
        random.seed(random_state)
        
        self.results = []
        self.tried_params = []
        self.params_type = params_type
        print(self.params_type)
        
    def fit(self, X, y): 
        print("Random search start...\n")
        for i in trange(self.n_iter):
            params_light = {
                'objective': self.objective,
                'num_leaves': choice([15, 31, 63, 81, 127, 197, 231, 275, 302, 511]),
                'bagging_fraction': choice([0.5, 0.7, 0.8, 0.9]),
                'learning_rate': choice([0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5]),
                'min_data': choice([300, 400, 450, 500, 550, 650]),
                'is_unbalance': choice([True, False]),
                'max_bin': choice([3, 5, 10, 12, 18, 20, 22]),
                'boosting_type' : choice(['gbdt', 'rf', ]),
                'bagging_freq': choice([3, 9, 11, 15, 17, 23, 31]),
                # 'max_depth': choice([3, 4, 5, 6, 7, 9, 11]),
                'feature_fraction': choice([0.5, 0.7, 0.8, 0.9]),
                'lambda_l1': choice([0, 10, 20, 30, 40]),
                'verbosity': -1,
            }
            
            params_default = {
                'objective': self.objective,
                'boosting': choice(['gbdt', 'dart', 'rf', ]),
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
                'verbosity': -1,
            }
            
            params = params_default if self.params_type=='params_default' else params_light
            
            if self.objective == 'regression':
                params['metric'] = 'l2'
            elif self.objective == 'multiclass':
                params['num_class'] = len(pd.unique(y))
                params['metric'] = 'logloss'
            
            if params in self.tried_params: print('params exist!'); continue

            result_temp = {}
            result_temp['iter'] = i
            result_temp['params'] = params
            result_temp['best_lgb_iter'] = []
            for fun in self.score_function:
                result_temp['train_' + fun.__qualname__] = 0
                result_temp['val_' + fun.__qualname__] = 0

            # cross validation
            for trn_idx, val_idx in self.kf.split(X, y):
                trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])#, categorical_feature=categorical_feats)
                val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=trn_data)#, categorical_feature=categorical_feats)

                gbm = lgb.train(params, 
                                trn_data,
                                num_boost_round=10000,
                                valid_sets=[trn_data, val_data],
                                early_stopping_rounds=100,
                                verbose_eval=0,
                                )
                y_train_pred = gbm.predict(X.iloc[trn_idx], num_iteration=gbm.best_iteration)   # 
                y_val_pred = gbm.predict(X.iloc[val_idx], num_iteration=gbm.best_iteration)   # 
                
                if self.objective == 'multiclass':
                    y_train_pred = np.argmax(y_train_pred, axis=1)
                    y_val_pred = np.argmax(y_val_pred, axis=1)
                
                # add info such as val_score
                result_temp['best_lgb_iter'].append(gbm.best_iteration)
                for fun in self.score_function:
                    result_temp['train_'+fun.__qualname__] += fun(y.iloc[trn_idx], y_train_pred)/self.cv
                    result_temp['val_'+fun.__qualname__] += fun(y.iloc[val_idx], y_val_pred)/self.cv
                    
            self.tried_params.append(copy.deepcopy(params))
            self.results.append(copy.deepcopy(result_temp))
            gc.collect()

    def score(self, X, y):
        pass

    def results_store(self, path=None):
        if path is None:
            if os.path.exists('./output') is False: os.makedirs('./output')
            pd.DataFrame(self.results).to_excel('./output/rscv_results.xls')
            return
        pd.DataFrame(self.results).to_excel(path)
        
        
def train_cv_predict(X, y, X_test, params, metric_function, kf, objective, random_state):
    results = []
    # cross validation
    for fold_, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])  # , categorical_feature=categorical_feats)
        val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx],
                               reference=trn_data)  # , categorical_feature=categorical_feats)

        gbm = lgb.train(params,
                        trn_data,
                        num_boost_round=20000,
                        valid_sets=[trn_data, val_data],
                        early_stopping_rounds=100,
                        verbose_eval=0,
                        )
        y_train_pred = gbm.predict(X.iloc[trn_idx], num_iteration=gbm.best_iteration)  #
        y_val_pred = gbm.predict(X.iloc[val_idx], num_iteration=gbm.best_iteration)  #

        if objective == 'multiclass':
            y_train_pred = np.argmax(y_train_pred, axis=1)
            y_val_pred = np.argmax(y_val_pred, axis=1)

        # add info such as val_score
        result_temp = {}
        result_temp['fold'] = fold_
        result_temp['best_lgb_iter'] = gbm.best_iteration
        result_temp['train_' + metric_function.__qualname__] = metric_function(y.iloc[trn_idx], y_train_pred)
        result_temp['val_' + metric_function.__qualname__] = metric_function(y.iloc[val_idx], y_val_pred)
        result_temp['test_pred'] = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        print('fold{}:'.format(fold_), metric_function(y.iloc[val_idx], y_val_pred))
        results.append(copy.deepcopy(result_temp))
    return results


def train_cv_predict_by_average_fold(X, y, X_test, params, metric_function, kf, objective, random_state):
    random.seed(random_state)
    kf.shuffle = True
    kf.random_state = random_state

    results = train_cv_predict(X, y,  X_test, params, metric_function, kf, objective, random_state)
    predictions = np.zeros(len(results[0]['test_pred']))
    for result in results:
        predictions += result['test_pred'] / kf.n_splits
    return predictions

def train_cv_predict_by_best_fold(X, y, X_test, params, metric_function, kf, objective, random_state):
    random.seed(random_state)
    kf.shuffle = True
    kf.random_state = random_state

    results = train_cv_predict(X, y,  X_test, params, metric_function, kf, objective, random_state)

    # to find the validation's scoring name
    score_name = ''
    for key in results[0].keys():
        if 'val_' in key:
            score_name = key
            break

    # find the best
    best_fold = results[0]
    for result in results:
        if result[score_name] > best_fold[score_name]:
            best_fold = result

    return best_fold['test_pred']

def train_cv_predict_by_top_n_fold(X, y, X_test, params, metric_function, kf, objective, random_state, n_top=3):
    random.seed(random_state)
    kf.shuffle = True
    kf.random_state = random_state

    results = train_cv_predict(X, y,  X_test, params, metric_function, kf, objective, random_state)
    results = pd.DataFrame(results)
    score_name = results.columns[4]
    results.sort_values([score_name], inplace=True, ascending=False)

    predictions = np.zeros(len(results['test_pred'][0]))
    for i in range(n_top):
        predictions += results['test_pred'][i] / n_top

    return predictions


        
#    warm start
#         if more_iter_warm_start == 0:
#             tqdm_iterator = trange(self.n_iter)
#         else:
#             tqdm_iterator = trange(self.n_iter, self.n_iter + more_iter_warm_start)
#             self.n_iter += more_iter_warm_start
            
    
    
    
    
    
#  for multiclass train
#     lgb.train(
#           [
#               objective = "multiclass", ## 多分类
#               metric = "multi_logloss", ## 多元交叉熵
#               num_class = 5, ## 5类
#               learning_rate = grid_search[i, 'Learning_rate'],
#               num_leaves = grid_search[i, 'Num_leaves']
#           ],
#           data = dtrain,
#           nrounds = 10, ## 十折交叉验证（K-fold Cross Validation）
#           valids = valids,
#           early_stopping_rounds = 10,
#           num_iterations = 100
#     )

