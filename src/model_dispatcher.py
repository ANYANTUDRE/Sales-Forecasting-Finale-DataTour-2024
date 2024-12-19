from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import linear_model

import xgboost as xgb
import lightgbm as lgbm
import catboost


models = {  
            ########################### TREE based models ##########################
            #"random_forest": ensemble.RandomForestRegressor(random_state=42), # MAPE=0.2570, takes too much time

            "xgb_reg": xgb.XGBRegressor(n_jobs=-1, objective='reg:squaredlogerror', 
                                        **{'learning_rate': 0.024586620116602568, 
                                           'max_depth': 3, 
                                           'n_estimators': 1023, 
                                           'subsample': 0.8977345599104098, 
                                           'colsample_bytree': 0.6223663433173937, 
                                           'reg_alpha': 5.386632692050109, 
                                           'reg_lambda': 0.19546599138830106, 
                                           'min_child_weight': 3}), # MAPE=0.2542
            
            "hist": ensemble.HistGradientBoostingRegressor(random_state=1,
                                                           **{'max_iter': 899, 
                                                              'max_depth': 12, 
                                                              'learning_rate': 0.044200322727998786}), #  MAPE=0.2473

            #"extra": ensemble.ExtraTreesRegressor(), # MAPE=0.2554

            "lgbm": lgbm.LGBMRegressor(n_jobs=-1, verbose=-1, random_state=0,
                                       **{'num_leaves': 11, 
                                          'max_depth': 20, 
                                          'learning_rate': 0.0151678662401369, 
                                          'n_estimators': 1749, 
                                          'min_child_samples': 11, 
                                          'subsample': 0.5292201210404861, 
                                          'colsample_bytree': 0.6475575118022834}), # MAPE=0.2468

            'cat': catboost.CatBoostRegressor(verbose=False), # MAPE=0.2494

            'bagging': ensemble.BaggingRegressor(estimator=ensemble.HistGradientBoostingRegressor(), n_estimators=100, random_state=0),

            'stacking': ensemble.StackingRegressor(estimators=[ ('hist', ensemble.HistGradientBoostingRegressor(random_state=1,
                                                               **{'max_iter': 899, 
                                                                  'max_depth': 12, 
                                                                  'learning_rate': 0.044200322727998786})
                                                                  ),  
                                                                  ('lgbm', lgbm.LGBMRegressor(n_jobs=-1, verbose=-1, random_state=0,
                                                                    **{'num_leaves': 11, 
                                                                        'max_depth': 20, 
                                                                        'learning_rate': 0.0151678662401369, 
                                                                        'n_estimators': 1749, 
                                                                        'min_child_samples': 11, 
                                                                        'subsample': 0.5292201210404861, 
                                                                        'colsample_bytree': 0.6475575118022834})
                                                                  ),
                                                               ], 
                                                    final_estimator=xgb.XGBRegressor(n_jobs=-1, objective='reg:squarederror', 
                                                                            **{'learning_rate': 0.024586620116602568, 
                                                                            'max_depth': 3, 
                                                                            'n_estimators': 1023, 
                                                                            'subsample': 0.8977345599104098, 
                                                                            'colsample_bytree': 0.6223663433173937, 
                                                                            'reg_alpha': 5.386632692050109, 
                                                                            'reg_lambda': 0.19546599138830106, 
                                                                            'min_child_weight': 3}
                                                                  )),

            
            'voting_reg': ensemble.VotingRegressor([('hist', ensemble.HistGradientBoostingRegressor(random_state=1,
                                                           **{'max_iter': 899, 
                                                              'max_depth': 12, 
                                                              'learning_rate': 0.044200322727998786})
                                                              ), 

                                                    ('lgbm', lgbm.LGBMRegressor(n_jobs=-1, verbose=-1, random_state=0,
                                                                    **{'num_leaves': 11, 
                                                                        'max_depth': 20, 
                                                                        'learning_rate': 0.0151678662401369, 
                                                                        'n_estimators': 1749, 
                                                                        'min_child_samples': 11, 
                                                                        'subsample': 0.5292201210404861, 
                                                                        'colsample_bytree': 0.6475575118022834})
                                                                        ), 
                                                    ('xgb', xgb.XGBRegressor(n_jobs=-1, objective='reg:squarederror', 
                                                                            **{'learning_rate': 0.024586620116602568, 
                                                                            'max_depth': 3, 
                                                                            'n_estimators': 1023, 
                                                                            'subsample': 0.8977345599104098, 
                                                                            'colsample_bytree': 0.6223663433173937, 
                                                                            'reg_alpha': 5.386632692050109, 
                                                                            'reg_lambda': 0.19546599138830106, 
                                                                            'min_child_weight': 3})
                                                                            ),
                                                    #('cat', catboost.CatBoostRegressor(verbose=False)),
                                                    ])
}
