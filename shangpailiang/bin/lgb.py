import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib as plt
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
import time
from data import load_data

print("Load and process datasets.")
(X_train, Y_train, X_valid, Y_valid, X_test, train, test) = load_data()
'''
Y = train['cnt']
X = train.drop(["cnt"], axis=1)
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 1],
    'n_estimators': [10,20,40,60,80,100,120,140,160,180,200]
}
features = X.columns
estimator = lgb.LGBMRegressor(num_leaves=64)
gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X, Y)
print('Best parameters found by grid search are:', gbm.best_params_)
'''

print('Start training...')
gbm = lgb.LGBMRegressor(objective='regression', num_leaves=64)  # ,  n_estimators=40,learning_rate=0.1)
gbm.fit(X_train, Y_train,
        eval_set=[(X_valid, Y_valid)],
        eval_metric='mse',
        early_stopping_rounds=500)

# categorical_feature=["brand","date_type"]
print('Start predicting...')
# predict
y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration_)
# eval
print('MSE-lgb', mean_squared_error(Y_valid, y_pred))

# feature importances
print('Feature importances:', list(gbm.feature_importances_))

# feat_imp = pd.Series(gbm.feature_importances_, features).sort_values(ascending=False)
# feat_imp.plot(kind='bar', title='Feature Importances')

print("Making predictions on the test set.")
predictions = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

print("Saving predictions to csv file for submission.")
result = pandas.DataFrame(
    {"date": test["date"], "brand": test["brandid"], "cnt": predictions})
result = result[['date', 'brand', 'cnt']]
result['cnt'] = result['cnt'].astype(int)
result['date'] = result['date'].astype(int)
result['brand'] = result['brand'].astype(int)
result.to_csv('../output/lgb.csv', index=False)

print("Done")
