import numpy as np
import pandas 
import pickle
import xgboost
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
import time
from config import train_config
from data import load_data

print("Load and process datasets.")
(X_train, Y_train, X_valid, Y_valid, X_test, train, test) = load_data()
dtrain = xgboost.DMatrix(X_train, Y_train)
dvalid = xgboost.DMatrix(X_valid, Y_valid)

print("Load test set.")
dtest = xgboost.DMatrix(X_test)

watchlist = [(dtrain, "train"), (dvalid, "eval")]

print("Training model. This may take awhile...")
params = {"objective": "reg:linear",
            "booster": "gbtree",
            'eval_metric': 'rmse',
            "eta": 0.01,
            "max_depth": 10,
            "min_child_weight": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.85,
            "silent": 1,
            "lambda": 0.01,
            "seed": 1301}
num_boost_round = 50000
model = xgboost.train(params, dtrain, num_boost_round, evals=watchlist,
                      early_stopping_rounds=400, verbose_eval=True)

#pickle.dump(model, open("model.dat", "wb"))

print("Validating the model.")
yhat = model.predict(xgboost.DMatrix(X_valid))
print('MSE-xgb: ', mean_squared_error(Y_valid, yhat))

print("Making predictions on the test set.")
predictions = model.predict(dtest)

print("Saving predictions to csv file for submission.")
result = pandas.DataFrame(
    {"date": test["date"], "brand": test["brandid"], "cnt": predictions})
result = result[['date', 'brand', 'cnt']]
result['cnt'] = result['cnt'].astype(int)
result['date'] = result['date'].astype(int)
result['brand'] = result['brand'].astype(int)
result.to_csv('../output/xgb.csv', index=False)

print("Done")

