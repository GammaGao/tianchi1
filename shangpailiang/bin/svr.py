import numpy as np
import pandas
import pickle
import xgboost
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.svm import SVR
from data import load_data

print("Load and process datasets.")
(X_train, Y_train, X_valid, Y_valid, X_test, train, test) = load_data()

print("Training model. This may take awhile...")
svr = SVR(C=1000, gamma=0.05)

svr.fit(X_train, Y_train)

print("Validating the model.")
score = svr.score(X_valid, Y_valid)
print("Score-svr: ", score)
yhat = svr.predict(X_valid)
print('MSE-svr: ', mean_squared_error(Y_valid, yhat))

print("Making predictions on the test set.")
predictions = svr.predict(X_test)

print("Saving predictions to csv file for submission.")
result = pandas.DataFrame(
    {"date": test["date"], "brand": test["brandid"], "cnt": predictions})
result = result[['date', 'brand', 'cnt']]
result['cnt'] = result['cnt'].astype(int)
result['date'] = result['date'].astype(int)
result['brand'] = result['brand'].astype(int)
result.to_csv('../output/svr.csv', index=False)

print("Done")
