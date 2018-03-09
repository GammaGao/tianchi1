import numpy as np
import pandas
import pickle
import xgboost
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
import time
from config import train_config


def load_data():
    myconfig = train_config()
    (trainf, testaf, testbf) = myconfig.get_train_settings()
    print("Load and process datasets.")

    train = pandas.read_csv(trainf)
    train = train.drop(["day", "day_of_year", "date", "brandid"], axis=1)
    # 右偏分布转正态分布?
    # train['cnt'] = np.log(train['cnt'])

    print("Split the datasets into training set and validation set.")
    X_train, X_valid = train_test_split(train, test_size=0.1, random_state=10)

    Y_train = X_train['cnt']
    X_train = X_train.drop(["cnt"], axis=1)

    Y_valid = X_valid['cnt']
    X_valid = X_valid.drop(["cnt"], axis=1)

    print("Load test set.")
    test = pandas.read_csv(testbf)
    X_test = test.drop(["day", "day_of_year", "date", "brandid"], axis=1)
    return (X_train, Y_train, X_valid, Y_valid, X_test, train, test)
