import pandas as pd
from data_process import ReadData
from sklearn import model_selection
import numpy as np

def pois_spam():
    data_name = 'spam'
    dataset = ReadData(data_name)
    r, c = dataset.shape
    X = dataset.iloc[:,0:(c-1)]
    Y = dataset.iloc[:,(c-1)]
    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size=0.3, random_state=0)
    X_train, X_val, Y_train, Y_val = X_train.values, X_val.values, Y_train.values, Y_val.values
    start_point = 10
    adv_size = int(r*0.3)
    theta = 1.001
    np.random.seed(1)
    X_adv = X_train[start_point:start_point+adv_size,:] * theta
    Y_adv = np.random.randint(0,2, size=adv_size)
    X_adverse = np.concatenate((X_train, X_adv))
    Y_adverse = np.concatenate((Y_train,Y_adv))
    return X_adverse, X_val, Y_adverse, Y_val

def pois_arr():
    data_name = 'arrhythmia'
    dataset = ReadData(data_name)
    r, c = dataset.shape
    X = dataset.iloc[:,0:(c-1)]
    Y = dataset.iloc[:,(c-1)]
    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size=0.3, random_state=0)
    X_train, X_val, Y_train, Y_val = X_train.values, X_val.values, Y_train.values, Y_val.values
    start_point = 10
    adv_size = int(r*0.3)
    np.random.seed(1)
    X_adv = X_train[start_point:start_point+adv_size,:]
    Y_adv = np.random.randint(0,16, size=adv_size)

    X_adverse = np.concatenate((X_train, X_adv))
    Y_adverse = np.concatenate((Y_train,Y_adv))
    return X_adverse, X_val, Y_adverse, Y_val


def normal_spam():
    data_name = 'spam'
    dataset = ReadData(data_name)
    r, c = dataset.shape
    X = dataset.iloc[:,0:(c-1)]
    Y = dataset.iloc[:,(c-1)]
    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size=0.3, random_state=0)
    X_train, X_val, Y_train, Y_val = X_train.values, X_val.values, Y_train.values, Y_val.values
    return X_train, X_val, Y_train, Y_val


def normal_arr():
    data_name = 'arrhythmia'
    dataset = ReadData(data_name)
    r, c = dataset.shape
    X = dataset.iloc[:,0:(c-1)]
    Y = dataset.iloc[:,(c-1)]
    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size=0.3, random_state=0)
    X_train, X_val, Y_train, Y_val = X_train.values, X_val.values, Y_train.values, Y_val.values
    return X_train, X_val, Y_train, Y_val