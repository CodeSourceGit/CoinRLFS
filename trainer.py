import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def outlier_trainer(X_train,Y_train):
    from sklearn import svm
    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest
    labels = set(Y_train)
    result = np.ones(len(Y_train))
    for label in labels:
        index = np.where(Y_train == label)[0]
        current_x = X_train[index]
        current_y = Y_train[index]
        clf = IsolationForest(contamination=0.20)
        clf.fit(current_x)
        predict_outliers = clf.predict(current_x)
        for oidx, outlier in enumerate(predict_outliers):
            if outlier == -1:
                result[index[oidx]] = 0
    return result


def feature_trainer(X_train,Y_train):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X_train,Y_train)
    selection_prob = clf.feature_importances_ * X_train.shape[1]
    result = []
    for item in selection_prob:
        if np.random.random() < item:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)
                
    