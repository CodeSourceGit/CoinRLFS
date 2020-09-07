from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def RFRFE(X_train,Y_train,X_val,Y_val,model):
    from sklearn.feature_selection import RFE
    estimator = RandomForestClassifier()
    selector = RFE(estimator, int(c/2)-1, step=1)
    selector = selector.fit(X_train, Y_train)
    rfe_mask = selector.get_support()
    X_selected = X_train[:,rfe_mask==1]
    model.fit(X_selected,Y_train)
    predict_result = model.predict(X_val[:,rfe_mask==1])
    accuracy = accuracy_score(predict_result,Y_val) 
    fscore = f1_score(predict_result,Y_val)
    print('RFRFE for '+data_name, accuracy,fscore)