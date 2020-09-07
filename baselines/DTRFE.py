from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def DTRFE(X_train,Y_train,X_val,Y_val,model):
    from sklearn.feature_selection import RFE
    estimator = DecisionTreeClassifier()
    selector = RFE(estimator, int(c/2)+1, step=1)
    selector = selector.fit(X_train, Y_train)
    rfe_mask = selector.get_support()
        
    X_selected = X_train[:,rfe_mask==1]
    model.fit(X_selected,Y_train)
    
    predict_result = model.predict(X_val[:,rfe_mask==1])
    accuracy = accuracy_score(predict_result,Y_val) 
    fscore = f1_score(predict_result,Y_val)
    #accuracy = model.score(X_val[:,rfe_mask==1], Y_val)
    print('DTRFE for '+data_name, accuracy,fscore)