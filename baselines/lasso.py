from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def LASSO(X_train,Y_train,X_val,Y_val,model):
    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=0.15)
    clf.fit(X_train, Y_train)
    action_list = clf.coef_ != 0
    #print(action_list)
    X_selected = X_train[:,action_list==1]
    model.fit(X_selected,Y_train)
    predict_result = model.predict(X_val[:,action_list==1])
    accuracy = accuracy_score(predict_result,Y_val) 
    fscore = f1_score(predict_result,Y_val)
    #accuracy = model.score(X_val[:,action_list==1], Y_val)
    print('LASSO for '+data_name,accuracy,fscore)

    