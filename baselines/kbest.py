from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import  mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def KBest(X1,Y1,X2,Y2,model):
    action_list = SelectKBest(mutual_info_classif, int(c/2) ).fit(X1,Y1).get_support(indices=False)
    X_selected = X1[:,action_list==1]
    model.fit(X_selected,Y1)
    
    predict_result = model.predict(X2[:,action_list==1])
    accuracy = accuracy_score(predict_result,Y2) 
    fscore = f1_score(predict_result,Y2)
    #print(action_list)
    print('KBest for '+data_name, accuracy,fscore)
