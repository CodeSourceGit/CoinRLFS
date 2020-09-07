import pandas as pd
from data_process import ReadData
from sklearn import model_selection
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
from data_process import *
from poison import *
from DQN import *
from trainer import *
import argparse
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_name', type=str, default='spam', help='dataset')
parser.add_argument('-model','--model', type=str, default='dt', help='downstream model')
parser.add_argument('-p','--poisoned', type=int, default=0, help='1:poison, 0:non-poison')
parser.add_argument('-it','--instance_trainer', type=int, default=0, help='1: use, 0: no use')
parser.add_argument('-ft','--feature_trainer', type=int, default=0, help='1: use, 0: no use')
parser.add_argument('-od','--output_dir', type=str, default='', help='output file name')

args = parser.parse_args()


data_name = args.data_name
if data_name == 'spam' and args.poisoned == 1:
    X_train, X_val, Y_train, Y_val = pois_spam()
elif data_name == 'spam' and args.poisoned == 0:
    X_train, X_val, Y_train, Y_val = normal_spam()
elif data_name == 'arr' and args.poisoned == 1:
    X_train, X_val, Y_train, Y_val = pois_arr()
elif data_name == 'arr' and args.poisoned == 0:
    X_train, X_val, Y_train, Y_val = normal_arr()

if args.model == 'dt':
    model = DecisionTreeClassifier()
elif args.model == 'lr':
    model = LogisticRegression()

    
class Env():
    def __init__(self,X_train,Y_train,X_val,Y_val):
        self.X_train, self.Y_train = X_train, Y_train
        self.X_val, self.Y_val = X_val,Y_val
        self.last_acc = 0
        self.last_f1 = 0
    def step(self, sample_action, feature_action):
        sample_num = sample_action.shape[0]
        feature_num = feature_action.shape[0]
        feature_matrix = feature_action.repeat(sample_num,axis=0)
        sample_matrix = sample_action.repeat(feature_num,axis=1)
        current_state = self.X_train * feature_matrix * sample_matrix
        model.fit(self.X_train[sample_action>0,:][:,feature_action>0], self.Y_train[sample_action>0])
        predict_result = model.predict(self.X_val[:,feature_action>0])
        current_acc = accuracy_score(predict_result,self.Y_val) 
        current_f1 = f1_score(predict_result,self.Y_val)
        reward = ((current_acc - self.last_acc) + (current_f1 - self.last_f1))/2
        self.last_acc = current_acc
        self.last_f1 = current_f1
        return current_state, reward, (current_acc, current_f1)


if args.instance_trainer:
    instance_advice = outlier_trainer(X_train,Y_train)
if args.feature_trainer:
    feature_advice = feature_trainer(X_train,Y_train)

instance_flag = True
feature_flag = True

import torch
from utils import Net,DQN
from torch.autograd import Variable

BATCH_SIZE = 16
LR = 0.01
EPSILON = 0.8
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 200


sample_num, feature_num = X_train.shape
FeatureAgent = DQN(sample_num, feature_num, BATCH_SIZE, LR, MEMORY_CAPACITY, TARGET_REPLACE_ITER, EPSILON, GAMMA)
SampleAgent = DQN(sample_num, feature_num, BATCH_SIZE, LR, MEMORY_CAPACITY, TARGET_REPLACE_ITER, EPSILON, GAMMA)
env = Env(X_train,Y_train,X_val,Y_val)


FA_count = 0
SA_count = 0
sample_action = np.ones(sample_num)
feature_action = np.ones(feature_num)
state,reward,_ = env.step(sample_action,feature_action)

result_list = []

file_name = args.output_dir

for step in range(20000):
    # Sample_action
    if SA_count >= sample_num:
        SA_count = SA_count % sample_num
        sample_action = np.ones(sample_num)
        state,_,_ = env.step(sample_action,feature_action)
        instance_flag = False
        continue
    else:
        if args.instance_trainer and instance_flag:
            temp_sample_action = int(instance_advice[SA_count])
            sample_action[SA_count] = temp_sample_action
            SA_count += 1
        else:    
            temp_sample_action = SampleAgent.choose_action(state)
            sample_action[SA_count] = temp_sample_action
            SA_count += 1
        
    # Feature_action
    if FA_count >= feature_num:
        FA_count = FA_count % feature_num
        feature_action = np.ones(feature_num)
        state,_,_ = env.step(sample_action,feature_action)
        feature_flag = False
        continue
    else:
        if args.feature_trainer and feature_flag:
            temp_feature_action = int(feature_advice[FA_count])
            feature_action[FA_count] = temp_feature_action
            FA_count += 1
        else:
            temp_feature_action = FeatureAgent.choose_action(state)
            feature_action[FA_count] = temp_feature_action
            FA_count += 1

    next_state, reward, performance = env.step(sample_action,feature_action)
    reward = reward
    SampleAgent.store_transition(state,temp_sample_action, reward, next_state)
    FeatureAgent.store_transition(state,temp_feature_action, reward, next_state)
    if step > MEMORY_CAPACITY+1:
        SampleAgent.learn()
        FeatureAgent.learn()
    
    result_list.append(performance)
    if step % 1000 ==0:
        print(step, "max performance:",max(result_list))
    
    with open(file_name,'a+') as f:
        f.write(str(performance) + '\n' )

    state = next_state