import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, sample_num, feature_num, action_num=2):
        super(Net,self).__init__()
        # state representation
        self.fc1 = nn.Linear(feature_num, 1)
        self.fc1.weight.data.normal_(0, 0.1)
        # real network
        self.fc2 = nn.Linear(sample_num, 512)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(512, 512)
        self.fc3.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(512,action_num)
        
    def forward(self,x):
        x = self.fc1(x).squeeze(-1)
        x = self.fc3(self.fc2(x))
        x = F.relu(x)
        action_value = self.out(x)
        return action_value
        # (batch,2)

    
class DQN(object):
    def __init__(self,sample_num, feature_num, BATCH_SIZE, MEMORY_CAPACITY, TARGET_REPLACE_ITER, EPSILON, GAMMA):
        self.eval_net, self.target_net = Net(sample_num, feature_num), Net(sample_num, feature_num)
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.EPSILON = EPSILON
        self.GAMMA = GAMMA
        self.memory_counter = 0
        self.memory = [()] * MEMORY_CAPACITY
        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=self.LR)
        self.loss_func = nn.MSELoss()
        
    def choose_action(self,x):
        x = Variable(torch.FloatTensor(x),requires_grad=False)
        if np.random.uniform() < self.EPSILON:
            action_value = self.eval_net.forward(x)
            _, action = torch.topk(action_value,1)
            action = action[0].data.numpy().item()
        else:
            action = np.random.randint(0,2)
        return action
    
    def store_transition(self,state,action,reward,next_state):
        # state: X_train, action: 0,1  reward, scalar  
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index] = (state,action,reward,next_state)
        self.memory_counter +=1
            
    def learn(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter +=1
        
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_s = Variable(FloatTensor([ item[0]  for item,idx in enumerate(self.memory) if idx in sample_index]))
        b_a = Variable(LongTensor([ item[1] for item,idx in enumerate(self.memory) if idx in sample_index]))
        b_r = Variable(FloatTensor([item[2] for item,idx in enumerate(self.memory) if idx in sample_index ] ))
        b_s_ = Variable(FloatTensor([ item[3] for item,idx in enumerate(self.memory) if idx in sample_index]))
        
        q_eval = self.eval_net(b_s).gather(1,b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.GAMMA*q_next.max(1)
        loss = self.loss_func(q_eval,q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        