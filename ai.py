# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)  #full connection between input layer and hidden layer
        self.fc2 = nn.Linear(30, nb_action)   #full connection between hidden layer and output layer
    
    def forward(self, state):
        x = F.relu(self.fc1(state))      #for activation of hidden layers
        q_values = self.fc2(x)           #for activation of output layers
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):       #for replay we have to pass the capacity to save the transistions
        self.capacity = capacity
        self.memory = []             #here we create the list with capacity=100
    
    def push(self, event):     
        self.memory.append(event)   #appending theprevious transistion in the memory
        if len(self.memory) > self.capacity: #if the capcity exceeds the limit 100 then delete the first element
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))     #it will batch the contents of the list to like reward in one group and actions in one group
        return map(lambda x: Variable(torch.cat(x, 0)), samples)   #maping the values to batches and the tensors and gradients are used to point to these

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*200) # T=100  #here we are using softmax to select the highest q value from the probability distribution and we are taking temperature difference =7
        action = probs.multinomial(num_samples=1)   #if the  temperature difference increases than the q value also increases 
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)  #it will take the output in the form of batches than select the one value
        next_outputs = self.model(batch_next_state).detach().max(1)[0]  #next ouput will be the maximum value from the 
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):  #for update the score 
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):   #for saving the file in the memory
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):    #for loading the save file and use it
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")