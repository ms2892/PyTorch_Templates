import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import numpy as np
import math

class LogisticRegression(nn.Module):
    
    def __init__(self,input_dim,output_dim):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(input_dim,output_dim)
        self.softmax = nn.Softmax()
        
    def forward(self,x):
        print(self.linear(x))
        y_pred = self.softmax(self.linear(x))
        
        return y_pred


class WineDataset(Dataset):
    
    def __init__(self):
        # Data Loading
        xy = np.loadtxt('data/wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,0])   # n_sample, 1
        self.y = nn.functional.one_hot(self.y.to(torch.int64)) # One Hot Encoding n_samples,n_classes
        self.n_samples = xy.shape[0]
        self.n_features = self.x.shape[1]
        self.n_classes = self.y.shape[1]
        
    def __getitem__(self,index):
        # dataset[index]
        # Return the dataset at that particular index
        # Return Training Sample and Training Output
        return self.x[index],self.y[index]
        
    def __len__(self):
        
        # Returns the number of samples present in the dataset
        return self.n_samples
    
    def get_features(self):
        return self.n_features

    def get_num_classes(self):
        return self.n_classes
    
# Initilize a Dataset Class
dataset = WineDataset()

# Initialize a dataloader
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# This is an iterator that goes through the entire dataset
dataiter = iter(dataloader)
# The next item can be called using this method
# data = dataiter.next()
# Returns a batch of samples 

# Training Loop
num_epochs = 100
total_samples = len(dataset)

print(dataset.get_features(),dataset.get_num_classes())

model = LogisticRegression(dataset.get_features(),dataset.get_num_classes())




n_iters = math.ceil(total_samples/4)

# Needs to be fixed
criterion = nn.BCELoss()
#-----------------#

optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

print(total_samples,n_iters)


for epoch in range(num_epochs):
    model.train()
    total_loss=0
    correct_pred=0
    for i,(inputs,labels) in enumerate(dataloader):
        # Forward Pass
        y_pred = model(inputs)
        l = criterion(y_pred,labels)
        
        # Backward Pass
        l.backward()
        
        # Update
        optimizer.step()
        
        # Flush
        optimizer.zero_grad()
        
        total_loss += l.item()
        y_pred = y_pred.round()
        correct_pred += y_pred.eq(labels).sum()
    
    if (epoch+1)%10==0:
        model.eval()
        print(f'Epoch {epoch+1}: Accuracy = {correct_pred/total_samples}, Loss = {total_loss:.4f}')