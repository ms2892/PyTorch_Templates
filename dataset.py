import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import numpy as np
import math

class WineDataset(Dataset):
    
    def __init__(self):
        # Data Loading
        xy = np.loadtxt('data/wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,0].astype(np.int64)-1)   # n_sample, 1
        self.y_onehot = nn.functional.one_hot(self.y.to(torch.int64)) # One Hot Encoding n_samples,n_classes
        # print(self.y_onehot[0])
        # t=input()
        self.n_samples = xy.shape[0]
        self.n_features = self.x.shape[1]
        self.n_classes = self.y_onehot.shape[1]
        
    def __getitem__(self,index):
        # dataset[index]
        # Return the dataset at that particular index
        # Return Training Sample and Training Output
        # print(self.y[index])
        # t=input()
        return self.x[index],self.y[index]
        
    def __len__(self):
        
        # Returns the number of samples present in the dataset
        return self.n_samples
    
    def get_features(self):
        return self.n_features

    def get_num_classes(self):
        return self.n_classes
    
if __name__=='__main__':
    # Initilize a Dataset Class
    dataset = WineDataset()

    # Initialize a dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

    # This is an iterator that goes through the entire dataset
    dataiter = iter(dataloader)
    # The next item can be called using this method
    # data = dataiter.next()
    # Returns a batch of samples 
