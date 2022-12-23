import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import numpy as np
import math

class WineDataset(Dataset):
    
    def __init__(self, transform=None):
        xy = np.loadtxt('data/wine.csv',delimiter=',',skiprows=1,dtype=np.float32)
        self.n_samples = xy.shape[0]
        
        self.x = xy[:,1:]
        self.y = xy[:,[0]]
        self.transform = transform
    
    def __getitem__(self,index):
        sample = self.x[index],self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.n_samples

class ToTensor:
    # Convert Sample to Tensor
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    
    def __init__(self,factor):
        self.factor = factor
    def __call__(self,sample):
        inputs, targets = sample
        inputs = self.factor * inputs
        return inputs,targets

if __name__=='__main__':
    composed = torchvision.transforms.Compose([ToTensor(),MulTransform(2)])
    dataset = WineDataset(transform=composed)
    print(dataset[0])