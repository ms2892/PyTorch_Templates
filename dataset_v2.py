import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import numpy as np
import math

class WineDataset(Dataset):
    
    def __init__(self):
        xy = np.loadtxt('data/wine.csv',delimiter=',',skiprows=1,dtype=np.float32)