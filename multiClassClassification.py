import torch
import torch.nn as nn
import numpy as np
from dataset import WineDataset
import torchvision
from torch.utils.data import Dataset,DataLoader

# Multi Class Problem
class NeuralNetwork(nn.Module):
    
    def __init__(self,input_dim,hidden_nodes, num_classes):
        super(NeuralNetwork,self).__init__()
        
        self.linear1 = nn.Linear(input_dim,hidden_nodes)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_nodes,num_classes)
    
    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # NO SOFTMAX
        return out

if __name__=='__main__':
    
    criterion = nn.CrossEntropyLoss()
    dataset = WineDataset()

    model = NeuralNetwork(input_dim=dataset.get_features(), hidden_nodes=15,num_classes=dataset.get_num_classes())
    n_samples = len(dataset)
    dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True)
    
    num_epochs=100
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    
    for epoch in range(num_epochs):
        total_loss=0
        model.train()
        correct_preds=0
        for i,(inputs,labels) in enumerate(dataloader):
            y_pred = model(inputs)
            # print(y_pred,labels)
            l = criterion(y_pred,labels)
            
            l.backward()
            total_loss+=l.item()
            optimizer.step()
            optimizer.zero_grad()
            # print(y_pred.shape)
            _,predictions = torch.max(y_pred,1)
            # print(predictions,labels)
            # t=input()
            correct_preds += predictions.eq(labels).sum()
            
        if (epoch+1)%10==0:
            tr_acc = correct_preds/n_samples
            print(f'Epoch {epoch+1}: Accuracy = {tr_acc}, Loss = {total_loss:0.4f}')
            
            