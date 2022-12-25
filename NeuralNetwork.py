# Neural Network for MNIST dataset

# This Python file will contain
# DataLoader, Transformations
# MultiLayer Neural Network, Activation Functions
# Loss and Optimizer
# Training Loop (Batch Training)
# Model Evaluation

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
input_size = 28*28 # Flatten the image to 784 X 1
hidden_size = 100  # Hidden Layer Nodes
num_classes = 10   # Number of digits
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST 
train_dataset = torchvision.datasets.MNIST(root='data/MNIST', train=True,
                                           transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='data/MNIST', train=False,
                                           transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim,num_classes)
        
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
model = NeuralNet(input_size,hidden_size,num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for i,(img,labels) in enumerate(train_loader):
        img = img.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        
        # Forward Pass
        output = model(img)
        loss = criterion(output,labels)
        
                 
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1)%100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}: Loss={loss.item():.4f}')

# test
model.eval()

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for i,(input,labels) in enumerate(test_loader):
        input = input.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        
        output = model(input)
        
        _,preds = torch.max(output,1)
        
        n_samples+=labels.shape[0]
        n_correct += (preds == labels).sum().item()
print(f'Test Accuracy: {n_correct/n_samples}')
    