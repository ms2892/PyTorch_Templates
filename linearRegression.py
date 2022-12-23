# Basic Training Pipeline
# 1) Design Model (input,output size, forward pass)
# 2) Construct Loss and Optimizer
# 3) Training Loop
#     - forward pass: Compute the prediction
#     - backward pass: Computer the gradients
#     - update weights: using something like gradient descent etc

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare Data
X_numpy,Y_numpy = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))

# Convert Y to column vector
Y = Y.view(Y.shape[0],1)

n_samples,n_features = X.shape

# 1) Design the Model 

class LinearRegression(nn.Module):
    
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        
        # Define Layers
        self.lin = nn.Linear(input_dim,output_dim)
        
    def forward(self,x):
        
        # Forward Pass
        return self.lin(x)

input_size = n_features
output_size = 1
model = LinearRegression(input_size,output_size)

# 2) Construct Loss and Optimizer
learning_rate=0.01
creterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# 3) Training Loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward Pass
    
    y_pred = model(X)
    
    # Loss
    l = creterion(Y,y_pred)
    
    # Backward Pass
    l.backward()
    
    # Update
    optimizer.step()
    
    optimizer.zero_grad()
    
    if (epoch+1)%10==0:
        print(f'Epoch {epoch+1}: Loss={l.item():.4f}')

# Plot the values

predicted = model(X).detach().numpy()
        
plt.plot(X_numpy,Y_numpy,'go')
plt.plot(X_numpy,predicted,'r')

plt.show()