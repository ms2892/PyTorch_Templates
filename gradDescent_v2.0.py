import torch
import torch.nn as nn

# 1) Design Model (input,output size, forward pass)
# 2) Construct Loss and Optimizer
# 3) Training Loop
#     - forward pass: Compute the prediction
#     - backward pass: Computer the gradients
#     - update weights: using something like gradient descent etc



# Here we will replace the Loss and optimizer


# f = w * x
# No bias included

# f = 2 * x (Generating Function)

# Training Inputs
X = torch.tensor([1,2,3,4],dtype=torch.float32)

# Training Outputs
Y = torch.tensor([2,4,6,8],dtype=torch.float32)

# Initialize weights
w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

# Model Prediction
def forward(x):
    # Function
    # Returns the Forward Pass of the model
    # Inputs:
    #   x   (tensor/np.array)    Input Data
    # Outputs:
    #   (tensor/np.array)        Predicted values [For Linear Regression = w*x]    
    return w * x

print(f'Prediction before training: f(5) = {forward(5):.3f}')


# Training 
learning_rate = 0.01
n_iters = 20

loss = nn.MSELoss()  # Wrapper function to calculate the MSE Loss
optimizer = torch.optim.SGD([w],lr = learning_rate)   # Stochastic Gradient Descent

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    
    # Loss
    l = loss(Y,y_pred)
    
    # Gradients = backward pass
    l.backward()  # dl/dw -> Not as accurate as calculating numerically 
    
    # Update the weights (Stochastic Gradient Descent)
    optimizer.step()
    
    # zero the gradients
    optimizer.zero_grad()
    
    if epoch % 2 ==0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
        
print(f'Prediction after training: f(5) = {forward(5):.3f}')
