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

# Training Inputs - For Linear Layer it has to be a 2D shape batches X features

# Model Wrapper
class LinearRegression(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        
        self.lin = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.lin(x)
    

X = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)

# Training Outputs - batches X Output
Y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

# Test Input
X_test = torch.tensor([[5],[6],[7],[8]],dtype=torch.float32)

n_samples, n_features = X.shape

print(n_samples,n_features)

input_size = n_features
output_size = n_features


model = LinearRegression(input_size,output_size)



print(f'Prediction before training: f(5) = {model(X_test)}')


# Training 
learning_rate = 0.01
n_iters = 1000

loss = nn.MSELoss()  # Wrapper function to calculate the MSE Loss
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)   # Stochastic Gradient Descent

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)
    
    # Loss
    l = loss(Y,y_pred)
    
    # Gradients = backward pass
    l.backward()  # dl/dw -> Not as accurate as calculating numerically 
    
    # Update the weights (Stochastic Gradient Descent)
    optimizer.step()
    
    # zero the gradients
    optimizer.zero_grad()
    
    if epoch % 10 ==0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f} loss = {l:.8f}')
        
print(f'Prediction after training: f(5) = {model(X_test)}')
