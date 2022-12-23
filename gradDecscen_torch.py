import torch

# 1) Design Model (input,output size, forward pass)
# 2) Construct Loss and Optimizer
# 3) Training Loop
#     - forward pass: Compute the prediction
#     - backward pass: Computer the gradients
#     - update weights: using something like gradient descent etc



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

# Loss = MSE
def loss(y,y_predicted):
    # Function
    # Returns the Loss value of the predicted values and output
    # Inputs:
    #   y             (tensor/np.array)   Ground Truth
    #   y_predicted   (tensor/np.array)   Predicted Value
    # Outputs:
    #   (tensor/np.array)   Returns the Loss calcuated [For this example we are using MSE]
    return ((y_predicted-y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')


# Training 
learning_rate = 0.01
n_iters = 20
for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    
    # Loss
    l = loss(Y,y_pred)
    
    # Gradients = backward pass
    l.backward()  # dl/dw -> Not as accurate as calculating numerically 
    
    # Update the weights (Gradient Descent)
    with torch.no_grad():  # No grad because we don't want it in tracking our gradient history
        w -= learning_rate * w.grad
    
    # zero the gradients
    w.grad.zero_()
    
    if epoch % 2 ==0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
        
print(f'Prediction after training: f(5) = {forward(5):.3f}')
