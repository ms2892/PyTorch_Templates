import numpy as np

# f = w * x
# No bias included

# f = 2 * x (Generating Function)

# Training Inputs
X = np.array([1,2,3,4],dtype=np.float32)

# Training Outputs
Y = np.array([2,4,6,8],dtype=np.float32)

# Initialize weights
w = 0.0

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

# Gradients
# MSE = 1/N * (w*x - y)**2
# dMSE/dw = 1/N 2x (w*x - y)
def gradient(x,y,y_predicted):
    # Function
    # Returns the gradient of the Loss Function
    # Inputs:
    #   x             (tensor/np.array)   Input Points
    #   y             (tensor/np.array)   Ground Truth
    #   y_predicted   (tensor/np.array)   Predicted Value
    # Outputs:
    #   (tensor/np.array)   Returns the gradient of the loss wrt weights [dMSE/dw = 1/N 2x (w*x - y)]
    return np.dot(2*x,(y_predicted-y)).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')


# Training 
learning_rate = 0.01
n_iters = 10
for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    
    # Loss
    l = loss(Y,y_pred)
    
    # Gradients
    dw = gradient(X,Y,y_pred)
    
    # Update the weights (Gradient Descent)
    w = w - learning_rate * dw
    
    if epoch % 1 ==0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
        
print(f'Prediction after training: f(5) = {forward(5):.3f}')
