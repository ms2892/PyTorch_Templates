import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0,requires_grad=True)

# Forward Pass and Compute the Loss

y_hat = w*x
loss = (y_hat-y)**2

print(loss)

# Backward Pass

loss.backward()

print(w.grad)

# Update the Weights


# Next forward and backward pass for a couple of iterations