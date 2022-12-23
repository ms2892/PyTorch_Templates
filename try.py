import torch
# Torch add

# x + y | torch.add(x,y) | y.add_(x)

# Torch Multiplication

# x * y | torch.mul(x,y) | y.mul_(x)

# Subtraction

# x - y | torch.sub(x,y) | y.sub_(x)

# Division

# x / y | torch.div(x,y) | y.div_(x)

# Slicing Operations

# x[:,0] -> Like Numpy

# Convert to value

# x[1,1].item() -> gets the actual value at (1,1)

# Reshaping tensor x->4X4

# y = x.view(16)
# y = x.view(-1,8)  -> automatically computes the place at -1



import numpy as np

# Torch tensor to Numpy
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)  # Both point to the same memory location
print(b)

# Numpy to Torch Tensor
c= np.ones(5)
print(c)
d = torch.from_numpy(c)
print(d)   # Same issue as before same memory location


# Numpy can only handle cpu tensors

# device = torch.device('cuda')
# x = torch.ones(5,device=device) -> puts it to GPU




# Requires Grad

x = torch.ones(5,requires_grad=True) # Tells Pytorch that it Needs to calculate the gradients for this tensor
print(x)


