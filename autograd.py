import torch

# x = torch.randn(3,requires_grad=True)
# print(x)

# y = x + 2
# print(y)

# z = y*y*2
# # z = z.mean()
# print(z)

# v = torch.tensor([0.1,1.0,0.001],dtype=torch.float32)
# z.backward(v) # Calculate dz/dx  -> Vector Jacobian Product

# print(x.grad) # prints gradients of x

# #---------------------------------#

# x = torch.randn(3,requires_grad=True)
# print(x)

# # To prevent from calculating gradients for x

# # x.requires_grad_(False)
# # x.requires_grad_(False)
# # print(x)

# # x.detach()
# # y = x.detach()
# # print(y)

# # with torch.no_grad():
# #   operations
# with torch.no_grad():
#     y = x + 2
#     print(y)


weights = torch.ones(4,requires_grad=True)

for epoch in range(5):
    print(weights)
    model_output = (weights*3).sum()

    print(model_output)
    model_output.backward()
    
    print(weights.grad)
    
    weights.grad.zero_()





