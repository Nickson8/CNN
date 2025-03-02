import torch

# Create a 2D tensor
tensor = torch.tensor([[2, 3], [4, 5], [6, 7]])

# Create a column of ones
ones_column = torch.ones((tensor.shape[0], 1))

# Concatenate along dimension 1 (columns)
new_tensor = torch.cat((tensor, ones_column), dim=1)

print(new_tensor)

new_tensor = new_tensor[:, :-1]

print(new_tensor)
