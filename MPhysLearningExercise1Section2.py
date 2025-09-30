import torch
tensor1 = torch.tensor([1,2,3])
tensor2 = torch.tensor([4,5,6])

tensor3 = torch.stack((tensor1,tensor2,), dim=0)
print(tensor3)
tensor4 = torch.stack((tensor1,tensor2,), dim=1)
print(tensor4)

print(tensor1.reshape(-1,1))
print(tensor1.reshape(-1,1).shape)

print(tensor3.reshape(-1,1))
print(tensor4.reshape(-1,1))

tensor1 = torch.rand(100)
mask = tensor1 > 0.5

print(tensor1.shape)
print(mask)

filtered_tensor = tensor1[mask]
print(filtered_tensor)
print(filtered_tensor.shape)

max_index = torch.argmax(tensor1)
min_index = torch.argmin(tensor1)

print(max_index, min_index)