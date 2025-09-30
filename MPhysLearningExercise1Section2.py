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