import torch
import numpy as np

mean = 0
standard_deviation = 100
px = mean + (standard_deviation * torch.randn(1000))
py = mean + (standard_deviation * torch.randn(1000))
pz = mean + (standard_deviation * torch.randn(1000))

energy = torch.sqrt(px**2 + py**2 + pz**2)

full_info = torch.stack((px, py, pz, energy), dim=1)

print(full_info)

max_energy = torch.argmax(full_info[:,3])
min_energy = torch.argmin(full_info[:,3])


pz_closest_to_mean = torch.argmin(torch.abs(torch.abs(full_info[:,2]) - mean))

print(max_energy)
print(min_energy)
print(pz_closest_to_mean)

mask = np.sqrt(((full_info[:,0]**2) + (full_info[:,1]**2))) > 50
filtered_info = full_info[mask]
print(filtered_info)
print(filtered_info.shape)

# Z-score Normalisation

mean_values = torch.mean(filtered_info, dim=0)
std_values = torch.std(filtered_info, dim=0)

z_normalised_info = (filtered_info - mean_values) / std_values
print(z_normalised_info)

# Other Normalisation (Between -1 and +1))

min_values = torch.min(filtered_info, dim=0).values
max_values = torch.max(filtered_info, dim=0).values

range = (torch.abs(max_values) + torch.abs(min_values))
new_central_point = max_values - (range / 2)

other_normalised_info = (filtered_info - new_central_point) / (range / 2)
print(other_normalised_info)