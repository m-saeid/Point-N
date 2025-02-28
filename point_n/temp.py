import torch

from models.point_gn_light import PointGNPCls

batch_size = 32
num_points = 1024
in_ch = 3

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create random input data
random_xyz = torch.randn(batch_size, num_points, in_ch).to(device)
# [B, N, 3]

# Initialize the model
model = PointGNPCls(num_points=1024).to(device)



# Run the model
output = model(random_xyz)

print(output.requires_grad)