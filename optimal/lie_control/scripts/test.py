import torch
from lie_control.models.nn_models import MassFixed

device = torch.device('cuda')
eps = torch.Tensor([10., 10., 10.]).to(device)
model = MassFixed(m_dim=3, eps=eps, init_gain=1).to(device)

for name, param in model.named_parameters():
    if param.device != device:
        print(f"Parameter {name} is on device {param.device}")
        # param.data = param.data.to(self.device)
    if param.dtype != torch.float64:
        print(f"Parameter {name} has dtype {param.dtype}")
        # param.data = param.data.double()