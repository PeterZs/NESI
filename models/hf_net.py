import torch
import torch.nn as nn
from models.mlp import MLP

class HFNet(nn.Module):
    def __init__(
            self,
            net_width,
            net_depth,
            input_dim,
            **kwargs
    ):
        super(HFNet, self).__init__()

        self.mlp_hf = MLP(net_width=net_width, net_depth=net_depth, input_dim=input_dim, output_dim=1, **kwargs)
        self.mlp_mask = MLP(net_width=16, net_depth=3, input_dim=input_dim, output_dim=1, **kwargs)


    def forward(self, input):
        hf = self.mlp_hf(input)
        mask = self.mlp_mask(input)
        return torch.cat([hf, mask], dim=-1)


