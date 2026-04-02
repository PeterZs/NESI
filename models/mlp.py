import torch.nn as nn
import torch
import numpy as np
from typing import Callable, Optional


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def last_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)


def init_weights_normal(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        nn.init.zeros_(m.bias)


def init_weights_selu(m):
    if hasattr(m, 'weight'):
        num_input = m.weight.size(-1)
        nn.init.normal_(m.weight, std=1 / np.sqrt(num_input))
        nn.init.zeros_(m.bias)


def init_weights_elu(m):
    if hasattr(m, 'weight'):
        num_input = m.weight.size(-1)
        nn.init.normal_(m.weight, std=np.sqrt(1.5505188080679277) / np.sqrt(num_input))
        nn.init.zeros_(m.bias)


# The MLP implementation by nerfacc
class BaseMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = None,  # The number of input tensor channels.
        output_dim: int = None,  # The number of output tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = None,  # The layer to add skip layers to.
        hidden_activation: str = 'relu',
        output_enabled: bool = True,
        output_activation: str = None,
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init, sine_init),#last_layer_sine_init
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None, None),
                         'sigmoid':(nn.Sigmoid(), nn.init.xavier_normal_, None, None),
                         'tanh':(nn.Tanh(), nn.init.xavier_normal_, None, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None, None),
                         'softplus':(nn.Softplus(beta=100), init_weights_normal, None, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None, None)}

        self.hidden_activation, self.hidden_init, self.first_layer_init, self.last_layer_init = nls_and_inits[hidden_activation]

        self.output_enabled = output_enabled
        self.output_activation = output_activation
        if output_activation == "relu":
            self.output_activation = nn.ReLU()
        elif output_activation == "softplus":
            self.output_activation = nn.Softplus(beta=100)
        else:
            self.output_activation = None

        self.bias_enabled = bias_enabled

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(
                nn.Linear(in_features, self.net_width, bias=bias_enabled)
            )
            if (
                (self.skip_layer is not None)
                and (i % self.skip_layer == 0)
                and (i > 0)
            ):
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(
                in_features, self.output_dim, bias=bias_enabled
            )
        else:
            self.output_dim = in_features

        self.initialize()

    def initialize(self):

        if self.hidden_init is not None:
            self.hidden_layers.apply(self.hidden_init)

        if self.first_layer_init is not None and len(self.hidden_layers) > 0: # Apply special initialization to first layer, if applicable.
            self.hidden_layers[0].apply(self.first_layer_init)

        if self.output_enabled and self.last_layer_init is not None:
            self.output_layer.apply(self.last_layer_init)

    def forward(self, x):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            # if self.last_activation_relu:
            #     if i < self.net_depth - 1:
            #         x = self.hidden_activation(x)
            #     else:
            #         x = nn.ReLU()(x)
            # else:
            x = self.hidden_activation(x)
            if (
                (self.skip_layer is not None)
                and (i % self.skip_layer == 0)
                and (i > 0)
            ):
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            if self.output_activation is not None:
                x = self.output_activation(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = None,  # The layer to add skip layers to.
        pe: bool = False,
        latent_dim: int = 0,
        **kwargs
    ) -> None:
        super().__init__()
        self.pe = pe
        self.latent_dim = latent_dim
        if self.pe:
            self.pos_encoder = SinusoidalEncoder(input_dim, 0, 10, True)
            mlp_input_dim = self.pos_encoder.latent_dim + self.latent_dim
        else:
            mlp_input_dim = input_dim + self.latent_dim

        self.mlp = BaseMLP(
            input_dim=mlp_input_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            **kwargs
        )

    def forward(self, x, latent_vec=None):
        if self.pe:
            x = self.pos_encoder(x)

        x = self.mlp(x)
        return x



class DualMLPNet(nn.Module):
    def __init__(
            self,
            arch = 'MLP',
            **kwargs
    ):
        from models import OctreeSDF
        super(DualMLPNet, self).__init__()
        if arch == 'MLP':
            self.mlp_front = MLP(**kwargs) # MLPNet(**kwargs)
            self.mlp_back = MLP(**kwargs) # MLPNet(**kwargs)
        else:
            self.mlp_front = OctreeSDF(**kwargs)#MLPNet(**kwargs)
            self.mlp_back = OctreeSDF(**kwargs)#MLPNet(**kwargs)

    def forward(self, input, write_debug=False):
        front_h = self.mlp_front(input)
        back_h = self.mlp_back(input)
        return torch.concatenate([front_h, back_h], dim=-1)