import gin
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CouplingLayerNICE(nn.Module):
    def __init__(self, input_dim, cond_dim, n_layers, mask_type, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.mask = self.get_mask(input_dim, mask_type)

        layers = []
        in_dim = input_dim + cond_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.1))

        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.1))

        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, cond):
        # x: [B, input_dim]
        # cond: [B, cond_dim]
        z = x.view(x.size(0), -1)
        h1 = z * self.mask
        h2 = z * (1 - self.mask)

        inp = torch.cat([h1, cond], dim=1)
        m = self.net(inp) * (1 - self.mask)

        h2 = h2 + m
        out = h1 + h2
        return out.view_as(x)

    def inverse(self, z, cond):
        x = z.view(z.size(0), -1)
        h1 = x * self.mask
        h2 = x * (1 - self.mask)

        inp = torch.cat([h1, cond], dim=1)
        m = self.net(inp) * (1 - self.mask)

        h2 = h2 - m
        out = h1 + h2
        return out.view_as(z)

    def get_mask(self, input_dim, mask_type):
        mask = torch.zeros(input_dim)
        if mask_type == 0:
            mask[::2] = 1
        else:
            mask[1::2] = 1
        return mask.view(1, -1).to(device).float()


@gin.configurable
class FlowModel(nn.Module):
    def __init__(self, input_dim, cond_dim, n_layers, n_couplings, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        self.couplings = nn.ModuleList([
            CouplingLayerNICE(
                input_dim=input_dim,
                cond_dim=cond_dim,
                n_layers=n_layers,
                mask_type=i % 2,
                hidden_dim=hidden_dim
            ).to(device).float()
            for i in range(n_couplings)
        ])

    def forward(self, x, cond):
        return self.flow(x, cond)

    def flow(self, x, cond):
        x = x.view(-1, self.input_dim).float()
        cond = cond.view(-1, self.cond_dim).float()

        for layer in self.couplings:
            x = layer(x, cond)
        return x

    def inv_flow(self, z, cond):
        z = z.view(-1, self.input_dim).float()
        cond = cond.view(-1, self.cond_dim).float()

        for layer in reversed(self.couplings):
            z = layer.inverse(z, cond)
        return z

    def load_w(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path, map_location=device))
