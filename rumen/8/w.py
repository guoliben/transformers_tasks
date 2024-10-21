import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TransformerLayer, self).__init__()
        self.w_q = nn.Linear(input_dim, hidden_dim)
        self.w_k = nn.Linear(input_dim, hidden_dim)
        self.w_v = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        return q, k, v

model = TransformerLayer(input_dim, hidden_dim)

