import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TransformerLayer, self).__init__()
        self.w_q = nn.Linear(input_dim, hidden_dim)
        self.w_k = nn.Linear(input_dim, hidden_dim)
        self.w_v = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Map input to hidden_dim dimension
        x = nn.Linear(input_dim, hidden_dim)(x)

        # Q, K, V calculation
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Self-attention
        attn_output, _ = self.attention(q, k, v)
        x = self.layer_norm1(x + attn_output)

        # Feed-forward
        ff_output = self.feed_forward(x)
        return self.layer_norm2(x + ff_output)

# 使用示例
batch_size = 32
sequence_length = 100
input_dim = 512
hidden_dim = 64

model = TransformerLayer(input_dim, hidden_dim)
input_data = torch.randn(batch_size, sequence_length, input_dim)
output = model(input_data)
print(output.shape)
