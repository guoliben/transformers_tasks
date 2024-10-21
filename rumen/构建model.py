import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'mps'
print(f'Using {device} device')

class NeuralNetworkXXXXX(nn.Module):
    def __init__(self):
        super(NeuralNetworkXXXXX, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Dropout(p=0.2)
        )
        print(self.flatten)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetworkXXXXX().to(device)
print(model)

#
# class A:
#     def some_method(self):
#         print("A method", self)
#
#
# A().some_method()
