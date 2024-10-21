import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchkeras #Attention this line 


#================================================================================
# 一，准备数据
#================================================================================

import torchvision 
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])
ds_train = torchvision.datasets.MNIST(root="mnist/",train=True,download=True,transform=transform)
ds_val = torchvision.datasets.MNIST(root="mnist/",train=False,download=True,transform=transform)
dl_train =  torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=2)
dl_val =  torch.utils.data.DataLoader(ds_val, batch_size=128, shuffle=False, num_workers=2)

for features,labels in dl_train:
    break 

#================================================================================
# 二，定义模型
#================================================================================


def create_net():
    net = nn.Sequential()
    net.add_module("conv1",nn.Conv2d(in_channels=1,out_channels=64,kernel_size = 3))
    net.add_module("pool1",nn.MaxPool2d(kernel_size = 2,stride = 2))
    net.add_module("conv2",nn.Conv2d(in_channels=64,out_channels=512,kernel_size = 3))
    net.add_module("pool2",nn.MaxPool2d(kernel_size = 2,stride = 2))
    net.add_module("dropout",nn.Dropout2d(p = 0.1))
    net.add_module("adaptive_pool",nn.AdaptiveMaxPool2d((1,1)))
    net.add_module("flatten",nn.Flatten())
    net.add_module("linear1",nn.Linear(512,1024))
    net.add_module("relu",nn.ReLU())
    net.add_module("linear2",nn.Linear(1024,10))
    return net

net = create_net()
print(net)

# 评估指标
class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

        self.correct = nn.Parameter(torch.tensor(0.0),requires_grad=False)
        self.total = nn.Parameter(torch.tensor(0.0),requires_grad=False)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.argmax(dim=-1)
        m = (preds == targets).sum()
        n = targets.shape[0] 
        self.correct += m 
        self.total += n
        
        return m/n

    def compute(self):
        return self.correct.float() / self.total 
    
    def reset(self):
        self.correct -= self.correct
        self.total -= self.total
        


#================================================================================
# 三，训练模型
#================================================================================

model = torchkeras.KerasModel(net,
      loss_fn = nn.CrossEntropyLoss(),
      optimizer= torch.optim.Adam(net.parameters(),lr=0.001),
      metrics_dict = {"acc":Accuracy()}
    )

from torchkeras import summary
summary(model,input_data=features);


# if gpu/mps is available, will auto use it, otherwise cpu will be used.

dfhistory=model.fit(train_data=dl_train, 
                    val_data=dl_val, 
                    epochs=15, 
                    patience=5, 
                    monitor="val_acc",mode="max",
                    ckpt_path='checkpoint.pt')

#================================================================================
# 四，评估模型
#================================================================================

model.evaluate(dl_val)


#================================================================================
# 五，使用模型
#================================================================================

model.predict(dl_val)[0:10]

#================================================================================
# 六，保存模型
#================================================================================
# The best net parameters  has been saved at ckpt_path='checkpoint.pt' during training.
net_clone = create_net() 
net_clone.load_state_dict(torch.load("checkpoint.pt"))



