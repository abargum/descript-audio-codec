import torch
from rave.blocks2 import Resnet1dCC

netCC = Resnet1dCC()
t = torch.rand(16, 64, 1, 191)
batch_size = t.shape[0]
t = t.flatten(0, 1)
print(t.shape)
f0 = netCC(t)
print(f0.shape)
f0 = f0.view(batch_size, -1, f0.size(-1))
print(f0.shape)