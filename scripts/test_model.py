import torch
import argbind
from rave.rave_model import RAVE
from rave.blocks import GeneratorV2

RAVE = RAVE().to('cuda')
t = torch.rand(1, 1, 65536).to('cuda')
print(RAVE(t)["audio"].shape)

GEN = GeneratorV2(data_size = 16,
                  capacity = 16,
                  ratios = [4, 4, 2, 2],
                  latent_size = 64 + 256,
                  kernel_size = 3,
                  dilations = [[1, 3, 9], [1, 3, 9], [1, 3, 9], [1, 3]])

t = torch.rand(1, 320, 64)
p = torch.rand(1, 1, 64)
y, pi = GEN(t, p)
print(y.shape, pi.shape)