from DSNet import Conf, MyModel
from EdgeLoss import get_edge_loss
import torch

conf = Conf(base_coe=8,
            encoder='convnext',
            decoder='mobilenet',
            early_output=True,
            fuse_stride2=True)
model = MyModel(Conf)
input = torch.randn(4, 1, 224, 224)
label = torch.randn(4, 1, 224, 224)
output, *aux = model(input)
print(output.shape)
edge_loss = get_edge_loss(torch.sigmoid(output), label, channel=input.shape[1])
print(edge_loss)
