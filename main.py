from KBNet import get_KBNet
from EdgeLoss import get_edge_loss
import torch

model = get_KBNet(classes=1, bd=4)
input = torch.randn(4, 1, 224, 224)
label = torch.randn(4, 1, 224, 224)
output, *aux = model(input)
print(output.shape)
edge_loss = get_edge_loss(torch.sigmoid(output), label, channel=input.shape[1])
print(edge_loss)