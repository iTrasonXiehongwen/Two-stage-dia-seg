import torch
import torch.nn.functional as F
from torch import nn

class Sobel(nn.Module):

    def __init__(self, channel):
        super(Sobel, self).__init__()
        kernel_x = [[-3, 0., 3], [-10, 0., 10], [-3, 0., 3]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).expand(
            channel, 3, 3).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        #! for gpu:
        # self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).cuda()

        kernel_y = [[-3, -10, -3], [0., 0., 0.], [3, 10, 3]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).expand(
            channel, 3, 3).unsqueeze(0)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
        #! for gpu:
        # self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).cuda()


    def forward(self, pred):
        edge_x = F.conv2d(pred, self.weight_x, padding=1)
        abs_x = torch.abs(edge_x)
        edge_y = F.conv2d(pred, self.weight_y, padding=1)
        abs_y = torch.abs(edge_y)
        z = abs_x + abs_y
        out = torch.tanh(z)

        return out


def get_edge_loss(pred: torch.tensor, target: torch.tensor, channel=1):
    #! when tensor shape is BCHW, channel=C

    # pred:BCHW, target: BCHW
    if (len(pred.shape) == 4) and (len(target.shape) == 4):
        pred_edge = Sobel(channel)(pred)
        target_edge = Sobel(channel)(target.float())
        inter = (pred_edge * target_edge).sum(dim=1).sum(dim=1).sum(dim=1)
        union = (pred_edge + target_edge).sum(dim=1).sum(dim=1).sum(dim=1)

    # pred:BHW, target: BHW
    elif (len(pred.shape) == 3) and (len(target.shape) == 3):
        pred_edge = Sobel(channel)(pred.unsqueeze(1))[:, 0]
        target_edge = Sobel(channel)(target.unsqueeze(1).float())[:, 0]
        inter = (pred_edge * target_edge).sum(dim=1).sum(dim=1)
        union = (pred_edge + target_edge).sum(dim=1).sum(dim=1)

    else:
        raise ValueError('Wrong input shape')

    edge_loss = 1 - 2 * (inter + 1) / (union + 2)
    return edge_loss.mean()