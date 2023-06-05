import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):

    def __init__(self, dim, out_dim, kernel=7):
        super().__init__()
        padding = int((kernel - 1) / 2)
        self.dwconv = nn.Conv2d(dim,
                                dim,
                                kernel_size=kernel,
                                padding=padding,
                                groups=dim,
                                bias=False)  
        self.norm = nn.BatchNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1, 1, groups=dim)
        self.act = nn.SiLU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1, 1, groups=dim)
        self.dp = nn.Dropout(0.1)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.dp(x)
        return x


class UpSampling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.interpolate(x,
                             scale_factor=2,
                             mode='bilinear',
                             align_corners=True)

def conv3x3(in_channels, out_channels, depthwise=False, s=1, k=3):
    p = int((k - 1) / 2)
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=(k, k),
                  stride=(s, s),
                  padding=(p, p),
                  groups=out_channels if depthwise else 1,
                  bias=False),
        nn.BatchNorm2d(out_channels,
                       eps=1e-05,
                       momentum=0.1,
                       affine=True,
                       track_running_stats=True), nn.ReLU(inplace=True),
        nn.Dropout(0.1))


def conv1x1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=(1, 1),
                  stride=(1, 1),
                  bias=False),
        nn.BatchNorm2d(out_channels,
                       eps=1e-05,
                       momentum=0.1,
                       affine=True,
                       track_running_stats=True), nn.ReLU(inplace=True),
        nn.Dropout(0.1))


class depth_conv(nn.Module):

    def __init__(self, in_channels, out_channels, expand_ratio=1, k=3):
        super(depth_conv, self).__init__()
        mid_channels = out_channels * expand_ratio
        self.conv1 = conv1x1(in_channels, mid_channels)
        self.conv2 = conv3x3(mid_channels, mid_channels, True, k=k)
        self.conv3 = conv1x1(mid_channels, out_channels)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class UpBlock(nn.Module):

    def __init__(self, dim, add_dim, out_dim):
        super().__init__()
        self.up = UpSampling()
        self.block = depth_conv(dim + add_dim, out_dim, 6, k=3)

    def forward(self, xup, xdown, x_extra=None):
        xup = self.up(xup)
        if x_extra is not None:
            x_extra = F.interpolate(input=x_extra,
                                    scale_factor=0.5,
                                    mode='bilinear',
                                    align_corners=True,
                                    recompute_scale_factor=True)
            x = torch.cat((xup, xdown, x_extra), dim=1)
        else:
            x = torch.cat((xup, xdown), dim=1)
        x = self.block(x)
        return x


class SimpleSegHead(nn.Module):

    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        self.conv1 = depth_conv(in_channels, 16 * out_channels)
        self.conv3 = nn.Conv2d(16 * out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        self.scale = scale

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv3(x)
        x = F.interpolate(x,
                          scale_factor=self.scale,
                          mode='bilinear',
                          align_corners=True)
        return x


class DiaSegNet_base(nn.Module):

    def __init__(self,
                 in_chans=3,
                 depths=[1, 1, 1, 1, 1],
                 dims=[96, 192, 384, 768, 100],
                 classes=1):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dims[0], eps=1e-6))
        self.downsample_layers.append(stem)
        for i in range(4):
            downsample_layer = nn.Sequential(
                nn.Conv2d(dims[i],
                          dims[i + 1],
                          kernel_size=2,
                          stride=2,
                          bias=False), nn.BatchNorm2d(dims[i + 1], eps=1e-6))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(5):
            stage = nn.Sequential(*[
                Block(dim=dims[i], out_dim=dims[i]) for j in range(depths[i])
            ])
            self.stages.append(stage)

        self.up4 = UpBlock(dim=dims[4], add_dim=dims[3], out_dim=dims[3])

        self.up3 = UpBlock(dim=dims[3], add_dim=dims[2], out_dim=dims[2])
        self.up2 = UpBlock(dim=dims[2],
                           add_dim=dims[1] + dims[0],
                           out_dim=dims[1])
        self.up1 = UpBlock(dim=dims[1], add_dim=dims[0], out_dim=dims[0])

        self.axu5 = SimpleSegHead(dims[4], classes, 2)
        self.aux4 = SimpleSegHead(dims[3], classes, 2)
        self.aux3 = SimpleSegHead(dims[2], classes, 2)
        self.aux2 = SimpleSegHead(dims[1], classes, 4)
        self.aux1 = SimpleSegHead(dims[0], classes, 2)

    def forward(self, x):
        down_features = []
        for i in range(5):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            down_features.append(x)
        if self.training:
            aux5 = self.axu5(down_features[4])
        x = self.up4(down_features[4], down_features[3])

        if self.training:
            aux4 = self.aux4(x)
        x = self.up3(x, down_features[2])

        if self.training:
            aux3 = self.aux3(x)
        x = self.up2(x, down_features[1], down_features[0])

        aux2 = self.aux2(x)

        x = self.up1(x, down_features[0])

        if self.training:
            return [aux2, aux3, aux4, aux5]
        else:
            return aux2


def get_KBNet(classes=1, bd=4):
    #! bd: base dim
    return DiaSegNet_base(in_chans=1,
                          dims=[bd, 2 * bd, 4 * bd, 8 * bd, 16 * bd],
                          classes=classes)


def see_params(model, name="model", shape=(1, 1, 320, 320)):
    from thop import profile
    from thop import clever_format
    input = torch.rand(*shape)
    model.eval()
    macs, params = profile(model, inputs=(input, ), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print(f'{name}: ', end='')
    print(f'FLOPs:{macs}, params:{params}')


if __name__ == "__main__":
    model = get_KBNet()
    try:
        #! requires thop lib to see params
        see_params(model,shape=(1,1,224,224))
    except:
        pass

