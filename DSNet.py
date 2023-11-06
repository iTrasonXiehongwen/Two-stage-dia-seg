import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ConvNext_Downsampling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=2,
                      stride=2,
                      bias=False), nn.BatchNorm2d(out_channels, eps=1e-6))

    def forward(self, x):
        x = self.downsample(x)
        return x


class ConvNextBlock(nn.Module):

    def __init__(self, dim, out_dim, kernel=7):
        super().__init__()
        padding = int((kernel - 1) / 2)
        self.dwconv = nn.Conv2d(dim,
                                dim,
                                kernel_size=kernel,
                                padding=padding,
                                groups=dim,
                                bias=False)  # depthwise conv
        self.norm = nn.BatchNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1, 1, groups=dim)
        self.act = nn.SiLU()
        self.pwconv2 = nn.Conv2d(4 * dim,
                                 out_dim,
                                 1,
                                 1,
                                 groups=min(4 * dim, out_dim))
        self.dp = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.dp(x)
        return x


class ConvNextLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_layer = ConvNext_Downsampling(in_channels, out_channels)
        self.block = ConvNextBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.down_layer(x)
        x = self.block(x)
        return x


class Unet_Downsampling(nn.Module):

    def __init__(self):
        super().__init__()
        self.downsample = nn.Sequential(nn.MaxPool2d(2))

    def forward(self, x):
        x = self.downsample(x)
        return x


class UnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class UnetLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_layer = Unet_Downsampling()
        self.block = UnetBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.down_layer(x)
        x = self.block(x)
        return x


class scale_cat(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, feature, feature_extra=None):
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        if feature_extra is not None:
            feature_extra = F.interpolate(input=feature_extra,
                                          scale_factor=0.5,
                                          mode='bilinear',
                                          align_corners=True,
                                          recompute_scale_factor=True)
            return torch.cat((x, feature, feature_extra), dim=1)
        else:
            return torch.cat((x, feature), dim=1)


class OutHead(nn.Module):

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


class MobileNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, expand_ratio=1, k=3):
        super().__init__()
        mid_channels = out_channels * expand_ratio
        self.conv1 = conv1x1(in_channels, mid_channels)
        self.conv2 = conv3x3(mid_channels, mid_channels, True, k=k)
        self.conv3 = conv1x1(mid_channels, out_channels)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Conf:

    def __init__(self,
                 encoder='unet',
                 decoder='unet',
                 early_output=False,
                 fuse_stride2=False,
                 base_coe=64,
                 deep_supervision=False):
        self.encoder = encoder
        self.decoder = decoder
        self.early_output = early_output
        self.fuse_stride2 = fuse_stride2
        self.base_coe = base_coe
        self.deep_supervision = deep_supervision

class MyModel(nn.Module):

    def __init__(self, conf: Conf):
        super().__init__()
        self.conf = conf
        c = conf.base_coe
        if conf.encoder == 'unet':
            self.stem = UnetBlock
            self.downbase = UnetLayer
        if conf.encoder == 'convnext':
            self.stem = ConvNextBlock
            self.downbase = ConvNextLayer

        if conf.decoder == 'unet':
            self.upbase = UnetBlock
        if conf.decoder == 'convnext':
            self.upbase = ConvNextBlock
        if conf.decoder == 'mobilenet':
            self.upbase = MobileNetBlock

        self.down1 = self.stem(1, c)
        self.down2 = self.downbase(c, c * 2)
        self.down3 = self.downbase(c * 2, c * 4)
        self.down4 = self.downbase(c * 4, c * 8)
        self.down5 = self.downbase(c * 8, c * 8)

        self.up1 = self.upbase(c * 16, c * 4)
        self.up2 = self.upbase(c * 8, c * 2)

        if conf.fuse_stride2:
            self.up3 = self.upbase(c * 5, c)
        else:
            self.up3 = self.upbase(c * 4, c)

        self.up4 = self.upbase(c * 2, c)

        class_num = 1
        if not conf.early_output:
            self.out = OutHead(c, class_num, 1)
        if conf.early_output:
            self.out = OutHead(c, class_num, 2)

        self.scale_cat = scale_cat()

        self.axu0 = OutHead(c * 8, class_num, 2)
        self.aux1 = OutHead(c * 4, class_num, 2)
        self.aux2 = OutHead(c * 2, class_num, 2)

    def forward(self, x):
        x1 = self.down1(x)  #320
        x2 = self.down2(x1)  #160
        x3 = self.down3(x2)  #80
        x4 = self.down4(x3)  #40
        x5 = self.down5(x4)  #20
        if self.training:
            aux0 = self.axu0(x5)
        x = self.up1(self.scale_cat(x5, x4))  #40
        if self.training:
            aux1 = self.aux1(x)
        x = self.up2(self.scale_cat(x, x3))  #80
        if self.training:
            aux2 = self.aux2(x)
        if self.conf.early_output:
            if self.conf.fuse_stride2:
                x = self.up3(self.scale_cat(x, x2, x1))  #160
            else:
                x = self.up3(self.scale_cat(x, x2))
        else:
            x = self.up3(self.scale_cat(x, x2))
            x = self.up4(self.scale_cat(x, x1))

        x = self.out(x)
        if self.training:
            return [x, aux2, aux1, aux0]
        else:
            return x

def run_model(model, size_=(224, 224)):
    import numpy as np
    img = np.random.rand(*size_) * 255
    input = ((np.array(img))[:, :, np.newaxis] / 255.).astype(np.float32)
    input = torch.tensor((input.transpose((2, 0, 1)))[np.newaxis, :])
    input = torch.cat([input, input], dim=0)
    preds = model(input)
    print(f"preds:{preds.shape}")
    
def see_params(model, name="model", shape=(1, 1, 224, 224)):
    from thop import profile
    from thop import clever_format
    input = torch.rand(*shape)
    model.eval()
    macs, params = profile(model, inputs=(input, ), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print(f'{name}: ', end='')
    print(f'FLOPs:{macs}, params:{params}')
    
if __name__ == "__main__":
    conf = Conf(base_coe=8,
                encoder='convnext',
                decoder='mobilenet',
                early_output=True,
                fuse_stride2=True)
    DSNet = MyModel(conf)
    DSNet.eval()
    try:
        run_model(DSNet)
        #! requires thop lib to see params
        see_params(DSNet,shape=(1,1,224,224))
    except:
        pass
