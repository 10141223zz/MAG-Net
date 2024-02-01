import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from functools import partial

nonlinearity = partial(F.relu, inplace=False)    # 必须设成Flase



class MRDM(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(MRDM, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(out_channel*3, out_channel, kernel_size=1, stride=1)
        self.BN = nn.BatchNorm2d(out_channel*3)
        self.RULE = nn.ReLU()
        self.conv6 = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x)
        out1 = torch.cat([x1, x2, x3], dim=1)
        out2 = self.BN(out1)
        out3 = self.conv5(out2)
        out4 = x4 + out3
        out5 = self.conv6(out4)
        out = self.RULE(out5)
        return out

class MRDM1(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(MRDM1, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(out_channel*3, out_channel, kernel_size=1, stride=1)
        self.BN = nn.BatchNorm2d(out_channel*3)
        self.RULE = nn.ReLU()
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x)
        out1 = torch.cat([x1, x2, x3], dim=1)
        out2 = self.BN(out1)
        out3 = self.conv5(out2)
        out4 = x4 + out3
        out = self.RULE(out4)
        return out

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.bn(out)
        out = nonlinearity(out)
        # out = self.dropout(out)
        return out

class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DenseBlock_light, self).__init__()
        out_channels_def = int(in_channels / 2)
        denseblock = []
        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class SE_block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(channel, channel//2, 2, 2, padding=0, output_padding=0, bias=True),
            nn.BatchNorm2d(channel//2),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(channel//2, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.up_conv(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        return out

class ConvBnRelu(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat

class ASPP_module(nn.Module):
    def __init__(self, in_chan, out_chan, dilation):
        super(ASPP_module, self).__init__()
        self.atrous_convolution = ConvBnRelu(in_chan,out_chan, ks=3, stride=1, padding=dilation,dilation=dilation)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Bilinear(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(Bilinear, self).__init__()
        self.conv = nn.Conv2d(in_chan,out_chan,3,1,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.conv(x))
        feat = nn.functional.interpolate(x, x.shape[2])
        return feat

class LinkNet34(nn.Module):

    def __init__(self, num_classes=2):
        super(LinkNet34, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv_opt = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        # self.firstmaxpool = resnet.maxpool


        self.encoder1 = MRDM(64, 64)
        self.encoder2 = MRDM(64, 128)
        self.encoder3 = MRDM(128, 256)
        self.encoder4 = MRDM1(256, 512)

        self.up2 = DecoderBlock(256, 128)
        self.up1 = DecoderBlock(128, 64)
        self.up0 = DecoderBlock(64, 32)
        self.decoder0 = DecoderBlock(32, 32)

        self.se2 = SE_block(256)
        self.se1 = SE_block(128)
        self.se0 = SE_block(64)

        self.out3_1 = DenseBlock_light(768, 256)
        self.out2_1 = DenseBlock_light(256, 128)
        self.out2_2 = DenseBlock_light(384, 128)
        self.out1_1 = DenseBlock_light(128, 64)
        self.out1_2 = DenseBlock_light(192, 64)
        self.out1_3 = DenseBlock_light(256, 64)
        self.out0_1 = DenseBlock_light(96, 32)
        self.out0_2 = DenseBlock_light(128, 32)
        self.out0_3 = DenseBlock_light(160, 32)
        self.out0_4 = DenseBlock_light(192, 32)

        dilations = [1, 2, 5]  # ,9]
        self.aspp1 = ASPP_module(512, 128, dilation=dilations[0])
        self.aspp2 = ASPP_module(512, 128, dilation=dilations[1])
        self.aspp3 = ASPP_module(512, 128, dilation=dilations[2])
        self.global_avg_pool = nn.Sequential(ConvBnRelu(512, 128, ks=3, stride=1, padding=1),
                                             nn.AdaptiveAvgPool2d((1, 1)))

        self.bilinear3_1 = Bilinear(256,2)
        self.bilinear2_2 = Bilinear(128,2)
        self.bilinear1_3 = Bilinear(64,2)
        self.bilinear0_1 = Bilinear(32,2)

        self.finalrelu1 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)



    def forward(self, x):
        # Encoder
        x = self.firstconv_opt(x)
        x = self.firstbn(x)
        ex_0 = self.firstrelu(x)
        # x = self.firstmaxpool(x)  # torch.Size([1, 64, 128, 128])

        ex_1 = self.encoder1(ex_0)  # torch.Size([1, 64, 128, 128])
        ex_2 = self.encoder2(ex_1)  # torch.Size([1, 128, 64, 64])
        ex_3 = self.encoder3(ex_2)  # torch.Size([1, 256, 32, 32])
        ex_4 = self.encoder4(ex_3)  # torch.Size([1, 512, 32, 32])

        # Decoder

        aspp_1 = self.aspp1(ex_4)  #aspp_1  torch.Size([1, 128, 32, 32])
        aspp_2 = self.aspp2(ex_4)  #torch.Size([1, 128, 32, 32])
        aspp_3 = self.aspp3(ex_4)  #torch.Size([1, 128, 32, 32])
        aspp_4 = self.global_avg_pool(ex_4) #torch.Size([1, 128, 1, 1])
        aspp_4 = F.interpolate(aspp_4, size=ex_4.size()[2:], mode='bilinear', align_corners=True) #torch.Size([1, 128, 32, 32])
        feat_aspp = torch.cat((aspp_1, aspp_2, aspp_3, aspp_4), dim=1) #torch.Size([1, 512, 32, 32])

        out3_1 = self.out3_1(torch.cat((ex_3, feat_aspp),dim=1))  # torch.Size([1, 256, 32, 32])
        # feat_2_32 = self.bilinear3_1(out3_1)                     # torch.Size([1, 2, 32, 32])
        up2_1 = self.up2(ex_3)                               # torch.Size([1, 128, 64, 64])
        up2_2 = self.up2(out3_1)                               # torch.Size([1, 128, 64, 64])
        se2_2 = self.se2(out3_1)                              # torch.Size([1, 128, 64, 64])
        feat2_2 = up2_2 + se2_2                              # torch.Size([1, 128, 64, 64])

        out2_1 = self.out2_1(torch.cat((ex_2, up2_1),dim=1))
        out2_2 = self.out2_2(torch.cat((ex_2,up2_1,feat2_2),dim=1))
        # feat_2_64 = self.bilinear2_2(out2_2)                        # torch.Size([1, 2, 64, 64])
        up1_1 = self.up1(ex_2)
        up1_2 = self.up1(out2_1)
        up1_3 = self.up1(out2_2)
        se1_3 = self.se1(out2_2)
        feat1_3 = up1_3 + se1_3                              # torch.Size([1, 64, 128, 128])

        out1_1 = self.out1_1(torch.cat((ex_1, up1_1),dim=1))
        out1_2 = self.out1_2(torch.cat((ex_1,out1_1,up1_2),dim=1))
        out1_3 = self.out1_3(torch.cat((ex_1,out1_1,out1_2,feat1_3),dim=1))  # torch.Size([1, 64, 128, 128])
        # feat_2_128 = self.bilinear1_3(out1_3)                                # torch.Size([1, 2, 128, 128])
        up0_1 = self.up0(ex_1)
        up0_2 = self.up0(out1_1)
        up0_3 = self.up0(out1_2)
        up0_4 = self.up0(out1_3)
        se0_4 = self.se0(out1_3)
        feat0_4 = up0_4 + se0_4

        out0_1 = self.out0_1(torch.cat((ex_0, up0_1), dim=1))
        out0_2 = self.out0_2(torch.cat((ex_0, out0_1, up0_2), dim=1))
        out0_3 = self.out0_3(torch.cat((ex_0, out0_1, out0_2, up0_3), dim=1))
        out0_4 = self.out0_4(torch.cat((ex_0, out0_1, out0_2, out0_3, feat0_4), dim=1))  # torch.Size([1, 32, 256, 256])
        # feat_2_256 = self.bilinear0_1(out0_4)                                # torch.Size([1, 2, 256, 256])

        out = self.decoder0(out0_4)   #torch.Size([1, 32, 512, 512])
        out = self.finalrelu1(out) # torch.Size([1, 32, 512, 512])
        out = self.finalconv3(out)  # torch.Size([1, 2, 512, 512])

        # return feat_2_32, feat_2_64, feat_2_128, feat_2_256, out
        return out


def unet():
    model = LinkNet34()
    return model



if __name__ == '__main__':
    print('#### Test Case ###')
    # import hiddenlayer as h1
    # from torchsummary import summary
    # model = LinkNet34().cpu()
    # summary(model, (3, 256, 256))

    x=torch.rand(1, 3, 512, 512)
    model=LinkNet34()
    print(model(x))

    # x = torch.rand(1, 512, 32, 32)
    # model = PSPUpsample(512, 128)
    # print(model(x))


    # 查看模型性能
    # from torchstat import stat
    # import torchvision.models as models
    # model = unet()
    # stat(model, (3, 512, 512))

    from thop import profile
    input = torch.randn(1, 3, 512, 512)
    model = LinkNet34().cpu()
    flops, params = profile(model, inputs=(input,))
    print(flops)
    print(params)