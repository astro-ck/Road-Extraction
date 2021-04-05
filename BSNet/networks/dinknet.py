"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""

import torch.nn as nn

from torchvision import models
import torch.nn.functional as F

from functools import partial

from networks.base_model import *
import numpy as np

nonlinearity = partial(F.relu,inplace=True)

class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

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

class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class=1, pad=0):
        super(DUpsampling, self).__init__()
        ## W matrix convert x to full resolution prediction
        self.conv_w = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=False)
        ## P matrix
        self.conv_p = nn.Conv2d(num_class * scale * scale, inplanes, kernel_size=1, padding=pad, bias=False)

        self.scale = scale

    def forward(self, x):
        x = self.conv_w(x)
        N, C, H, W = x.size()
        # N, W, H, C
        x_permuted = x.permute(0, 3, 2, 1)
        # N, W, H*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale, int(C / (self.scale))))

        # N, H*scale, W, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, H*scale, W*scale, C/(scale**2)
        # 有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。
        # view 类似于reshape，但是并没有对原始数据进行处理，而是返回一个新shape
        x_permuted = x_permuted.contiguous().view(
            (N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))
        # N, C/(scale**2), H*scale, W*scale
        # permute是维度换位，是更灵活的transpose，可以灵活的对原数据的维度进行调换，而数据本身不变（transpose也是）。
        x = x_permuted.permute(0, 3, 1, 2)

        return x


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()

        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        # self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        # dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out #+ dilate4_out + dilate5_out
        return out


class Decoder(nn.Module):
    def __init__(self, num_class,input_channel):
        super(Decoder, self).__init__()
        # low feature input channels: 256+128=384
        # low feature input channels(ratio 16) 64+128=192 + 256=448, output channel should be 16*16=256
        self.conv1 = nn.Conv2d(input_channel, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU() 
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        # self.dropout2 = nn.Dropout(0.5)
        # self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.dropout3 = nn.Dropout(0.1)
        # self.conv4 = nn.Conv2d(256, 256, kernel_size=1)

        self.dupsample = DUpsampling(256, 16, num_class)
        self._init_weight()
        self.T = torch.nn.Parameter(torch.Tensor([1.00]))
        self.dblock=Dblock(256)


    def forward(self, x, low_level_feature):
        # x:512  low_level_feature=384

        # x: 512  low_level: 320
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)
        # N*1024*8*8
        x_4_cat = torch.cat((x, low_level_feature), dim=1) # channels

        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        d = self.dblock.forward(x_4_cat)
        # x_4_cat = self.dropout2(x_4_cat)
        # x_4_cat = self.conv3(x_4_cat)
        # x_4_cat = self.bn3(x_4_cat)
        # x_4_cat = self.relu(x_4_cat)
        # x_4_cat = self.dropout3(x_4_cat)
        # x_4_cat = self.conv4(x_4_cat)

        out = self.dupsample(d)
        out = torch.sigmoid(out)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DinkNet34(nn.Module):
    def __init__(self):
        super(DinkNet34, self).__init__()

        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4


    def forward(self, x):
        # Encoder
        # x N*3*256*256
        x = self.firstconv(x) # N*64*128*128
        x = self.firstbn(x)
        x = self.firstrelu(x)
        b1_3 = x
        x = self.firstmaxpool(x)  # N*64*64*64
        e1 = self.encoder1(x)  # N*64*64*64
        e2 = self.encoder2(e1)  # N*128*32*32
        e3 = self.encoder3(e2)  # N*256*16*16
        # e4 = self.encoder4(e3)  # N*512*16*16

        # downsample low dim
        # N*128*8*8
        # down_e2=F.upsample(e2,[e2.size()[2]//4,e2.size()[3]//4],mode="bilinear",align_corners=True)
        # N*64*16*16
        # down_e1 = F.upsample(e1, [e1.size()[2] // 4, e1.size()[3] // 4], mode="bilinear", align_corners=True)
        # b1_3 combine with e3
        down_e2 = F.interpolate(e2, [e3.size()[2], e3.size()[3]], mode="bilinear", align_corners=True)
        down_e1=F.interpolate(e1, [e3.size()[2], e3.size()[3]], mode="bilinear", align_corners=True)
        # N*256*8*8
        # down_e3=F.upsample(e3,[e3.size()[2]//2,e3.size()[3]//2],mode="bilinear",align_corners=True)
        # N * 128*16*16
        # down_e2 = F.upsample(e2, [e3.size()[2], e3.size()[3]], mode="bilinear", align_corners=True)

        # N*384*8*8
        # x_low = torch.cat((down_e2, down_e3), dim=1)
        # N*192*16*16
        # x_low=torch.cat((down_e1,down_e2),dim=1)
        # N*320*16*16
        x_low=torch.cat((down_e2,down_e1),dim=1)
        return e3,x_low


class DUNet(nn.Module):
    def __init__(self, num_class=1):
        super(DUNet, self).__init__()
        self.encoder = DinkNet34()
        # self.encoder = DinkNet50()

        self.decoder = Decoder(1,192)
    def forward(self, x):
        x, x_low = self.encoder.forward(x)
        out = self.decoder.forward(x, x_low)
        return out

pretrained_mean = torch.tensor([0.485, 0.456, 0.406], requires_grad = False).view((1, 3, 1, 1))
pretrained_std = torch.tensor([0.229, 0.224, 0.225], requires_grad = False).view((1, 3, 1, 1))

def choose_vgg(name):
    f = None
    if name == 'vgg11':
        f = models.vgg11(pretrained = True)
    elif name == 'vgg11_bn':
        f = models.vgg11_bn(pretrained = True)
    elif name == 'vgg13':
        f = models.vgg13(pretrained = True)
    elif name == 'vgg13_bn':
        f = models.vgg13_bn(pretrained = True)
    elif name == 'vgg16':
        f = models.vgg16(pretrained = True)
    elif name == 'vgg16_bn':
        f = models.vgg16_bn(pretrained = True)
    elif name == 'vgg19':
        f = models.vgg19(pretrained = True)
    elif name == 'vgg19_bn':
        f = models.vgg19_bn(pretrained = True)

    for params in f.parameters():
        params.requires_grad = False
    return f


class VGGNet(nn.Module):
    def __init__(self, name, layers, cuda=True): # layers???
        super(VGGNet, self).__init__()
        self.vgg = choose_vgg(name)
        self.layers = layers

        features = list(self.vgg.features)[:max(layers) + 1]
        self.features = nn.ModuleList(features).eval()

        self.mean = pretrained_mean.cuda() if cuda else pretrained_mean
        self.std = pretrained_std.cuda() if cuda else pretrained_std

    def forward(self, x):
        x = (x - self.mean) / self.std
        results = []
        for ii, model in enumerate(self.features): # feature???
            x = model(x)
            if ii in self.layers:
                results.append(x.view(x.shape[0], -1)) # Multi-dimensional tensor into one dimension
        return results


class TopologyNet(nn.Module):
    def __init__(self, dlinknet, vggnet):
        super(TopologyNet, self).__init__()
        self.encoder = dlinknet
        # self.encoder = DinkNet50()

        self.decoder = Decoder(1,192)


        self.vggnet = vggnet

    def forward(self, x):
        results = []
        x, x_low = self.encoder.forward(x)
        preds = self.decoder.forward(x, x_low)
        preds_vgg = self.vggnet(torch.cat((preds, preds, preds), dim=1))
        results.append([preds, preds_vgg])
        return results



if __name__=="__main__":
    device = torch.device("cuda:0")
    input = torch.rand(1, 3, 256, 256)
    input = input.to(device)
    net = TopologyNet()
    net.to(device)
    output = net(input)
    print(output.size())
#....
