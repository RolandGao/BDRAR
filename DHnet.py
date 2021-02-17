import torch
import torch.nn as nn
import torchvision
from IntermediateModule import IntermediateModule
from torch.nn import functional as F
#import tf_slim as slim
channel=64
vgg19=torchvision.models.vgg19()
dic={'3':'conv1_2',
     '8':'conv2_2',
     '13':'conv3_2',
     '22':'conv4_2',
     '31':'conv5_2'
     }
vgg19_features=IntermediateModule(vgg19.features,dic)
print(vgg19_features)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv2d(in_planes, out_planes,kernel_size,rate=1):
    conv=None
    if kernel_size==3:
        conv=conv3x3(in_planes,out_planes,dilation=rate)
    elif kernel_size==1:
        conv=conv1x1(in_planes,out_planes)
    else:
        raise NotImplementedError()
    activation_fn=nn.LeakyReLU(0.2)
    norm=nn.InstanceNorm2d(out_planes)
    return nn.Sequential(conv,norm,activation_fn)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):#originally 16
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def agg(in_planes,out_planes):
    return nn.Sequential(
        SELayer(in_planes),
        conv2d(in_planes,out_planes,3)
    )


class DHnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=conv2d(500,channel,1)
        self.conv0=conv2d(channel,channel,1)
        self.conv1=conv2d(channel,channel,3)
        self.agg0=agg(channel*2,channel)
        self.agg1 = agg(channel * 2, channel)
        self.conv2 = conv2d(channel, channel, 3,rate=2)
        self.conv3 = conv2d(channel, channel, 3,rate=4)
        self.agg2 = agg(channel * 3, channel)
        self.agg3 = agg(channel * 3, channel)
        self.conv4 = conv2d(channel, channel, 3, rate=8)
        self.conv5 = conv2d(channel, channel, 3, rate=16)
        self.agg4 = agg(channel * 3, channel)
        self.agg5 = agg(channel * 3, channel)
        self.conv6 = conv2d(channel, channel, 3, rate=32)
        self.conv7 = conv2d(channel, channel, 3, rate=64)
        self.agg6 = agg(channel * 4, channel)
        self.agg7 = agg(channel * 4, channel)
        self.img=conv1x1(channel,3)
        self.mask=conv1x1(channel,1)

    def forward(self,x):
        shape=x.shape[-2:]
        x=vgg19_features(x)
        features=[]
        for i in range(1,6):
            f=x[f"conv{i}_2"]
            f=F.interpolate(f, size=shape, mode='bilinear',align_corners=False)
            features.append(f)
        x=torch.cat(features,dim=1)
        x=self.project(x)
        x=self.conv0(x)
        x2=self.conv1(x)
        x=torch.cat([x,x2],dim=3)
        aggi=self.agg0(x)
        aggm=self.agg1(x)
        x=aggi*torch.sigmoid(aggm)
        x=self.conv2(x)
        x2=self.conv3(x)
        aggi=self.agg2(torch.cat([aggi,x,x2],dim=3))
        aggm=self.agg3(torch.cat([aggm,x,x2],dim=3))
        x = aggi * torch.sigmoid(aggm)
        x=self.conv4(x)
        x2=self.conv5(x)
        aggi_2 = self.agg4(torch.cat([aggi, x, x2], dim=3))
        aggm_2 = self.agg5(torch.cat([aggm, x, x2], dim=3))
        x = aggi_2 * torch.sigmoid(aggm_2)
        x=self.conv6(x)
        x2=self.conv7(x)
        aggi=self.agg6(torch.cat([aggi,aggi_2, x, x2], dim=3))
        aggm = self.agg7(torch.cat([aggm, aggm_2, x, x2], dim=3))
        #Spp
        aggi = aggi * torch.sigmoid(aggm)
        img=self.img(aggi)
        mask=self.mask(aggm)
        return img,mask
