

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch.nn.init import kaiming_normal_, constant_
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from collections import OrderedDict

import tools
from tools.ai.demo_utils import crf_inference
from tools.ai.torch_utils import resize_for_tensors
from tools.general.Q_util import *
from tools.ai.torch_utils import make_cam
from core.models.model_util import conv


from .deeplab_utils import ASPP, Decoder, ASPP_V2
from .arch_resnet import resnet
from .arch_resnest import resnest
from .abc_modules import ABC_Model



#######################################################################
# Normalization
#######################################################################
class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

def group_norm(features):
    return nn.GroupNorm(4, features)
#######################################################################

def conv_bn(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

def conv_dilation(batchNorm, in_planes, out_planes, kernel_size=3, stride=1,dilation=16):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation, bias=False,dilation=dilation,padding_mode='circular'),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True,dilation=dilation,padding_mode='circular'),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

def get_noliner(features):
            b, c, h, w = features.shape
            if(c==9):
                feat_pd = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
            elif(c==25):
                feat_pd = F.pad(features, (2, 2, 2, 2), mode='constant', value=0)

            diff_map_list=[]
            nn=int(math.sqrt(c))
            for i in range(nn):
                for j in range(nn):
                        diff_map_list.append(feat_pd[:,i*nn+j,i:i+h,j:j+w])
            ret = torch.stack(diff_map_list,dim=1)
            return ret


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1)
        )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1)
    )


class Backbone(nn.Module, ABC_Model):
    def __init__(self, model_name, num_classes=20, mode='fix', segmentation=False):
        super().__init__()

        self.mode = mode

        if self.mode == 'fix':
            self.norm_fn = FixedBatchNorm
        else:
            self.norm_fn = nn.BatchNorm2d

        if 'resnet' in model_name:

            if('moco' in model_name ):
                state_dict = torch.load("/media/ders/XS/SPCAM/models_ckpt/moco_r50_v2-e3b0c442.pth")['state_dict']
                model_name = model_name[:-5]

            elif('detco' in model_name ):
                state_dict = torch.load("/media/ders/XS/SPCAM/models_ckpt/detco_200ep.pth")
                model_name = model_name[:-6]
            elif('dino' in model_name ):
                state_dict = torch.load("/media/ders/XS/SPCAM/models_ckpt/dino_resnet50_pretrain.pth")
                model_name = model_name[:-5]

            elif('resnet101' in model_name):
                print("##################################################￥333333")
                state_dict = torch.load("/media/ders/sdb1/hjw/SPCAM_GCMS/resnet101-5d3b4d8f.pth")
            elif ('resnet50' in model_name):
                state_dict = torch.load("/media/ders/sdb1/hjw/SPCAM_GCMS/resnet50-19c8e357.pth")
            else:
                print('resnet101' in model_name)

                state_dict = model_zoo.load_url(resnet.urls_dic[model_name])
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
            self.model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[model_name], strides=(2, 2, 2, 1), batch_norm_fn=self.norm_fn)

            self.model.load_state_dict(state_dict)
        else:
            if segmentation:
                dilation, dilated = 4, True
            else:
                dilation, dilated = 2, False

            self.model = eval("resnest." + model_name)(pretrained=True, dilated=dilated, dilation=dilation, norm_layer=self.norm_fn)

            del self.model.avgpool
            del self.model.fc

        self.stage1 = nn.Sequential(self.model.conv1,
                                    self.model.bn1,
                                    self.model.relu,
                                    self.model.maxpool)
        self.stage2 = nn.Sequential(self.model.layer1)
        self.stage3 = nn.Sequential(self.model.layer2)
        self.stage4 = nn.Sequential(self.model.layer3)
        self.stage5 = nn.Sequential(self.model.layer4)


class CLSNet(Backbone):
    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix', segmentation=False)

        self.num_classes =num_classes
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)

    def forward(self, inputs, pcm=0, th=0.6):
        x = self.stage1(inputs)
        x = self.stage2(x)
        x = self.stage3(x).detach()
        x4 = self.stage4(x)
        x5 = self.stage5(x4)
       
        logits = self.classifier(x5)  
        
        logits_min =(F.adaptive_avg_pool2d(self.classifier(x5), 1))

        return logits, logits_min, x4
