import torch
from torch import nn
from deepPruner.modules import conv_bn_lrelu
from deepPruner.transformer.selfAttention import *
from deepPruner.modules import SubModule
from deepPruner.transformer.swinTransformer import swinTransformer
from deepPruner.config import config
from einops import rearrange
class refinement2(SubModule):
    def __init__(self,inplanes):
        super(refinement2, self).__init__()
        self.swinT=swinTransformer(1,
                                   [2],
                                   config.patchsize,
                                   inplanes,
                                   32,
                                   head_dims=16,
                                   window_size=8)
        self.classif1 = nn.Conv2d(32//(config.patchsize**2), 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.weight_init()

        # --------swinTransformer 1.1--------

    def patches2features(self,patches,h_scale,w_scale):
        '''
        :param patches:shape=[N,C,P,P,patch_num]
        :return: features: shape=[N,C,H,W]
        '''
        return rearrange(patches,'b c p1 p2 (h w)-> b c (h p1) (w p2)',h=h_scale,w=w_scale)

    def forward(self, input, disparity):

        # output0 = self.conv1(input)
        # output0 = self.classif1(output0)
        # output = self.relu(output0 + disparity)

        # N,C,H,W=input.shape
        # out1=self.encoder(input)
        # N,D,patch_num=out1.shape
        # out2=self.mlp(out1)
        # out2=out2.view(N,-1,config.patchsize,config.patchsize,patch_num)
        # out3=self.patches2features(out2,H//config.patchsize,W//config.patchsize)
        # output=self.relu(out3+disparity)

        N,C,H,W=input.shape
        out1=self.swinT(input,H,W)
        out2=self.classif1(out1)
        output=self.relu(out2+disparity)
        return output

class refinement1(SubModule):
    def __init__(self,inplanes):
        super(refinement1, self).__init__()
        self.inplanes=inplanes

        self.swinT=swinTransformer(1,
                                   [4],  # transformer_layer_num
                                   config.patchsize,
                                   inplanes,
                                   config.trans_vec_dim,
                                   head_dims=32,
                                   window_size=8)
        self.classif1 = nn.Conv2d(config.trans_vec_dim//(config.patchsize**2), 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.weight_init()

    def forward(self, input):
        N,C,H,W=input.shape
        out1=self.swinT(input,H,W)
        # out2=self.classif1(out1)
        # output=self.relu(out2+disparity)
        return out1

class refinement3(SubModule):
    def __init__(self,inplanes):
        super(refinement3, self).__init__()
        self.inplanes=inplanes
        self.swinT=swinTransformer(1,
                                   [1],  # transformer_layer_num
                                   patch_size=2,
                                   in_channels=inplanes,
                                   vec_dims=16,
                                   head_dims=4,
                                   window_size=4)
        self.classif1 = nn.Conv2d(16//(2**2), 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.weight_init()

    def forward(self, input, disparity):
        N,C,H,W=input.shape
        out1=self.swinT(input,H,W)
        out2=self.classif1(out1)
        output=self.relu(out2+disparity)
        return output