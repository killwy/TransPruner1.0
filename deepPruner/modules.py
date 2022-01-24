from torch import nn
import torch.nn.functional as F
import torch
import math
from timm.models.layers import trunc_normal_
def conv_bn_relu(inchanels,outchannels,ker,stride,pad,dilation):
    layer=nn.Sequential(
    nn.Conv2d(inchanels,outchannels,ker,stride,dilation if dilation>1 else pad,dilation,bias=False),
    nn.BatchNorm2d(outchannels),
    nn.ReLU(inplace=True)
    )
    return layer
def conv_bn(inchanels,outchannels,ker,stride,pad,dilation):
    return nn.Sequential(
        nn.Conv2d(inchanels,outchannels,ker,stride,dilation if dilation>1 else pad,dilation,bias=False),
        nn.BatchNorm2d(outchannels)
    )
def conv3d_bn_lrelu(inchannels,outchannels,ker,stride,pad):
    return nn.Sequential(nn.Conv3d(inchannels,outchannels,ker,(1,stride,stride),(pad,pad,pad),bias=False),
                         nn.BatchNorm3d(outchannels),
                         nn.LeakyReLU(0.1, inplace=True))

def convbn_transpose_3d(inplanes, outplanes, kernel_size, padding, output_padding, stride, bias):
    return nn.Sequential(nn.ConvTranspose3d(inplanes, outplanes, kernel_size, padding=padding,
                                            output_padding=output_padding, stride=stride, bias=bias),
                         nn.BatchNorm3d(outplanes))

def conv_relu(inchanels,outchannels,ker,stride,pad,dilation=1,bias=True):
    layer=nn.Sequential(
        nn.Conv2d(inchanels,outchannels,ker,stride,pad,dilation,bias=bias),
        nn.ReLU(inplace=True)
    )
    return layer

def conv_bn_lrelu(inchanels,outchannels,ker,stride,pad,dilation=1,bias=False):
    return nn.Sequential(
        nn.Conv2d(inchanels,outchannels,ker,stride,pad,dilation,bias=bias),
        nn.BatchNorm2d(outchannels),
        nn.LeakyReLU(0.1,inplace=True)
    )

# base class
class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    # to initialize the weight of networks
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

# only define the components of the network,instead of the structure.
# it helps to reuse these components
class BottleNeck(SubModule):
    def __init__(self,inplanes):
        super(BottleNeck, self).__init__()
        # encoder
        self.layer1=conv3d_bn_lrelu(inplanes,inplanes*2,ker=3,stride=2,pad=1)
        self.layer2=conv3d_bn_lrelu(inplanes*2,inplanes*2,ker=3,stride=1,pad=1)

        self.layer3=conv3d_bn_lrelu(inplanes*2,inplanes*4 ,ker=3,stride=2,pad=1)
        self.layer4=conv3d_bn_lrelu(inplanes*4,inplanes*4 ,ker=3,stride=1,pad=1)

        self.layer5=conv3d_bn_lrelu(inplanes*4,inplanes*8 ,ker=3,stride=2,pad=1)
        self.layer6=conv3d_bn_lrelu(inplanes*8,inplanes*8 ,ker=3,stride=1,pad=1)
        # bottleneck is here
        # decoder
        self.layer7=convbn_transpose_3d(inplanes*8,inplanes*4,kernel_size=3,padding=1,
                                        output_padding=(0,1,1),stride=(1,2,2),bias=False)
        self.layer8=convbn_transpose_3d(inplanes*4,inplanes*2,kernel_size=3,padding=1,
                                        output_padding=(0,1,1),stride=(1,2,2),bias=False)
        self.layer9=convbn_transpose_3d(inplanes*2,inplanes,kernel_size=3,padding=1,
                                        output_padding=(0,1,1),stride=(1,2,2),bias=False)
        # to eliminate the checkerboard effect, I guess
        self.last_conv3d_layer = nn.Sequential(
            conv3d_bn_lrelu(inplanes, inplanes * 2, 3, 1, 1),
            nn.Conv3d(inplanes * 2, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.softmax = nn.Softmax(dim=1)

        self.weight_init()

# it shares the same structure of MinDisparityPredictor
# but the weights of Nets are independent
class MaxDisparityPredictor(BottleNeck):
    def __init__(self,bottleNeck_inplanes=16):
        super(MaxDisparityPredictor, self).__init__(bottleNeck_inplanes)

    def forward(self,input,in_disp):
        # Reduce to 1/2
        out0=self.layer1(input)
        out1=self.layer2(out0)+out0
        # Reduce to 1/4
        out0=self.layer3(out1)
        out2=self.layer4(out0)+out0
        # Reduce to 1/8
        out0=self.layer5(out2)
        out3=self.layer6(out0)+out0
        # restore to 1/4,1/2,1
        out4=self.layer7(out3)+out2
        out5=self.layer8(out4)+out1
        out6=self.layer9(out5)

        out7=self.last_conv3d_layer(out6).squeeze(1)
        fetures=out7
        confidence=self.softmax(out7)
        disp_pred=torch.sum(confidence*(in_disp.squeeze(1)),dim=1).unsqueeze(1)
        # disp_pred=[B,1,H,W];fetures=[B,sample_num,H,W]
        return disp_pred,fetures

class MinDisparityPredictor(BottleNeck):
    def __init__(self,bottleNeck_inplanes=16):
        super(MinDisparityPredictor, self).__init__(bottleNeck_inplanes)
    def forward(self,input,in_disp):
        # Reduce to 1/2

        out0=self.layer1(input)
        out1=self.layer2(out0)+out0
        # Reduce to 1/4
        out0=self.layer3(out1)
        out2=self.layer4(out0)+out0
        # Reduce to 1/8
        out0=self.layer5(out2)
        out3=self.layer6(out0)+out0
        # restore to 1/4,1/2,1
        out4=self.layer7(out3)+out2
        out5=self.layer8(out4)+out1
        out6=self.layer9(out5)

        out7=self.last_conv3d_layer(out6).squeeze(1)
        fetures=out7
        confidence=self.softmax(out7)
        disp_pred=torch.sum(confidence*(in_disp.squeeze(1)),dim=1).unsqueeze(1)
        # disp_pred=[B,1,H,W];fetures=[B,sample_num,H,W]
        return disp_pred,fetures

# this resLayer is used in features extraction
class resLayer(nn.Module):
    def __init__(self,inchanels,outchannels,ker,stride,pad,dilation,downsample):
        super(resLayer,self).__init__()
        self.conv1=conv_bn_relu(inchanels,outchannels,ker,stride,dilation if dilation!=1 else pad,dilation)
            # nn.Conv2d(inchanels,outchannels,ker,stride,dilation if dilation!=1 else pad,dilation)
        self.conv2=conv_bn(outchannels,outchannels,ker,1,pad,dilation)
        self.downsample=downsample
    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        if self.downsample is not None:
            x=self.downsample(x)
        out+=x
        return out


class featureExtractor(nn.Module):
    def __init__(self):
        super(featureExtractor,self).__init__()
        # downsample->1/2
        self.conv0=conv_bn_relu(3,16,3,1,1,1)
        self.conv1=nn.Sequential(
            conv_bn_relu(16,32,3,2,1,1),  # 3
            conv_bn_relu(32,32,3,1,1,1),  # 7
            conv_bn_relu(32,32,3,1,1,1)  # 11
        )
        self.currentChannels = 32
        self.resLayer1=self.resLayers(32,3,1,1,1,3)  # 15~23
        self.resLayer2=self.resLayers(64,3,2,1,1,16)  # 27~87;downsample here，1/4
        self.resLayer3=self.resLayers(128,3,1,1,1,3)
        self.resLayer4=self.resLayers(128,3,1,1,2,3)  # 135

        self.branch1=nn.Sequential(nn.AvgPool2d((64,64),(64,64))
                                   ,conv_bn_relu(128,32,1,1,0,1))
        self.branch2=nn.Sequential(nn.AvgPool2d((32,32),(32,32)),
                                   conv_bn_relu(128,32,1,1,0,1))
        self.branch3=nn.Sequential(nn.AvgPool2d((16,16),(16,16)),
                                   conv_bn_relu(128,32,1,1,0,1))
        self.branch4=nn.Sequential(nn.AvgPool2d((8,8),(8,8)),
                                   conv_bn_relu(128,32,1,1,0,1))

        self.lastLayer=nn.Sequential(conv_bn_relu(320,128,3,1,1,1),nn.Conv2d(128,32,1,1,0,bias=False))

    def forward(self,input):
        features_0=self.conv0(input)
        features_=self.conv1(features_0)
        features0=self.resLayer1(features_)
        features1=self.resLayer2(features0)  # 87
        features=self.resLayer3(features1)
        features2=self.resLayer4(features)  # 感受野范围：135

        branchout1=self.branch1(features2)
        branchout1=F.upsample(branchout1,(features2.shape[2],features2.shape[3]),mode='bilinear')
        branchout2=self.branch2(features2)
        branchout2 = F.upsample(branchout2, (features2.shape[2], features2.shape[3]), mode='bilinear')
        branchout3=self.branch3(features2)
        branchout3 = F.upsample(branchout3, (features2.shape[2], features2.shape[3]), mode='bilinear')
        branchout4=self.branch4(features2)
        branchout4 = F.upsample(branchout4, (features2.shape[2], features2.shape[3]), mode='bilinear')
        lastinput=torch.cat((features1,features2,branchout1,branchout2,branchout3,branchout4),dim=1)
        features=self.lastLayer(lastinput)
        # 所以大概率不是感受野的问题，而是感受野强度的问题。CNN太过将每个像素点一视同仁了，导致可能对一些不知道
        # features=features0=[B,C=32,H,W]
        return features,features0,features_0  # 返回1/4大小的spp与1/2大小的low-level feature,与原始大小的

    def resLayers(self, outchannels, ker, stride, pad, dilation, block_num):
        layers = []
        downSampleLayer=None
        # 此处downSampleLayer是为了保证残差网络的x能与out相加
        # 当stride不为1时需要改变残差连接的图片大小，currentChannels!=outchannels时需要改变channel个数，都通过一个downsampler层实现
        if stride != 1 or self.currentChannels!=outchannels:
            downSampleLayer=nn.Sequential(
                nn.Conv2d(self.currentChannels,outchannels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(outchannels)
            )
        layers.append(resLayer(self.currentChannels,outchannels,ker,stride,pad,dilation,downSampleLayer))

        self.currentChannels=outchannels
        for i in range(1,block_num):
            layers.append(resLayer(self.currentChannels,self.currentChannels,ker,1,pad,dilation,None))
        return nn.Sequential(*layers)

if __name__=="__main__":
    fe=featureExtractor()
    print(fe)
    input=torch.rand((7,3,512,512))
    out,_=fe(input)
    print(out.shape)