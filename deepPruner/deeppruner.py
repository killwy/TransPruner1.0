from torch import nn
import torch.nn.functional as F
import torch
from deepPruner.modules import featureExtractor,conv_relu,conv_bn_relu,SubModule
from deepPruner.disparityRangePredict import rangePredict
from deepPruner.aggregate import aggregate
from deepPruner.refinement import refinement2,refinement1,refinement3
import deepPruner.patchmatch as pm
from deepPruner.config import config
from deepPruner.uniformSample import uniformSample
from PIL import Image
import numpy as np
import random
import logging


class deepPruner(SubModule):
    def __init__(self,training=True):
        super(deepPruner,self).__init__()
        # 只有实例才会继承device
        # 只有在__init__中实例化之后才能并入model的训练网络
        self.spp_scale_factor=4

        self.sample_num=config.patch_match_args.sample_count

        self.featureExtractor=featureExtractor()

        self.pm=pm.patchMatch()

        self.rangePredict=rangePredict()

        self.uniform_sample=uniformSample(config.max_disp,config.min_disp)

        inplanes=self.sample_num*2+64+1

        self.aggregate=aggregate(inplanes,hourglass_inplanes=16)

        # geometry branch
        self.low_level_feature_conv=nn.Sequential(
            conv_bn_relu(3,16,3,1,1,1),
            conv_bn_relu(16,16,3,1,1,1)
        )
        self.mid_level_feature_conv=nn.Sequential(
            conv_bn_relu(16,32,3,2,1,1),
            conv_bn_relu(32,32,3,1,1,1)
        )

        # refinement1
        self.refinement1=refinement1(32+self.sample_num+1+2)  # 47

        # refinement2
        self.refinement2=refinement2(32+4+1)

        # refinement3
        self.refinement3=refinement3(16+4+1)

        self.is_training=training
        self.device=None
        self.weight_init()

    def forward(self,left_input,right_input):
        self.device=left_input.get_device()
        # m1=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        # print("init memory %.5f MB,current memory %.5f MB"%(m1,m1))

        # feature extractor
        left_features=self.featureExtractor(left_input)
        right_features=self.featureExtractor(right_input)
        # m2=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        # print("feature extractor memory %.5f MB,current memory %.5f MB"%(m2-m1,m2))

        # patchMatch
        disp_pred = self.pm(left_features,right_features)     # disp_pred=[B,sample_num,H,W]
        # m3=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        # print("pm memory %.5f MB,current memory %.5f MB"%(m3-m2,m3))

        # range-predict
        mindisp,maxdisp,minfeatures,maxfeatures=self.rangePredict(left_features,right_features,disp_pred)
        # m4=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        # print("range predictor memory %.5f MB,current memory %.5f MB"%(m4-m3,m4))

        # patchMatch,kiiti换成uniform采样  [B,SAMPLES+2,H,W]
        disp_pred = self.pm(left_features,right_features,mindisp,maxdisp)
        # disp_pred = self.uniform_sample(mindisp,maxdisp,self.sample_num)
        # m5=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        # print("pm2 memory %.5f MB,current memory %.5f MB"%(m5-m4,m5))

        # aggregation
        # inplanes=left_features.shape[1]+right_features.shape[1]+1+maxfeatures.shape[1]+minfeatures.shape[1]
        disp_pred_agg,aggregated_features=self.aggregate(left_features,right_features,disp_pred,maxfeatures,minfeatures)
        # m6=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        # print("aggregation memory %.5f MB,current memory %.5f MB"%(m6-m5,m6))

        # geometry branch: we want to extract the geometry-related features
        left_low_level=self.low_level_feature_conv(left_input)
        left_mid_level=self.mid_level_feature_conv(left_low_level)


        # refinement1
        refinement1_net_input=torch.cat((left_features,aggregated_features,disp_pred_agg),dim=1)
        global_feature=self.refinement1(refinement1_net_input)
        # m7=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        # print("refinement1 memory %.5f MB,current memory %.5f MB"%(m7-m6,m7))

        # up-sampling
        global_feature=F.interpolate(global_feature,scale_factor=(2,2),mode='bilinear')
        disp_pred_agg=F.interpolate(disp_pred_agg*2,scale_factor=(2,2),mode='bilinear')

        # refinement2
        refinement2_net_input=torch.cat((left_mid_level,global_feature,disp_pred_agg),dim=1)
        disp_pred_refine2 = self.refinement2(refinement2_net_input, disp_pred_agg)
        # m8=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        # print("refinement2 memory %.5f MB,current memory %.5f MB"%(m8-m7,m8))

        # up-sampling
        disp_pred_refine2 = F.interpolate(disp_pred_refine2 * 2, scale_factor=(2, 2), mode='bilinear')
        global_feature=F.interpolate(global_feature,scale_factor=(2,2),mode='bilinear')

        # refinement3
        refinement3_net_input=torch.cat((left_low_level,global_feature,disp_pred_refine2),dim=1)
        disp_pred_refine3=self.refinement3(refinement3_net_input,disp_pred_refine2)
        # m9=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        # print("refinement3 memory %.5f MB,current memory %.5f MB"%(m9-m8,m9))

        #upsample
        disp_pred_agg=F.interpolate(disp_pred_agg*2,scale_factor=(2,2),mode='bilinear')
        mindisp=F.interpolate(mindisp*4,scale_factor=(4,4),mode='bilinear')
        maxdisp=F.interpolate(maxdisp*4,scale_factor=(4,4),mode='bilinear')
        # m10=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        # print("total memory %.5f MB"%(m10))

        return disp_pred_refine3,disp_pred_refine2,disp_pred_agg,mindisp,maxdisp

if __name__=="__main__":
    # load image
    left=Image.open('/home/jiaxi/workspace/KITTI/training/colored_0/000000_10.png')
    right=Image.open('/home/jiaxi/workspace/KITTI/training/colored_1/000000_10.png')
    # pre-process
    # left=np.array(left)
    # print(left.shape)
    right=np.array(right)
    left=torch.from_numpy(left).to('cuda:1').permute(2,0,1).unsqueeze_(0).float()
    right=torch.from_numpy(right).to('cuda:1').permute(2,0,1).unsqueeze_(0).float()
    #
    x=deepPruner()
    disp_pred=x.forward(left,right)
