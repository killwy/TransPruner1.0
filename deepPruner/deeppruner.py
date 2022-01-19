from torch import nn
import torch.nn.functional as F
import torch
from deepPruner.modules import featureExtractor,conv_relu,SubModule
from deepPruner.disparityRangePredict import rangePredict
from deepPruner.aggregate import aggregate
from deepPruner.refinement import refinement
import deepPruner.patchmatch as pm
from deepPruner.config import config
from deepPruner.uniformSample import uniformSample
from PIL import Image
import numpy
class deepPruner(SubModule):
    def __init__(self,training=True):
        super(deepPruner,self).__init__()
        self.spp_scale_factor=4
        self.sample_num=config.patch_match_args.sample_count
        # 只有实例才会继承device
        # 只有在__init__中实例化之后才能并入model的训练网络
        self.featureExtractor=featureExtractor()
        self.pm1=pm.patchMatch()
        self.rangePredict=rangePredict()
        # self.pm2=pm.patchMatch()
        self.uniform_sample=uniformSample(config.max_disp,config.min_disp)
        inplanes=self.sample_num*2+64+1
        self.aggregate=aggregate(inplanes,hourglass_inplanes=16)
        self.aggregated_features_conv=conv_relu(self.sample_num+2, self.sample_num+2, 5, 1, 2)
        self.refinement=refinement(32+self.sample_num+1+2)
        self.disp_pred_conv=conv_relu(1, 1, 5, 1, 2)
        self.device=None
        # self.disp_final_pred_conv=conv_relu(1,1,3,1,1)
        self.is_training=training
        self.weight_init()

    def forward(self,left_input,right_input):

        # feature extractor
        self.device=left_input.get_device()
        left_features,left_low_level=self.featureExtractor(left_input)
        right_features,right_low_level=self.featureExtractor(right_input)

        # patchMatch
        # x=self.pm(left_features,right_features).to(self.device)
        disp_pred = self.pm1(left_features,right_features)     # disp_pred=[B,sample_num,H,W]
        # range-predict
        mindisp,maxdisp,minfeatures,maxfeatures=self.rangePredict(left_features,right_features,disp_pred)

        # patchMatch,kiiti换成uniform采样
        # disp_pred = self.pm1(left_features,right_features,mindisp,maxdisp)
        # [B,SAMPLES+2,H,W]
        disp_pred = self.uniform_sample(mindisp,maxdisp,self.sample_num)

        # aggregation
        # inplanes=left_features.shape[1]+right_features.shape[1]+1+maxfeatures.shape[1]+minfeatures.shape[1]
        disp_pred,aggregated_features=self.aggregate(left_features,right_features,disp_pred,maxfeatures,minfeatures)

        # refinement
        # up-sampling
        disp_pred=F.interpolate(disp_pred*2,size=(left_low_level.shape[2],left_low_level.shape[3]),mode='bilinear')
        aggregated_features=F.interpolate(aggregated_features,size=(left_low_level.shape[2],left_low_level.shape[3]),mode='bilinear')
        disp_pred = self.disp_pred_conv(disp_pred)
        aggregated_features = self.aggregated_features_conv(aggregated_features)
        # refinement
        refinement_net_input=torch.cat((left_low_level,aggregated_features,disp_pred),dim=1)
        refined_disparity = self.refinement(refinement_net_input, disp_pred)
        # up-sampling
        refined_disparity = F.interpolate(refined_disparity * 2, scale_factor=(2, 2), mode='bilinear')
        disp_pred=F.interpolate(disp_pred*2, scale_factor=(2, 2), mode='bilinear')
        # 自己加的
        # disp_pred=self.disp_final_pred_conv(disp_pred)
        mindisp=F.interpolate(mindisp*self.spp_scale_factor,size=(refined_disparity.shape[2],refined_disparity.shape[3]), mode='bilinear')
        maxdisp=F.interpolate(maxdisp*self.spp_scale_factor,size=(refined_disparity.shape[2],refined_disparity.shape[3]), mode='bilinear')
        return refined_disparity,disp_pred,mindisp,maxdisp

if __name__=="__main__":
    # load image
    left=Image.open('/home/jiaxi/workspace/KITTI/training/colored_0/000000_10.png')
    right=Image.open('/home/jiaxi/workspace/KITTI/training/colored_1/000000_10.png')
    # pre-process
    left=numpy.array(left)
    print(left.shape)
    right=numpy.array(right)
    left=torch.from_numpy(left).to('cuda:1').permute(2,0,1).unsqueeze_(0).float()
    right=torch.from_numpy(right).to('cuda:1').permute(2,0,1).unsqueeze_(0).float()
    #
    x=deepPruner()
    disp_pred=x.forward(left,right)
