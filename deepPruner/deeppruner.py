from torch import nn
import torch.nn.functional as F
import torch
from deepPruner.modules import featureExtractor,conv_relu,SubModule
from deepPruner.disparityRangePredict import rangePredict
from deepPruner.aggregate import aggregate
from deepPruner.refinement import refinement2,refinement1,refinement3
import deepPruner.patchmatch as pm
from deepPruner.config import config
from deepPruner.uniformSample import uniformSample
from PIL import Image
import numpy
import logging
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
        # refinement1
        self.refinement1=refinement1(32+self.sample_num+1+2) # 47
        self.disp_pred_conv=conv_relu(1, 1, 5, 1, 2)
        self.refinement1_features_conv=conv_relu(4, 4, 5, 1, 2)
        self.mid_level_feature_conv=conv_relu(32,16,3,1,1)
        # refinement2
        self.refinement2=refinement2(16+4+1)
        self.refined2_disp_pred_conv=conv_relu(1, 1, 5, 1, 2)
        self.refinement2_features_conv=conv_relu(2, 2, 5, 1, 2)
        self.low_level_feature_conv=conv_relu(16,8,3,1,1)
        # refinement3
        self.refinement3=refinement3(8+2+1)
        self.disp_final_pred_conv=conv_relu(1,1,3,1,1)
        self.is_training=training
        self.device=None
        self.weight_init()

    def forward(self,left_input,right_input):
        self.device=left_input.get_device()
        m1=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        logging.info("init memory %.5f MB,current memory %.5f MB"%(m1,m1))

        # feature extractor
        left_features,left_mid_level,left_low_level=self.featureExtractor(left_input)
        right_features,_,_=self.featureExtractor(right_input)
        m2=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        logging.info("feature extractor memory %.5f MB,current memory %.5f MB"%(m2-m1,m2))

        # patchMatch
        # x=self.pm(left_features,right_features).to(self.device)
        disp_pred = self.pm1(left_features,right_features)     # disp_pred=[B,sample_num,H,W]
        m3=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        logging.info("pm1 memory %.5f MB,current memory %.5f MB"%(m3-m2,m3))

        # range-predict
        mindisp,maxdisp,minfeatures,maxfeatures=self.rangePredict(left_features,right_features,disp_pred)
        m4=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        logging.info("range predictor memory %.5f MB,current memory %.5f MB"%(m4-m3,m4))

        # patchMatch,kiiti换成uniform采样  [B,SAMPLES+2,H,W]
        disp_pred = self.pm1(left_features,right_features,mindisp,maxdisp)
        m5=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        logging.info("pm2 memory %.5f MB,current memory %.5f MB"%(m5-m4,m5))
        # disp_pred = self.uniform_sample(mindisp,maxdisp,self.sample_num)

        # aggregation
        # inplanes=left_features.shape[1]+right_features.shape[1]+1+maxfeatures.shape[1]+minfeatures.shape[1]
        disp_pred_agg,aggregated_features=self.aggregate(left_features,right_features,disp_pred,maxfeatures,minfeatures)
        m6=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        logging.info("aggregation memory %.5f MB,current memory %.5f MB"%(m6-m5,m6))

        # refinement1
        refinement1_net_input=torch.cat((left_features,aggregated_features,disp_pred_agg),dim=1)
        refinement1_feature,disp_pred_refine1=self.refinement1(refinement1_net_input,disp_pred_agg)
        m7=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        logging.info("refinement1 memory %.5f MB,current memory %.5f MB"%(m7-m6,m7))
        # up-sampling
        disp_pred_refine1=F.interpolate(disp_pred_refine1*2,size=(left_mid_level.shape[2],left_mid_level.shape[3]),mode='bilinear')
        refinement1_feature=F.interpolate(refinement1_feature,size=(left_mid_level.shape[2],left_mid_level.shape[3]),mode='bilinear')
        disp_pred_refine1 = self.disp_pred_conv(disp_pred_refine1)
        refinement1_feature = self.refinement1_features_conv(refinement1_feature)
        left_mid_level=self.mid_level_feature_conv(left_mid_level)

        # refinement2
        refinement2_net_input=torch.cat((left_mid_level,refinement1_feature,disp_pred_refine1),dim=1)
        refinement2_feature,disp_pred_refine2 = self.refinement2(refinement2_net_input, disp_pred_refine1)
        m8=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        logging.info("refinement2 memory %.5f MB,current memory %.5f MB"%(m8-m7,m8))

        # up-sampling
        disp_pred_refine2 = F.interpolate(disp_pred_refine2 * 2, scale_factor=(2, 2), mode='bilinear')
        refinement2_feature=F.interpolate(refinement2_feature, scale_factor=(2, 2), mode='bilinear')
        disp_pred_refine2=self.refined2_disp_pred_conv(disp_pred_refine2)
        refinement2_feature=self.refinement2_features_conv(refinement2_feature)
        left_low_level=self.low_level_feature_conv(left_low_level)

        # refinement3
        refinement3_net_input=torch.cat((left_low_level,refinement2_feature,disp_pred_refine2),dim=1)
        disp_pred_refine3=self.refinement3(refinement3_net_input,disp_pred_refine2)
        disp_pred_refine3=self.disp_final_pred_conv(disp_pred_refine3)
        m9=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        logging.info("refinement3 memory %.5f MB,current memory %.5f MB"%(m9-m8,m9))
        #upsample
        disp_pred_refine1=F.interpolate(disp_pred_refine1*2,scale_factor=(2,2),mode='bilinear')
        disp_pred_agg=F.interpolate(disp_pred_agg*4,scale_factor=(4,4),mode='bilinear')
        mindisp=F.interpolate(mindisp*4,scale_factor=(4,4),mode='bilinear')
        maxdisp=F.interpolate(maxdisp*4,scale_factor=(4,4),mode='bilinear')
        m10=torch.cuda.memory_allocated(device=self.device)/(1024*1024)
        logging.info("total memory %.5f MB"%(m10))
        return disp_pred_refine3,disp_pred_refine2,disp_pred_refine1,\
               disp_pred_agg,mindisp,maxdisp

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
