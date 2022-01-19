from torch import nn
import torch
import torch.nn.functional as F
from deepPruner.modules import MinDisparityPredictor,MaxDisparityPredictor,conv_relu,conv_bn_lrelu,conv3d_bn_lrelu
from deepPruner.config import config
class rangePredict(nn.Module):
    def __init__(self):
        super(rangePredict,self).__init__()
        self.leftfeatures=None
        self.device=None
        self.W=None
        self.H=None
        self.C=None
        self.B=None
        self.rightfeatures=None
        self.samples=None
        self.sample_num=config.patch_match_args.sample_count
        self.inplanes=65
        self.hourglass_inplanes=16
        self.dres0 = nn.Sequential(conv3d_bn_lrelu(self.inplanes, 64, 3, 1, 1),
                                   conv3d_bn_lrelu(64, 32, 3, 1, 1))
        self.dres1 = nn.Sequential(conv3d_bn_lrelu(32, 32, 3, 1, 1),
                                   conv3d_bn_lrelu(32, self.hourglass_inplanes, 3, 1, 1))
        self.min_disp_conv=conv_relu(1,1,5,1,2)
        self.max_disp_conv=conv_relu(1,1,5,1,2)
        self.max_features_conv=conv_bn_lrelu(self.sample_num,self.sample_num,5,1,2,dilation=1,bias=True)
        self.min_feature_conv=conv_bn_lrelu(self.sample_num,self.sample_num,5,1,2,dilation=1,bias=True)
        self.MinDisparityPredictor=MinDisparityPredictor(self.hourglass_inplanes)
        self.MaxDisparityPredictor=MaxDisparityPredictor(self.hourglass_inplanes)

    def warp(self):
        # left_x_coodinate = torch.arange(self.W).unsqueeze_(0).expand(self.H, -1).unsqueeze_(0).unsqueeze_(0).expand(
        #     self.B, 1, self.H, self.W).to(self.device)
        # right_x_coodinate = torch.clamp(left_x_coodinate.float() - self.samples, min=0, max=self.W - 1)
        # right_y_coodinate = torch.arange(self.H).unsqueeze_(1).expand(-1, self.W).unsqueeze_(0).unsqueeze_(0).expand(
        #     self.B, right_x_coodinate.shape[1], self.H, self.W).to(self.device)
        # right_x_coodinate -= right_x_coodinate.shape[3] / 2
        # right_x_coodinate /= (right_x_coodinate.shape[3] / 2)
        # right_y_coodinate = right_y_coodinate - right_y_coodinate.shape[2] / 2
        # right_y_coodinate /= (right_y_coodinate.shape[2] / 2)
        # right_xy_coodinate = torch.cat((right_x_coodinate.unsqueeze_(4),right_y_coodinate.unsqueeze_(4).float()),
        #                                dim=4)
        # # right_xy_coodinate.shape=[N*5*sample_num,H,W,2]
        # right_xy_coodinate = right_xy_coodinate.view(right_xy_coodinate.shape[0] * right_xy_coodinate.shape[1],
        #                                              right_xy_coodinate.shape[2], right_xy_coodinate.shape[3],
        #                                              right_xy_coodinate.shape[4])
        # # disp_candidates.shape=[N,sample_num,H,W];right_features=[N,C,H,W]
        # wraped_right_featutes = F.grid_sample(
        #     self.rightfeatures.repeat(self.samples.shape[1] ,1,1,1).float(),
        #     right_xy_coodinate,align_corners=True)
        left_x_coodinate=torch.arange(self.W).expand(self.H,-1).unsqueeze(0).expand(self.sample_num,-1,-1) \
            .unsqueeze(0).expand(self.B,-1,-1,-1).to(self.device)
        right_x_coodinate=torch.clamp(left_x_coodinate-self.samples,min=0,max=self.W-1)
        right_features=self.rightfeatures.expand(self.sample_num,-1,-1,-1,-1).permute([1,2,0,3,4])

        warped_right_feature=torch.gather(right_features,dim=4,
                                          index=right_x_coodinate.expand(self.rightfeatures.shape[1],-1,-1,-1,-1).permute([1,0,2,3,4]).long()
                                          )
        warped_right_feature=warped_right_feature  # [N,C,sample_num,H,W]
        # wraped_right_featute=warped_right_feature.view(self.B,self.samples.shape[1],-1,self.H,self.W)
        return warped_right_feature

    def featuresPacked(self):
        # 将feature和samples打包
        warped_right_features=self.warp()
        self.leftfeatures=self.leftfeatures.expand(self.samples.shape[1],-1,-1,-1,-1).permute(1,2,0,3,4)
        self.rightfeatures=warped_right_features
        self.samples=self.samples.unsqueeze(1)
        # left_features,rightfeatures=[B,C,sample_num,H,W]
        # samples=[B,1,sample_num,H,W]
        packed=torch.cat((self.leftfeatures,self.rightfeatures,self.samples),dim=1)
        # packed=[B,1+2*C,sample_num,H,W]
        return packed  # 打包的时候不应该把samples给算进去吧？

    def confidence_range_predict(self,packed):
        # 得到packed tensor之后进行一系列的残差连接、卷积、得到最大和最小的预测值，之后再在这个最大最小的预测值中进行patchmath，
        # 再求得一个samples，拼接成volume，丢给aggregation模块，再做refinement.
        packed=self.dres0(packed)
        packed=self.dres1(packed)

        # minPredictor=self.MinDisparityPredictor(self.hourglass_inplanes).to(self.device)
        # maxPredictor=self.MaxDisparityPredictor(self.hourglass_inplanes).to(self.device)
        # disp=[B,1,H,W];feature=[B,sample_num,H,W]
        mindisp,minfeature=self.MinDisparityPredictor(packed,self.samples)
        maxdisp,maxfeature=self.MaxDisparityPredictor(packed,self.samples)
        mindisp=self.min_disp_conv(mindisp)
        maxdisp=self.max_disp_conv(maxdisp)
        maxfeature=self.max_features_conv(maxfeature)
        minfeature=self.min_feature_conv(minfeature)
        return mindisp,maxdisp,minfeature,maxfeature

    def forward(self,leftfeatures,rightfeatures,samples):
        self.leftfeatures=leftfeatures
        self.device=self.leftfeatures.get_device()
        self.W=self.leftfeatures.shape[3]
        self.H=self.leftfeatures.shape[2]
        self.C=self.leftfeatures.shape[1]
        self.B=self.leftfeatures.shape[0]
        self.rightfeatures=rightfeatures
        self.samples=samples

        packed=self.featuresPacked()
        mindisp,maxdisp,minfeature,maxfeature=self.confidence_range_predict(packed)
        return mindisp,maxdisp,minfeature,maxfeature

