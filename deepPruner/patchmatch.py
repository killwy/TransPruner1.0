import numpy
from torch import nn
import torch.nn.functional as F
import torch
from deepPruner.config import config
from PIL import Image
import numpy as np

class patchMatch(nn.Module):
    def __init__(self):
        super(patchMatch,self).__init__()
        self.B=None
        self.C=None
        self.H=None
        self.W=None
        self.device=None
        self.minRange=None
        self.maxRange=None
        self.prop_times=3
        self.iter_times=config.patch_match_args.iteration_count
        self.left_features=None
        self.right_features=None
        self.max_disp=config.max_disp
        self.min_disp=config.min_disp
        self.sample_num=config.patch_match_args.sample_count
        self.left_disp=None
        self.filtersize=config.patch_match_args.propagation_filter_size
        self.disp_window=config.disp_window
        self.temperature=config.temperature
        self.prop_conv2d=nn.functional.conv2d

    def randomSample(self,mode):
        # segments shape=[B,sample_num,H,W]
        left_disp_candidates=None
        if mode=='init':
            segments = torch.arange(self.min_disp,self.max_disp,(self.max_disp-self.min_disp)/self.sample_num)\
                .view(self.sample_num,1).unsqueeze_(0).expand(
                (self.B, -1, self.H * self.W)).view(self.B, -1, self.H, self.W)
            randsample = torch.rand(segments.shape) * ((self.max_disp - self.min_disp) / self.sample_num)
            left_disp_candidates = segments + randsample
            left_disp_candidates.clamp_(self.min_disp, self.max_disp-1)
        elif mode=='post':
            gap=self.maxRange-self.minRange
            segments=torch.arange(0,self.sample_num).view(self.sample_num,1).unsqueeze_(0).expand(self.B,-1,self.H*self.W).view(self.B, -1, self.H, self.W).to(self.device)
            randsample = torch.rand(segments.shape).to(self.device) + segments
            left_disp_candidates = (gap/self.sample_num).repeat(1,self.sample_num,1,1).mul(randsample)+self.minRange
        # [B,sample_num,H,W]
        return left_disp_candidates

    def evaluate(self,disp_candidates):
        # 构建坐标系
        left_x_coodinate=torch.arange(self.W).expand(self.H,-1).unsqueeze(0).expand(self.sample_num*self.prop_times,-1,-1)\
            .unsqueeze(0).expand(self.B,-1,-1,-1).to(self.device)
        right_x_coodinate=torch.clamp(left_x_coodinate-disp_candidates,min=0,max=self.W-1)
        right_features=self.right_features.expand(self.sample_num*self.prop_times,-1,-1,-1,-1).permute([1,2,0,3,4])
        left_features=self.left_features.expand(self.sample_num*self.prop_times,-1,-1,-1,-1).permute([1,2,0,3,4])
        warped_right_feature=torch.gather(right_features,dim=4,
                                          index=right_x_coodinate.expand(self.right_features.shape[1],-1,-1,-1,-1).permute([1,0,2,3,4]).long()
                                          )
        weight=torch.mean(left_features * warped_right_feature, dim=1) * self.temperature
        weight=weight.view(self.B,self.sample_num,self.prop_times,self.H,self.W)
        disp_candidates=disp_candidates.view(self.B,self.sample_num,self.prop_times,self.H,self.W)
        weight=torch.softmax(weight,dim=2)
        disp_pred = torch.sum(disp_candidates * weight, dim=2)
        return disp_pred

    def warp(self,disp):
        left_x_coodinate = torch.arange(self.W).unsqueeze_(0).expand(self.H, -1).unsqueeze_(0).unsqueeze_(0).expand(
            self.B, 1, self.H, self.W)
        right_x_coodinate = torch.clamp(left_x_coodinate.clone().float() - disp, min=0, max=self.W - 1)
        right_y_coodinate = torch.arange(self.H).unsqueeze_(1).expand(-1, self.W).unsqueeze_(0).unsqueeze_(0).expand(
            self.B, right_x_coodinate.shape[1], self.H, self.W)
        right_x_coodinate -= right_x_coodinate.shape[3] / 2
        right_x_coodinate /= (right_x_coodinate.shape[3] / 2)
        right_y_coodinate = right_y_coodinate - right_y_coodinate.shape[2] / 2
        right_y_coodinate /= (right_y_coodinate.shape[2] / 2)
        right_xy_coodinate = torch.cat((right_y_coodinate.unsqueeze_(4).float(), right_x_coodinate.unsqueeze_(4)),
                                       dim=4)
        # right_xy_coodinate.shape=[N*5*sample_num,H,W,2]
        right_xy_coodinate = right_xy_coodinate.view(right_xy_coodinate.shape[0] * right_xy_coodinate.shape[1],
                                                     right_xy_coodinate.shape[2], right_xy_coodinate.shape[3],
                                                     right_xy_coodinate.shape[4])
        # disp_candidates.shape=[N,5*sample_num,H,W];right_features=[N,C,H,W]
        wraped_right_featutes = F.grid_sample(
            self.right_features.expand(disp.shape[1] * self.B, self.C, self.H, self.W).float(),
            right_xy_coodinate,align_corners=True)
        return wraped_right_featutes

    def propagate(self,disp_candidates,mode):
        # 往上下左右做平移并且合并。
        # oneHotFilter=torch.tensor([[
        #     [[0, 1, 0],
        #      [0, 0, 0],
        #      [0, 0, 0]],
        #     [[0, 0, 0],
        #      [1, 0, 0],
        #      [0, 0, 0]],
        #     [[0, 0, 0],
        #      [0, 1, 0],
        #      [0, 0, 0]],
        #     [[0, 0, 0],
        #      [0, 0, 1],
        #      [0, 0, 0]],
        #     [[0, 0, 0],
        #      [0, 0, 0],
        #      [0, 1, 0]]
        # ]]).permute([1,0,2,3]).repeat(self.sample_num,1,1,1).float().to(self.device)
        # # 使用复制pad来填充
        # pad=nn.ReplicationPad2d((self.filtersize//2,self.filtersize//2,self.filtersize//2,self.filtersize//2))
        # disp_candidates=pad(disp_candidates)
        # prop_disp=self.prop_conv2d(disp_candidates,oneHotFilter,groups=self.sample_num)
        # 返回[B,sample_num*5,H,W]
        disp_candidates=disp_candidates.view(self.B,1,self.sample_num,self.H,self.W)
        if mode=='horizontal':
            label= torch.arange(0,self.filtersize,device=self.device).repeat(self.filtersize).view(self.filtersize,1,1,1,self.filtersize)
            one_hot_filter = torch.zeros_like(label).scatter_(0, label, 1).float()
            pad=nn.ReplicationPad3d((self.filtersize // 2,self.filtersize // 2,0,0,0,0))
            disp_candidates=pad(disp_candidates)
            aggregated_disparity_samples = F.conv3d(disp_candidates,
                                                    one_hot_filter)
        else:
            label= torch.arange(0,self.filtersize,device=self.device).repeat(self.filtersize).view(self.filtersize,1,1,self.filtersize,1)
            one_hot_filter = torch.zeros_like(label).scatter_(0, label, 1).float()
            pad=nn.ReplicationPad3d((0,0,self.filtersize // 2,self.filtersize // 2,0,0))
            disp_candidates=pad(disp_candidates)
            aggregated_disparity_samples = F.conv3d(disp_candidates,
                                                    one_hot_filter)

        prop_disp=aggregated_disparity_samples.permute([0,2,1,3,4]).contiguous().view(self.B,self.sample_num*self.filtersize,self.H,self.W)

        return prop_disp

    def forward(self,left_features,right_features,minrange=None,maxrange=None):
        self.left_features=left_features
        self.right_features=right_features
        self.device=self.left_features.get_device()
        self.B=int(self.left_features.shape[0])
        self.C=int(self.left_features.shape[1])
        self.H=int(self.left_features.shape[2])
        self.W=int(self.left_features.shape[3])

        if minrange is not None and maxrange is not None:
            mode="post"
            self.maxRange=torch.max(minrange,maxrange)
            self.minRange=torch.min(minrange,maxrange)
            min_disparity = torch.clamp(self.minRange - torch.clamp((
                    self.sample_num - self.maxRange + self.minRange), min=0) / 2.0, min=0, max=self.max_disp)
            max_disparity = torch.clamp(self.maxRange + torch.clamp(
                self.sample_num - self.maxRange + min_disparity, min=0), min=0, max=self.max_disp)
            self.maxRange=max_disparity
            self.minRange=min_disparity
        else:
            mode='init'
        disp_candidates = self.randomSample(mode=mode).to(self.device)
        for i in range(self.iter_times):
            disp_candidates=self.propagate(disp_candidates,mode='horizontal')
            disp_candidates=self.evaluate(disp_candidates)
            disp_candidates=self.propagate(disp_candidates,mode='vertical')
            disp_candidates=self.evaluate(disp_candidates)
        if mode=='post':
            disp_candidates = torch.cat((torch.floor(self.minRange), disp_candidates, torch.ceil(self.maxRange)),
                                            dim=1).long()
        return disp_candidates

if __name__=="__main__":
    left=Image.open('/home/mcislaber/WorkSpace_zjx/dataSet/KITTI/training/colored_0/000000_10.png')
    right=Image.open('/home/mcislaber/WorkSpace_zjx/dataSet/KITTI/training/colored_1/000000_10.png')
    left=numpy.array(left)
    right=numpy.array(right)
    left=torch.from_numpy(left).permute(2,0,1).unsqueeze_(0)
    right=torch.from_numpy(right).permute(2,0,1).unsqueeze_(0)
    x=patchMatch(left,right)
    disp_pred=x.forward()

