import torch
import torch.nn as nn

class uniformSample(nn.Module):
    def __init__(self,max_disp,min_disp):
        super(uniformSample, self).__init__()
        self.max_disp=max_disp
        self.min_disp=min_disp

    def forward(self,mindisp,maxdisp,sample_count):
        mindisp1=torch.min(mindisp,maxdisp)
        maxdisp1=torch.max(mindisp,maxdisp)
        # 拉伸，使得最少能容纳sample_count个
        min_disparity = torch.clamp(mindisp1 - torch.clamp((
                sample_count - maxdisp1 + mindisp1), min=0) / 2.0, min=0, max=self.max_disp)
        max_disparity = torch.clamp(maxdisp1 + torch.clamp(
            sample_count - maxdisp1 + min_disparity, min=0), min=0, max=self.max_disp)
        # 采样
        device=mindisp.get_device()
        multiplier = (max_disparity - min_disparity) / (sample_count + 1)
        range_multiplier = torch.arange(1.0, sample_count + 1, 1, device=device).view(sample_count, 1, 1)
        sampled_disparities = min_disparity + multiplier * range_multiplier

        sampled_disparities = torch.cat((torch.floor(min_disparity), sampled_disparities, torch.ceil(max_disparity)),
                                      dim=1).long()
        return sampled_disparities