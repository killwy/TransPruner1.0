from torch import nn
import torch
from deepPruner.modules import BottleNeck
from deepPruner.modules import conv3d_bn_lrelu
class aggregate(BottleNeck):
    def __init__(self,inplanes,hourglass_inplanes=16):
        super(aggregate, self).__init__(hourglass_inplanes)
        self.leftfeatures=None
        self.rightfeatures=None
        self.samples=None
        self.device=None
        self.dres0=nn.Sequential(
            conv3d_bn_lrelu(inplanes, 64, 3, 1, 1),
            conv3d_bn_lrelu(64, 32, 3, 1, 1)
        )
        self.dres1=nn.Sequential(
            conv3d_bn_lrelu(32, 32, 3, 1, 1),
            conv3d_bn_lrelu(32, hourglass_inplanes, 3, 1, 1)
        )

    def featuresPacked(self,max_disp_feature,min_disp_feature):
        # 将feature和samples打包
        self.leftfeatures=self.leftfeatures.expand(self.samples.shape[1],-1,-1,-1,-1).permute(1,2,0,3,4)
        self.rightfeatures=self.rightfeatures.expand(self.samples.shape[1],-1,-1,-1,-1).permute(1,2,0,3,4)
        self.samples=self.samples.unsqueeze(1)
        # min_disp_feature=max_disp_feature=[B,sample_num,sample_num,H,W]
        max_disp_feature=max_disp_feature.unsqueeze(2).expand(-1,-1,self.samples.shape[2],-1,-1)
        min_disp_feature=min_disp_feature.unsqueeze(2).expand(-1,-1,self.samples.shape[2],-1,-1)
        # left_features,rightfeatures=[B,C,sample_num,H,W]
        # samples=[B,1,sample_num,H,W]
        packed=torch.cat((self.leftfeatures,self.rightfeatures,self.samples,max_disp_feature,min_disp_feature),dim=1)
        # packed=[B,1+2*C,sample_num,H,W]
        return packed
    def forward(self,leftfeatures,rightfeatures,samples,max_disp_feature,min_disp_feature):
        self.leftfeatures=leftfeatures
        self.rightfeatures=rightfeatures
        self.samples=samples
        self.device=self.samples.get_device()

        packed=self.featuresPacked(max_disp_feature,min_disp_feature)
        out0=self.dres0(packed)
        out_record0=self.dres1(out0)

        out0=self.layer1(out_record0)
        out_record1=self.layer2(out0)+out0

        out0=self.layer3(out_record1)
        out_record2=self.layer4(out0)+out0

        out0=self.layer5(out_record2)
        out_record3=self.layer6(out0)+out0

        out0=self.layer7(out_record3)+out_record2
        out0=self.layer8(out0)+out_record1
        out0=self.layer9(out0)+out_record0

        out0=self.last_conv3d_layer(out0).squeeze(1)
        feature_output = out0

        confidence_output = self.softmax(out0)
        disparity_output = torch.sum(confidence_output * (self.samples.squeeze(1)), dim=1)

        return disparity_output.unsqueeze(1), feature_output  # 置信度特征
