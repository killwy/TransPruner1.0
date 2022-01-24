import torch
import torch.nn
# class de (torch.nn):
#     def __init__(self):
#         super(de, self).__init__()
#         self.conv1=torch.nn.Conv2d(3,3,1)
#     def forward(self,input):
#         out1=self.conv1(input)
#         out2=torch.nn.Conv2d(3,3,1)(input)
#         out3=torch.nn.Linear(3,3,True)
#         return out3
import numpy as np
from PIL import Image
# __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
#                     'std': [0.229, 0.224, 0.225]}
# print(*__imagenet_stats)
import torchvision.transforms.transforms as tf


# ospath='/home/FastDataLoader/flying3d/test/25.npy'
# savepath1='/home/jiaxi/25left.png'
# savepath2='/home/jiaxi/25right.png'
# disppath='/home/jiaxi/disp.png'
# mat=np.load(ospath)
# mat=torch.tensor(mat)
# mat1=mat[:,:,0:3]
# mat2=mat[:,:,3:6]
# mat3=mat[:,:,6]
# image_left=Image.fromarray(np.uint8(np.array(mat1)))
# image_right=Image.fromarray(np.uint8(np.array(mat2)))
# image_disp=Image.fromarray(np.uint8(np.array(mat3)))
# image_disp.save(disppath)
# image_right.save(savepath2)
# image_left.save(savepath1)

# path='/home/FastDataLoader/monkey/train.npy'
# a=np.load(path)
# print(a)


# grid_sample实验：
import torch.nn.functional as F

# left_x_coodinate=torch.arange(4).view(1,4).expand(4,4).unsqueeze(0)
# right_x_coodinate=torch.clamp(left_x_coodinate-1,0).unsqueeze(-1)
# right_x_coodinate=(right_x_coodinate-2)/2
# print(right_x_coodinate)
# right_y_coodinate=torch.arange(4).view(4,1).expand(4,4).unsqueeze(0).unsqueeze(-1)
# right_y_coodinate=(right_y_coodinate-2)/2
# print(right_y_coodinate)
# right_xy_coodinate=torch.cat((right_x_coodinate,right_y_coodinate),dim=3)
# print(right_xy_coodinate)
# a=F.grid_sample(left_x_coodinate.unsqueeze(0).float(),right_xy_coodinate.float(),align_corners=True)
# print(left_x_coodinate)
# print(a)

# torchmax实验:按位取较大值
# a=torch.rand((3,3))
# b=torch.rand((3,3))
# c=torch.max(a,b)
# print(a)
# print(b)
# print(c)


# 验证pretrain运行结果。
from deepPruner.deeppruner import deepPruner
from preprocess.Sceneflow_loader import SceneflowDataLoader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn
from deepPruner.config import config
from preprocess.dataLoader import ImageDataSet
import matplotlib.pyplot as plt
import logging
Log_Format='%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='Transpruner1.1_pretrain.log',level=logging.INFO,format=Log_Format)

seed=1
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

model=deepPruner()
device_ids = [0,1]
model=torch.nn.DataParallel(model,device_ids=device_ids)
model=model.cuda(device_ids[0])
model.load_state_dict(torch.load('/home/jiaxi/workspace/TransPruner/kitti_models/TransPruner_finetune_best.tar')['state_dict'])

# 测试Sceneflow
# dataloader=SceneflowDataLoader('/home/jiaxi/workspace/deepPruner/preprocess/Sceneflow_valid.csv',False)
# dataloader=DataLoader(dataloader,batch_size=20,shuffle=False,num_workers=20)
# 测试kitti
dataloader=ImageDataSet('/home/jiaxi/workspace/deepPruner/valid.csv',False)
dataloader=torch.utils.data.DataLoader(dataloader,batch_size=24,shuffle=False,num_workers=28,drop_last=False)

def validation_error_evaluate(gt,pred,mask):
    dif=torch.abs(gt[mask]-pred[mask])
    correct=((dif<3) | (dif<(gt[mask]*0.05))).float()
    correct=torch.sum(correct)
    total=torch.sum(mask.float())
    error_rate=1-correct/total
    return error_rate

def loss_evaluation(gt,pred,mask):
    # Loss_min_disp
    newmask1=((gt[mask]-pred[2][mask])<0).float()
    Loss_min_disp=(gt[mask]-pred[2][mask])*(0.05-newmask1)
    Loss_min_disp=Loss_min_disp.mean()
    Loss_min_disp2=F.smooth_l1_loss(pred[2][mask],gt[mask],size_average=True)
    # Loss_max_disp
    newmask2=((gt[mask]-pred[3][mask])<0).float()
    Loss_max_disp=(gt[mask]-pred[3][mask])*(0.95-newmask2)
    Loss_max_disp=Loss_max_disp.mean()
    Loss_max_disp2=F.smooth_l1_loss(pred[3][mask],gt[mask],size_average=True)
    # Loss_aggregated
    Loss_aggregated=F.smooth_l1_loss(pred[1][mask],gt[mask],size_average=True)
    # Loss_refine
    Loss_refine=F.smooth_l1_loss(pred[0][mask],gt[mask],size_average=True)
    loss=0
    loss+=Loss_refine*1.6+Loss_aggregated+(Loss_min_disp+Loss_max_disp)+(Loss_min_disp2+Loss_max_disp2)*0.7
    return loss
def test(imgL,imgR,disp_L,batch_idx,l_path,r_path):
    model.eval()
    with torch.no_grad():
        # 对于sceneflow
        # imgL=Variable(torch.FloatTensor(imgL))
        # imgR=Variable(torch.FloatTensor(imgR))
        # disp_L=Variable(torch.FloatTensor(disp_L))
        # 对于kitti
        imgL = Variable(imgL.float())
        imgR = Variable(imgR.float())
        disp_L=Variable(torch.FloatTensor(disp_L)).unsqueeze(1)
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
        results=model(imgL,imgR)
        refinedmap=results[0]
        refinedmap=np.array(refinedmap[0][0][:][:].cpu())
        gt=np.array(disp_L[0][0][:][:].cpu())

        mask=disp_true>0
        mask1=np.array(mask[0][0][:][:].cpu().int())
        dif=refinedmap*mask1-gt
        error_rate=validation_error_evaluate(disp_true,results[0],mask)
        if error_rate>0.00:
            plt.imsave('/home/jiaxi/workspace/deepPruner/test/validcases/%d_pred_(Trans1.0)_%3f.png'%(batch_idx,error_rate),refinedmap,cmap='jet',format='png')
            plt.imsave('/home/jiaxi/workspace/deepPruner/test/validcases/%d_gt.png'%(batch_idx),gt,cmap='jet',format='png')
            plt.imsave('/home/jiaxi/workspace/deepPruner/test/validcases/%d_dif_(Trans1.0).png'%(batch_idx),dif,cmap='jet',format='png')
            print(l_path)
            print(r_path)
            imgl=Image.open(l_path)
            imgr=Image.open(r_path)
            imgl.save('/home/jiaxi/workspace/deepPruner/test/validcases/%d_left.png'%(batch_idx))
            imgr.save('/home/jiaxi/workspace/deepPruner/test/validcases/%d_right.png'%(batch_idx))
        loss=loss_evaluation(disp_true,results,mask)
        return error_rate.cpu().item(),loss.item()

def main():
    total_error_rate=0
    for batch_idx,(left,right,disp,l_path,r_path) in enumerate(dataloader):
        print("batch_idx %d :"%(batch_idx))
        error_rate,loss=test(left,right,disp,batch_idx,l_path[0],r_path[0])
        total_error_rate+=error_rate
        print("3-pixel-error_rate: %f ,loss: %f"%(error_rate,loss))
    print("3-pixel-error_rate: %f "%(total_error_rate/len(dataloader)))
if __name__=="__main__":
    main()





