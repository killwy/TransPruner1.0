import torch
import torch.nn
import os
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
model=deepPruner()
device_ids = [0,1]
model=torch.nn.DataParallel(model,device_ids=device_ids)
model=model.cuda(device_ids[0])
# transpruner
# model_dict=torch.load('../pretrain_models/deepPruner_raw_pretrain_63.tar')
# deeppruner
# model_dict=torch.load('../pretrain_models/deepPruner_raw_pretrain_62.tar')
model_dict=torch.load('../sceneFlowTrainedModel/deepPruner_raw_pretrain_61.tar')
model.load_state_dict(model_dict['state_dict'])

# 测试Sceneflow
dataloader=SceneflowDataLoader('../preprocess/Sceneflow_valid.csv',False)
dataloader=DataLoader(dataloader,batch_size=1,shuffle=False,num_workers=20)
# 测试kitti
# dataloader=ImageDataSet('/home/jiaxi/workspace/deepPruner/annotation_valid.csv','/home/jiaxi/workspace/KITTI/training')
# dataloader=torch.utils.data.DataLoader(dataloader,batch_size=1*4,shuffle=False,num_workers=4,drop_last=False)

def validation_error_evaluate(gt,pred,mask):
    correct=((torch.abs(gt[mask]-pred[mask])<3)).float()
    correct=torch.sum(correct)
    total=torch.sum(mask.float())
    error_rate=1-correct/total
    return error_rate
def end_point_error(gt,pred,mask):
    diff=torch.abs(gt[mask]-pred[mask])
    epe=torch.mean(diff)
    return epe
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
def test(imgL,imgR,disp_L,batch_idx,left_img,right_img):
    model.eval()
    with torch.no_grad():
        # 对于
        imgL=Variable(torch.FloatTensor(imgL))
        imgR=Variable(torch.FloatTensor(imgR))
        disp_L=Variable(torch.FloatTensor(disp_L))
        # imgL = Variable(imgL.float())
        # imgR = Variable(imgR.float())
        # disp_L=Variable(torch.FloatTensor(disp_L)).unsqueeze(1)
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
        results=model(imgL,imgR)
        mask=disp_true<config.max_disp
        error_rate=validation_error_evaluate(disp_true,results[0],mask)
        loss=end_point_error(disp_true,results[0],mask)
        # 可视化
        if loss>2:
            refinedmaps=results[0]
            print(refinedmaps.shape)
            # with open('badcases/paths.txt','a+') as fp:
            for i in range(refinedmaps.shape[0]):
                # fp.write('batch_id:%d i:%d '%(batch_idx,i))
                # fp.write(right_path[i]+' ')
                # fp.write(right_path[i]+' \n')
                left=Image.fromarray(np.uint8(np.array(left_img[i][:][:][:])))
                right=Image.fromarray(np.uint8(np.array(right_img[i][:][:][:])))
                left.save('badcases/%d_%d_left.png'%(batch_idx,i))
                right.save('badcases/%d_%d_right.png'%(batch_idx,i))
                refinedmap=np.array(refinedmaps[i][0][:][:].cpu())
                gt=np.array(disp_L[i][0][:][:].cpu())
                dif=refinedmap-gt
                plt.imsave('badcases/%d_%d_pred(1)_loss_%.2f.png' %(batch_idx,i,loss),refinedmap,cmap='jet',vmin=0,vmax=192)
                plt.imsave('badcases/%d_%d_gt.png' %(batch_idx,i),gt,cmap='jet',vmin=0,vmax=192)
                plt.imsave('badcases/%d_%d_dif(1)_loss_%.2f.png' %(batch_idx,i,loss),dif,cmap='jet',vmin=-192,vmax=192)
                # plt.imsave('badcases/%d_%d_pred(TransPruner1.0).png' %(batch_idx,i),refinedmap,cmap='jet')
                # plt.imsave('badcases/%d_%d_gt(TransPruner1.0).png' %(batch_idx,i),gt,cmap='jet')
                # plt.imsave('badcases/%d_%d_dif(TransPruner1.0).png' %(batch_idx,i),dif,cmap='jet')
        return error_rate.cpu().item(),loss.item()

def main():
    totalloss=0
    for batch_idx,(left,right,disp,left_img,right_img) in enumerate(dataloader):
        print("batch_idx %d :"%(batch_idx))
        error_rate,loss=test(left,right,disp,batch_idx,left_img,right_img)
        totalloss+=loss
        print("3-pixel-error_rate: %f ,epe-loss: %f"%(error_rate,loss))

    print('average-epe-loss:%f'%(totalloss/len(dataloader)))

if __name__=="__main__":
    main()

