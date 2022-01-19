import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from preprocess.Sceneflow_loader import SceneflowDataLoader
from deepPruner.deeppruner import deepPruner
from torch.autograd import Variable
from deepPruner.config import config
import torch.nn.functional as F
import time
from visdom import Visdom

# 设置visdom窗口
wind1 = Visdom()
wind1.line([0.],  # Y的第一个点的坐标
           [0.],  # X的第一个点的坐标
           win='train_loss', # 窗口的名称
           opts=dict(title='training_loss')  # 图像的标例
           )
wind1.line([0.],  # Y的第一个点的坐标
           [0.],  # X的第一个点的坐标
           win='valid_loss',  # 窗口的名称
           opts=dict(title='valid_loss')  # 图像的标例
           )
wind1.line([0.],  # Y的第一个点的坐标
           [0.],  # X的第一个点的坐标
           win='valid_epe_loss',  # 窗口的名称
           opts=dict(title='valid_epe_loss')  # 图像的标例
           )
wind1.line([0.],  # Y的第一个点的坐标
           [0.],  # X的第一个点的坐标
           win='in_epoch_loss',  # 窗口的名称
           opts=dict(title='in_epoch_loss')  # 图像的标例
           )

# 设置seed
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# dataloader
train_loader=SceneflowDataLoader(file_path='/home/jiaxi/workspace/deepPruner/preprocess/Sceneflow_train.csv',train=True)
valid_loader=SceneflowDataLoader(file_path='/home/jiaxi/workspace/deepPruner/preprocess/Sceneflow_valid.csv',train=False)
trainImgLoader=DataLoader(train_loader,batch_size=16,shuffle=True,num_workers=28,drop_last=False)
testImgLoader=DataLoader(valid_loader,batch_size=20,shuffle=False,num_workers=28,drop_last=False)

# 初始化模型,并行化模型
model = deepPruner()
device_ids = [0,1]
model=torch.nn.DataParallel(model,device_ids=device_ids)
model = model.cuda(device=device_ids[0])  # 模型放在主设备
# print(summary(model,[(3,256,512),(3,256,512)],batch_size=1))

# 初始化优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# loss function
def loss_evaluation(gt,pred,mask):
    # Loss_min_disp
    # mask=mask.int()
    # print(mask)
    # print(type(mask))
    # print(mask)
    # print(pred[2])
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
    # logging.info("============== evaluated losses ==================")
    # logging.info('refined_depth_loss: %.6f' % Loss_refine)
    # logging.info('ca_depth_loss: %.6f' % Loss_aggregated)
    # logging.info('quantile_loss_max_disparity: %.6f' % Loss_max_disp)
    # logging.info('quantile_loss_min_disparity: %.6f' % Loss_min_disp)
    # logging.info('max_disparity_loss: %.6f' % Loss_max_disp2)
    # logging.info('min_disparity_loss: %.6f' % Loss_min_disp2)
    # logging.info("==================================================\n")

    return loss

# 3-pixel error function
def validation_error_evaluate(gt,pred,mask):
    correct=((torch.abs(gt[mask]-pred[mask])<3)).float()
    correct=torch.sum(correct)
    total=torch.sum(mask.float())
    error_rate=1-correct/total
    return error_rate

# endpoint error
def end_point_error(gt,pred,mask):
    diff=torch.abs(gt[mask]-pred[mask])
    epe=torch.mean(diff)
    return epe

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 20:
        lr = 0.001
    elif epoch <= 40:
        lr = 0.0007
    elif epoch <= 60:
        lr = 0.0003
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(imgL, imgR, disp_L):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))
    imgL, imgR, disp_true = imgL.cuda(0), imgR.cuda(0), disp_L.cuda(0)
    mask = disp_true < config.max_disp
    mask.detach_()
    optimizer.zero_grad()
    result = model(imgL, imgR)
    loss= loss_evaluation(disp_true, result, mask)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(imgL, imgR, disp_L):
    model.eval()
    with torch.no_grad():
        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))
        disp_L = Variable(torch.FloatTensor(disp_L))
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
        mask = disp_true < config.max_disp
        mask.detach_()
        if len(disp_true[mask]) == 0:
            print("invalid GT disaprity...")
            return 0, 0
        optimizer.zero_grad()
        result = model(imgL, imgR)
        loss = loss_evaluation(disp_true, result, mask)
        epe_loss= end_point_error(disp_true,result[0],mask)
    return loss.item(),epe_loss.item()

def main():
    # EPOCH
    for epoch in range(0, 64):
        total_train_loss = 0
        total_test_loss = 0
        total_epe_loss = 0
        adjust_learning_rate(optimizer, epoch)

        # TRAIN
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(trainImgLoader):
            start_time = time.time()
            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            total_train_loss += loss
            if batch_idx==0:
                wind1.line([loss],[batch_idx],win='in_epoch_loss')
            else:
                wind1.line([loss],[batch_idx],win='in_epoch_loss',update='append')
            print('Iter %d training loss = %.3f , time = %.2f \n' % (batch_idx, loss, time.time() - start_time))
        wind1.line([total_train_loss/len(trainImgLoader)],[epoch],win='train_loss',update='append')
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(trainImgLoader)))

        # TEST
        if epoch % 1 == 0 :
            print("testing...")
            for batch_idx, (imgL, imgR, disp_L) in enumerate(testImgLoader):
                start_time = time.time()
                test_loss,epe_loss= test(imgL, imgR, disp_L)
                total_test_loss += test_loss
                total_epe_loss+=epe_loss
                print('Iter %d 3-px error in val = %.3f, time = %.2f \n' %
                      (batch_idx, test_loss, time.time() - start_time))
            wind1.line([total_epe_loss/len(testImgLoader)],[epoch],win='valid_epe_loss',update='append')
            wind1.line([total_test_loss/len(testImgLoader)],[epoch],win='valid_loss',update='append')
            print('epoch %d total test loss = %.3f' % (epoch, total_test_loss / len(testImgLoader)))

        # SAVE
        if epoch % 1 == 0:
            savefilename = '/home/jiaxi/workspace/TransPruner/pretrain_models/deepPruner_raw_pretrain_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss,
                'test_loss': total_test_loss,
            }, savefilename)


if __name__ == '__main__':
    main()