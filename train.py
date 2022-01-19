from deepPruner.deeppruner import deepPruner
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from deepPruner.config import config
from preprocess.dataLoader import ImageDataSet
import torch.nn.functional as F
import logging
import torch.utils.data
from visdom import Visdom
import time
import numpy as np
# 设置随机数
seed=1
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
# visdom创建列表
wind1 = Visdom()
wind1.line([0.], # Y的第一个点的坐标
           [0.], # X的第一个点的坐标
           win = 'train_loss', # 窗口的名称
           opts = dict(title = 'train_loss') # 图像的标例
           )
wind1.line([0.], # Y的第一个点的坐标
           [0.], # X的第一个点的坐标
           win = 'valid_3-pixl-error-rate', # 窗口的名称
           opts = dict(title = '3-pixl-error-rate') # 图像的标例
           )
# wind1.line([0.], # Y的第一个点的坐标
#            [0.], # X的第一个点的坐标
#            win = 'in-epoch-valid_3-pixl-error-rate', # 窗口的名称
#            opts = dict(title = 'in-epoch-3-pixl-error-rate') # 图像的标例
#            )

# 加载模型
model=deepPruner()
device_ids = [0,1]
model = torch.nn.DataParallel(model, device_ids=device_ids) # 声明所有可用设备
model = model.cuda(device=device_ids[0])  # 模型放在主设备
model.load_state_dict(torch.load('/home/jiaxi/workspace/TransPruner/pretrain_models/deepPruner_raw_pretrain_62.tar')['state_dict'])
# 加载dataloader
trainloader=ImageDataSet('/home/jiaxi/workspace/deepPruner/train.csv',training=True)
train_dataloader=torch.utils.data.DataLoader(trainloader,batch_size=16,shuffle=True,num_workers=20,drop_last=False)
validloader=ImageDataSet('/home/jiaxi/workspace/deepPruner/valid.csv',training=False)
valid_dataloader=torch.utils.data.DataLoader(validloader,batch_size=20,shuffle=False,num_workers=20,drop_last=False)
optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

# lr change
def adjust_learning_rate(optimizer, epoch):
    if epoch <= 500:
        lr = 0.0001
    elif epoch<=1000:
        lr = 0.00005
    else:
        lr = 0.00001
    # logging.info('learning rate = %.5f' %(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 3-pixel error function
def validation_error_evaluate(gt,pred,mask):
    dif=torch.abs(gt[mask]-pred[mask])
    correct=((dif<3) | (dif<(gt[mask]*0.05))).float()
    correct=torch.sum(correct)
    total=torch.sum(mask.float())
    error_rate=1-(1.0*correct/total)
    return error_rate

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
    loss+=Loss_refine*1.3+Loss_aggregated+(Loss_min_disp+Loss_max_disp)+(Loss_min_disp2+Loss_max_disp2)*0.7
    logging.info("============== evaluated losses ==================")
    logging.info('refined_depth_loss: %.6f' % Loss_refine)
    logging.info('ca_depth_loss: %.6f' % Loss_aggregated)
    logging.info('quantile_loss_max_disparity: %.6f' % Loss_max_disp)
    logging.info('quantile_loss_min_disparity: %.6f' % Loss_min_disp)
    logging.info('max_disparity_loss: %.6f' % Loss_max_disp2)
    logging.info('min_disparity_loss: %.6f' % Loss_min_disp2)
    logging.info("==================================================\n")

    return loss

# a process for training
def train(imgL, imgR, disp_L,epoch):
    if epoch >= 800:
        model.eval()
    else:
        model.train()
    imgL = Variable(imgL.float())
    imgR = Variable(imgR.float())
    disp_L = Variable(torch.FloatTensor(disp_L)).unsqueeze(1)
    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
    mask = disp_true>0  # 因为gt是稀疏的，所以我们只在有点处进行mask
    mask.detach_()  # 阻止反向传播
    optimizer.zero_grad()
    result = model(imgL, imgR)
    loss= loss_evaluation(disp_true, result, mask)
    loss.backward()
    optimizer.step()
    return loss.item()

# a process for testing
def test(imgL,imgR,disp_L):
    model.eval()
    with torch.no_grad(): # 不需要构建计算图也不需要反向传播
        imgL=Variable(imgL.float())
        imgR=Variable(imgR.float())
        disp_L=Variable(torch.FloatTensor(disp_L)).unsqueeze(1)
        imgL, imgR, disp_true = imgL.cuda(0), imgR.cuda(0), disp_L.cuda(0)
        mask=(disp_true>0)
        mask.detach_()
        optimizer.zero_grad()
        result = model(imgL, imgR)
        loss= loss_evaluation(disp_true, result, mask)
        error_rate=validation_error_evaluate(disp_true,result[0],mask)
    return error_rate.cpu().item(),loss

# main function
def main():
    best_test_loss=1
    best_test_loss_epoch=0
    for epoch in range(config.training_epoches):
        start=time.time()
        total_train_loss = 0
        total_test_loss = 0
        adjust_learning_rate(optimizer,epoch)

        # training
        for i,(left,right,disp) in enumerate(train_dataloader):
            loss=train(left,right,disp,epoch)
            total_train_loss += loss
            print('Iter %d training loss = %.3f \n' %(i, loss))
        mean_train_loss=total_train_loss/len(train_dataloader)
        print(('epoch %d total training loss = %.3f' %(epoch, mean_train_loss)))
        wind1.line([mean_train_loss],[epoch],win='train_loss',update='append')

        # testing
        for i,(left,right,disp) in enumerate(valid_dataloader):
            loss=test(left,right,disp)
            total_test_loss += loss[0]
            print('Iter %d 3-px error in val = %.3f ' %(i, loss[0]))
            # wind1.line([loss[0]],[i],win='in-epoch-valid_3-pixl-error-rate',update='append')
        mean_test_loss=total_test_loss/len(valid_dataloader)
        print(('epoch %d total test error = %.3f' %(epoch, mean_test_loss)))
        wind1.line([mean_test_loss],[epoch],win='valid_3-pixl-error-rate',update='append')

        # save the model
        if epoch>600:
            if mean_test_loss<best_test_loss:
                best_test_loss=mean_test_loss
                best_test_loss_epoch=epoch
                savefilename = "/home/jiaxi/workspace/TransPruner/kitti_models/TransPruner_"+'finetune_best'+'.tar'
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss,
                    'test_loss': total_test_loss,
                }, savefilename)
        end=time.time()
        print('time:',end-start,'s')
    print("best_test_loss:",best_test_loss)
    print("best_test_loss_epoch:",best_test_loss_epoch)

if __name__=='__main__':
    main()

