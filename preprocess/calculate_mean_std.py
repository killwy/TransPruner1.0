import random

from PIL import Image
import pandas
import numpy as np
import torch
# pd=pandas.read_csv('/home/jiaxi/workspace/deepPruner/annotation_train.csv')
pd=pandas.read_csv('/home/jiaxi/workspace/deepPruner/preprocess/Sceneflow_train.csv')
img_dir1='/home/jiaxi/workspace/KITTI/training/colored_0/'
img_dir2='/home/jiaxi/workspace/KITTI/training/colored_0/'

total_mean=0
total_std=0
cat=None
for i in range(pd.__len__()):
    if random.random()<0.10:
        # left=Image.open(img_dir1+str(pd.iloc[i,1]))
        # right=Image.open(img_dir2+str(pd.iloc[i,1]))
        # left=Image.open(str(pd.iloc[i,1]))
        # right=Image.open(str(pd.iloc[i,1]))
        nump=np.load(str(pd.iloc[i,1]))[0:3,:,:]
        left=nump[0:3,:,:]
        right=nump[3:6,:,:]
        left_=torch.from_numpy(left).view(3,-1).float().cuda(1)/255
        right_=torch.from_numpy(left).view(3,-1).float().cuda(1)/255
        if cat is None:
            cat=torch.cat((left_,right_),dim=1)
        else:
            cat=torch.cat((cat,left_,right_),dim=1)
print(cat.shape)
print(torch.mean(cat,dim=1).cpu(),torch.std(cat,dim=1).cpu())



