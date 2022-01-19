import pandas
import os
import numpy as np
import random

# 4卡服务器上的地址
# monkey_path='/home/2T/FastDataLoader_RecoverSet/monkey/train'
# driving_path='/home/FastDataLoader/driving/train'
# flying3d_path='/home/FastDataLoader/flying3d/train'
# test_path='/home/FastDataLoader/flying3d/test'
# 3090上地址
import torch

monkey_path='/data/sceneflow/monkaa/frames_cleanpass'
driving_path='/data/sceneflow/driving/frames_cleanpass/15mm_focallength'  # 选用15mm因为更适合在kitti上finetune
flying3d_path='/data/sceneflow/flying3d/frames_cleanpass/TRAIN'
test_path='/data/sceneflow/flying3d/frames_cleanpass/TEST'

def resolve(path):
    left_list=[]
    right_list=[]
    disp_list=[]
    for current_path,directory,files in os.walk(path):
        if '/left' not in current_path:
            continue
        right_path=current_path.replace('/left','/right')
        disp_path=current_path.replace('/frames_cleanpass/','/disparity/')
        if len(files)>0:
            left_list=left_list+[current_path+'/'+x for x in files]
            right_list=right_list+[right_path+'/'+x for x in files]
            disp_list=disp_list+[disp_path+'/'+x.replace('.png','.pfm') for x in files]
    return left_list,right_list,disp_list


monkey_left,monkey_right,monkey_disp=resolve(monkey_path)

flying3d_left,flying3d_right,flying3d_disp=resolve(flying3d_path)

test_left,test_right,test_disp=resolve(test_path)

driving_left,driving_right,driving_disp=resolve(driving_path)
train_left=monkey_left+flying3d_left+driving_left
train_right=monkey_right+flying3d_right+driving_right
train_disp=monkey_disp+flying3d_disp+driving_disp
df1=pandas.DataFrame({'left':train_left,'right':train_right,'disp':train_disp})
df1.to_csv('Sceneflow_train.csv')
df2=pandas.DataFrame({'left':test_left,'right':test_right,'disp':test_disp})
df2.to_csv('Sceneflow_valid.csv')


torch.tensor([])

# # 4卡服务器版本
# for a1,b,flying3d_file_name in os.walk(flying3d_path):
#     print(flying3d_file_name)
# for a2,b,driving_file_name in os.walk(driving_path):
#     print(driving_file_name)
# for a3,b,monkey_file_name in os.walk(monkey_path):
#     print(monkey_file_name)
# for a4,b,test_file_name in os.walk(test_path):
#     print(test_file_name)
#
# num_flying3d=len(flying3d_file_name)
# num_driving=len(driving_file_name)
# num_monkey=len(monkey_file_name)
# num_test=len(test_file_name)
#
# sum_num=num_flying3d+num_driving+num_monkey
# valid_set=[]
# train_set=[]
# valid_num=0
# valid_num_flying3d=int(valid_num*num_flying3d/sum_num)
# valid_num_driving=int(valid_num*num_driving/sum_num)
# valid_num_monkey=int(valid_num*num_monkey/sum_num)
#
# for x in flying3d_file_name:
#     train_set.append(a1+'/'+x)
#
# for x in driving_file_name:
#     train_set.append(a2+'/'+x)
#
# for x in monkey_file_name:
#     train_set.append(a3+'/'+x)
#
# for x in test_file_name:
#     valid_set.append(a4+'/'+x)
#
# print('train:',num_flying3d+num_driving+num_monkey)
# print('test:',num_test)
# df1=pandas.DataFrame({'trainingSet':train_set})
# df1.to_csv('Sceneflow_train.csv')
# df2=pandas.DataFrame({'validSet':valid_set})
# df2.to_csv('Sceneflow_valid.csv')
