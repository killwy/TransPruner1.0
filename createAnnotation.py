import pandas
import os
import re
import random
annotation_file_name1='annotation_train.csv'
annotation_file_name2='annotation_valid.csv'
dataset_path='/home/jiaxi/workspace/KITTI/training'
for x,y,fnames1 in os.walk(dataset_path+'/colored_0'):
    print()
trainSet=[]
validSet=[]
for x in fnames1:
    if re.match('.+1\\.png',x) is None:
        print(x)
        if(random.random()<0.2):
            validSet.append(x)
        else:
            trainSet.append(x)
df=pandas.DataFrame({'left image':trainSet})
df.to_csv(annotation_file_name1)
df2=pandas.DataFrame({'left image':validSet})
df2.to_csv(annotation_file_name2)

