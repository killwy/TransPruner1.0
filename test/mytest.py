# from PIL import Image
# img=Image.open('/home/jiaxi/workspace/KITTI/training/colored_0/000000_10.png','r')
# img=img.crop((-10,-10,720,1260))
# w,h=img.size
# print(w,h)
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
def man_cmap(cmap):
    colors = cmap(np.arange(cmap.N))
    colors[255]=[1,1,1,1]
    return mcolors.LinearSegmentedColormap.from_list("", colors)

jet=plt.cm.get_cmap('jet')
img=np.arange(256).reshape([16,16])
plt.imsave('1.png',img,cmap=man_cmap(jet),vmax=192,vmin=0)