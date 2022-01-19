from PIL import Image
img=Image.open('/home/jiaxi/workspace/KITTI/training/colored_0/000000_10.png','r')
img=img.crop((-10,-10,720,1260))
w,h=img.size
print(w,h)
