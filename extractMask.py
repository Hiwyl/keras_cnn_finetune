# -*- encoding: utf-8 -*-
'''
@Author :      lance  
@Email :   wangyl306@163.com
 '''

from skimage import io
import numpy as np
import os
from skimage import color
import skimage.morphology as sm

#imgpath="./data/test/C3F/C3F_blockId#32756.bmp"
#
#
#img=io.imread(imgpath)
#io.imshow(img)

#print(type(img))  #显示类型
#print(img.shape)  #显示尺寸 #height/weight/channel
#print(img.size)   #显示总像素个数
#print(img.max())  #最大像素值
#print(img.min())  #最小像素值
#print(img.mean()) #像素平均值
 
    
str= './新建文件夹/tt/*.bmp'  #源路径
path= "./新建文件夹/tt1/"     #保存路径
if not os.path.exists(path):
    os.makedirs(path)
#先进行开运算去除小物体，再进行二值化取mask
nb=0
def mask(f):
    global nb
    nb+=1
    print(nb)
    image=io.imread(f)
    #开运算
    img_gray=color.rgb2gray(image)
    dst=sm.opening(img_gray,sm.disk(9))
    img=color.gray2rgb(dst) 
    #二值化
    img_gray=color.rgb2gray(img)
    rows,cols=img_gray.shape
    for i in range(rows):
        for j in range(cols):
            if (img_gray[i,j]<=0.5):
                img_gray[i,j]=0
            else:
                img_gray[i,j]=1
    #计算叶面大小-白色区域得大小
    area=img_gray[img_gray>0].size
    #在原图中找出mask区域
    index=np.where(img_gray==1)
    h1=index[0].min()
    h2=index[0].max()
    w1=index[1].min()
    w2=index[1].max()   
    roi=image[h1:h2,w1:w2]
    return roi,area

coll = io.ImageCollection(str,load_func=mask)
areas=[]
for n in range(len(coll)): 
    io.imsave(path+np.str(n)+'.bmp',coll[n][0]) #循环保存图片 
    areas.append(coll[n][1])
print("min:",np.min(areas))
print("median:",np.median(areas))
print("max:",np.max(areas))  
print("mean:",np.mean(areas))   


# test                                      train
    
#min: 1782782                               1233141
#median: 2211339.0                          2076964
#max: 2922081                               3447462
#mean: 2276614                              2103386
    
#min: 1446975
#median: 2263485.0
#max: 3885086
#mean: 2388327.9730941704  

#min: 994173
#median: 1726611.0
#max: 3245794
#mean: 1837663.5177304964  

#min: 939299
#median: 1655289.0
#max: 2541519
#mean: 1651290.857142857
    
    

    
    
    
    
    
    
    
    
    