# -*- encoding: utf-8 -*-
'''
@Author :      lance  
@Email :   wangyl306@163.com
 '''

from skimage import io,transform,exposure
import numpy as np
import os

imgpath="./data/test/C3F/C3F_blockId#32756.bmp"


img=io.imread(imgpath)
io.imshow(img)

print(type(img))  #显示类型
print(img.shape)  #显示尺寸 #height/weight/channel
print(img.size)   #显示总像素个数
print(img.max())  #最大像素值
print(img.min())  #最小像素值
print(img.mean()) #像素平均值

#保存图片
#io.imsave('d:/cat.jpg',img)

#随机生成5000个椒盐
rows,cols,dims=img.shape
for i in range(500000):
    x=np.random.randint(0,rows)
    y=np.random.randint(0,cols)
    img[x,y,:]=255
io.imshow(img)

#二值化
#颜色空间转换  skimage.color.convert_colorspace(arr, fromspace, tospace)
from skimage import color
img_gray=color.rgb2gray(img)
rows,cols=img_gray.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray[i,j]<=0.5):
            img_gray[i,j]=0
        else:
            img_gray[i,j]=1
io.imshow(img_gray)
print(img_gray[img_gray>0].size) #白色区域的像素个数 - 面积


index=np.where(img_gray==1)
h1=index[0].min()
h2=index[0].max()
w1=index[1].min()
w2=index[1].max()

roi=img[h1:h2,w1:w2]
io.imshow(roi)





#对于r通道大于170的像素值进行过滤后显示G通道为255
reddish = img[:, :, 0] >170
img[reddish] = [0, 255, 0]
io.imshow(img)

#分类着色
gray=color.rgb2gray(img)
rows,cols=gray.shape
labels=np.zeros([rows,cols])
for i in range(rows):
    for j in range(cols):
        if(gray[i,j]<0.4):
            labels[i,j]=0
        elif(gray[i,j]<0.75):
            labels[i,j]=1
        else:
            labels[i,j]=2
dst=color.label2rgb(labels)
io.imshow(dst)

#批处理
str= './data/test/C3F/*.bmp'
def convert_gray(f):
        rgb=io.imread(f) #依次读取rgb图片 
        gray=color.rgb2gray(rgb) #将rgb图片转换成灰度图 
        dst=transform.resize(gray,(256,256)) #将灰度图片大小转换为256*256 
        return dst 
coll = io.ImageCollection(str,load_func=convert_gray)
path= "./data/测试/"
if not os.path.exists(path):
    os.makedirs(path)
for i in range(len(coll)): 
    io.imsave(path+np.str(i)+'.jpg',coll[i]) #循环保存图片
    
    
#图像滤波
from skimage import filters
import matplotlib.pyplot as plt
#转换为灰度空间
img=color.rgb2gray(img)
#edges = filters.sobel(img)    #边缘检测
#edges = filters.roberts(img)  #边缘检测
edges = filters.gaussian(img,sigma=5)   #消除噪音  sigma越小越清晰
plt.imshow(edges,cmap="gray")
plt.show()

#图像自动阈值分割
from skimage import filters
import matplotlib.pyplot as plt
img=color.rgb2gray(img)
#thresh = filters.threshold_otsu(img) #返回一个阈值
thresh = filters.threshold_li(img)
dst =(img <= thresh)*1.0 #根据阈值进行分割
plt.figure('thresh',figsize=(8,8))
plt.subplot(121)
plt.title('original image')
plt.imshow(img,plt.cm.gray)
plt.subplot(122)
plt.title('binary image')
plt.imshow(dst,plt.cm.gray)
plt.show()


#阈值分割
from skimage import filters
import matplotlib.pyplot as plt
image =color.rgb2gray(img)
dst =filters.threshold_local(image, 15) #返回一个阈值图像
plt.figure('thresh',figsize=(8,8))
plt.subplot(121)
plt.title('original image')
plt.imshow(image,plt.cm.gray)
plt.subplot(122)
plt.title('binary image')
plt.imshow(dst,plt.cm.gray)
plt.show()


#膨胀  用来扩充边缘或填充小的孔洞。
import skimage.morphology as sm
import matplotlib.pyplot as plt
img=color.rgb2gray(img)
dst1=sm.dilation(img,sm.square(5)) #用边长为5的正方形滤波器进行膨胀滤波
dst2=sm.dilation(img,sm.square(15)) #用边长为15的正方形滤波器进行膨胀滤波
plt.figure('morphology',figsize=(8,8))
plt.subplot(131)
plt.title('origin image')
plt.imshow(img,plt.cm.gray)
plt.subplot(132)
plt.title('morphological image')
plt.imshow(dst1,plt.cm.gray)
plt.subplot(133)
plt.title('morphological image')
plt.imshow(dst2,plt.cm.gray)


#腐蚀  可用来提取骨干信息，去掉毛刺，去掉孤立的像素。
import skimage.morphology as sm
import matplotlib.pyplot as plt
img=img=color.rgb2gray(img)
dst1=sm.erosion(img,sm.square(5)) #用边长为5的正方形滤波器进行膨胀滤波
dst2=sm.erosion(img,sm.square(25)) #用边长为25的正方形滤波器进行膨胀滤波
plt.figure('morphology',figsize=(8,8))
plt.subplot(131)
plt.title('origin image')
plt.imshow(img,plt.cm.gray)
plt.subplot(132)
plt.title('morphological image')
plt.imshow(dst1,plt.cm.gray)
plt.subplot(133)
plt.title('morphological image')
plt.imshow(dst2,plt.cm.gray)




#开运算  先腐蚀再膨胀  消除小物体
from skimage import io,color
import skimage.morphology as sm
import matplotlib.pyplot as plt
img=color.rgb2gray(img)
dst=sm.opening(img,sm.disk(9)) #用边长为9的圆形滤波器进行膨胀滤波
plt.figure('morphology',figsize=(8,8))
plt.subplot(121)
plt.title('origin image')
plt.imshow(img)
plt.axis('off')
plt.subplot(122)
plt.title('morphological image')
plt.imshow(dst,plt.cm.gray)
plt.axis('off')












#闭运算 先膨胀再腐蚀 填充空洞
from skimage import io,color
import skimage.morphology as sm
import matplotlib.pyplot as plt
img=color.rgb2gray(img)
dst=sm.closing(img,sm.disk(9)) #用边长为5的圆形滤波器进行膨胀滤波
plt.figure('morphology',figsize=(8,8))
plt.subplot(121)
plt.title('origin image')
plt.imshow(img,plt.cm.gray)
plt.axis('off')
plt.subplot(122)
plt.title('morphological image')
plt.imshow(dst,plt.cm.gray)
plt.axis('off')









#%%
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.morphology import extrema
from scipy import linalg
import numpy as np
from skimage import io
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage import exposure
from skimage import color
from scipy import ndimage as ndi
from skimage.morphology import label, binary_erosion, binary_dilation, square, binary_opening, binary_closing 
import operator
from itertools import repeat


fn='E:\Python\ANN\Projects\固高\定位\Image1.jpg'
img= io.imread(fn)

# using functions in Skimage #
# note the input image is assumed to be sRGB, and will be normalized to [0,1] range #
img_xyz= color.rgb2xyz(img)
img_lab= color.xyz2lab(img_xyz)
#plt.imshow(img_lab, cmap="gray")
#plt.show()

# using VN boy's coversion method #
# assume the input is CIE RGB #
def rgb2xyz_vn(img):
    img= img.copy()/255.
    conv_mat = np.array([[ 0.4900,  0.3100,  0.2000],
                       [ 0.1769,  0.8124,  0.0107],
                       [ 0.0000,  0.0099,  0.9901]])
    xyz = np.dot(img, conv_mat.T)
    return xyz

img_xyz_vn= rgb2xyz_vn(img) 

# convert the xyz to lab:  
# according to http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
# the reference white for CIE RGB is E     
img_lab_vn= color.xyz2lab(img_xyz_vn, illuminant= "E")

gray= img_lab_vn[:, :, 0]

#gray = rgb2gray(img)*255
#gray = gray.astype(int)

gray = rgb2gray(img)
#gray = exposure.equalize_hist(gray)

plt.imshow(gray, cmap="gray")
plt.show()

############################################################################
########### Ostu thresholding #############  
import numpy as np
from skimage import img_as_float, filters, exposure    
from skimage.filters import gaussian, laplace
from skimage.exposure import rescale_intensity

from skimage import io
import cv2

img_h= gray

blur= gaussian(img_h, sigma= 1)

laplacian = laplace(blur)
theCutoff= np.percentile(abs(laplacian).flatten(), 80)
laplacian_index= abs(laplacian)> theCutoff
strong_pixels= blur[laplacian_index]

blur[laplacian_index]= 255
blur[~laplacian_index]= 0

laplacian_list= laplacian.flatten()


## Use skimage to perform otsu thresholding ##
val = filters.threshold_otsu(strong_pixels)


plt.hist(strong_pixels[strong_pixels<val], bins=50)
plt.show()

hist, bins_center = exposure.histogram(strong_pixels)
plt.imshow(img_h>val, cmap='gray', interpolation='nearest')
#plt.axis('off')
plt.show()

###########

image= img_h>val
image= ndi.morphology.binary_fill_holes(image)
image= binary_opening(image, selem=square(3))
image= binary_closing(image, selem=square(3))
image= binary_erosion(image, selem= square(5))
image= binary_dilation(image, selem= square(5))

plt.imshow(image, cmap='gray', interpolation='nearest')
#plt.axis('off')
plt.show()



theConnected, theConnected_num= label(image, return_num=True)
theConnected_ind= np.unique(theConnected).tolist()
del theConnected_ind[theConnected_ind.index(0)]


theConnected_num_P= []
for i in np.unique(theConnected_ind):    
    the_Ps= image[theConnected== i] 
    theConnected_num_P.append(len(the_Ps))

# Order the connected components by the number of pixels #
new_theConnected_num_P= np.array(theConnected_num_P).copy().tolist()
new_theConnected_num_P.sort(reverse=True)

cand_max= np.where(np.array(theConnected_num_P)== new_theConnected_num_P[0])[0]
cand_2= np.where(np.array(theConnected_num_P)== new_theConnected_num_P[1])[0]
cand_3= np.where(np.array(theConnected_num_P)== new_theConnected_num_P[2])[0]


newImg= np.zeros(image.shape, dtype= "uint8")
newImg[theConnected==theConnected_ind[cand_max[0]]]=1 
newImg[theConnected==theConnected_ind[cand_2[0]]]=2 
newImg[theConnected==theConnected_ind[cand_3[0]]]=3 


plt.imshow(newImg, cmap='gray', interpolation='nearest')
plt.show()


fig01= plt.imshow(newImg, cmap='gray', interpolation='nearest')
plt.xticks([]), plt.yticks([])
plt.tight_layout()
fig01.figure.savefig(samLoc + samName + 'thresholded.png', dpi=200)




############################
### Boundary tracing ###

# obtain the coordinates of each center #
cen_max= (int(np.median(np.where(newImg==1)[0])), int(np.median(np.where(newImg==1)[1])))
cen_2= (int(np.median(np.where(newImg==2)[0])), int(np.median(np.where(newImg==2)[1])))
cen_3= (int(np.median(np.where(newImg==3)[0])), int(np.median(np.where(newImg==3)[1])))
cen_max2= (cen_max[0], cen_2[1]) 
cen_max3= (cen_max[0], cen_3[1])  
 

all_bounds= []
for sz in [1,2,3]:
    
    oneLumen= newImg==sz
    eroded= binary_erosion(oneLumen, selem= square(3))
    boundary_01= oneLumen ^ eroded

    boundary= np.zeros(tuple(map(operator.add, boundary_01.shape, (2,2))), dtype= "uint8")
    boundary[1:(boundary.shape[0]-1), 1:(boundary.shape[1]-1)]= boundary_01
    boundary= boundary>0

    plt.imshow(boundary)
    plt.show()

    ### Sample lumen boundary pixels with roughly equal intervals ###
    # 1) trace the boundary in clockwise direction #
    theX, theY= np.where(boundary)
    totalNum= range(len(theX))
    n_s= [(0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1)]

    k= 0
    bound_pixels= [(theX[0], theY[0])]
    for i in totalNum:    
        if i>0:
            if bound_pixels[i]== bound_pixels[0]:
                break    
        if k != 0:
            ind= range(k, 8)+ range(k)
            tmp_n_s= [n_s[x] for x in ind]
        else:
            tmp_n_s= n_s
        
        for j in range(8):
            back_xy= tuple(np.add(bound_pixels[i], tmp_n_s[j]))
        
            if boundary[back_xy]:
                bound_pixels.append(back_xy)
                back_xy_last= tuple(np.add(bound_pixels[i], tmp_n_s[j-1]))
                #k= n_s.index(tmp_n_s[j-1])
                k= n_s.index(tuple(np.subtract(back_xy_last, back_xy)))
                break            
    bound_pixels= bound_pixels[:-1]
    bound_pixels= [(a[0]-1, a[1]-1) for a in bound_pixels]

    # 2) sample the boundary pixels at equal intervals #
    interval= 1
    num_elements= len(bound_pixels)/ interval
    if len(bound_pixels) % interval < 5:
        num_elements= num_elements
    else:
        num_elements= num_elements+1
    bp_index= [interval* x for x in range(num_elements)]

    lbp= [bound_pixels[x] for x in bp_index]

    all_bounds.append(lbp)


# visually test the result #
for szh in range(len(all_bounds)):
    lbp= all_bounds[szh]
                
    frame= np.zeros(boundary.shape, dtype= "uint8")
    for i in range(len(lbp)):
        frame[lbp[i][0], lbp[i][1]]=1
            
    fig01= plt.imshow(frame, cmap='gray')
    plt.show()
             


# Find the closest boundary pixels to the bar center #
theData= all_bounds[1]
theCenter= cen_max2  
theDis= []
for k in range(len(theData)):
    lu_x= theData[k][0]
    lu_y= theData[k][1]

    dis_01= np.round(np.sqrt((lu_x- theCenter[0])**2 + (lu_y- theCenter[1])**2), 2)
    theDis.append(dis_01)

theCand= np.where(np.array(theDis)== np.min(theDis))[0]
tip_1= theData[theCand[0]]


theData= all_bounds[2]
theCenter= cen_max3  
theDis= []
for k in range(len(theData)):
    lu_x= theData[k][0]
    lu_y= theData[k][1]

    dis_01= np.round(np.sqrt((lu_x- theCenter[0])**2 + (lu_y- theCenter[1])**2), 2)
    theDis.append(dis_01)

theCand= np.where(np.array(theDis)== np.min(theDis))[0]
tip_2= theData[theCand[0]]


# Place the tip point coordinates into the original figure # 
tip_1_x= tip_1[0]+ range(-30, 30)     
tip_1_y= tip_1[1]+ range(-30, 30) 

tip_2_x= tip_2[0]+ range(-30, 30)   
tip_2_y= tip_2[1]+ range(-30, 30)   

#frame= np.zeros(gray.shape, dtype= "uint8")
#frame[tip_1_x, np.array([tip_1[1]]*len(tip_1_x))]=1
#frame[np.array([tip_1[0]]*len(tip_1_y)), tip_1_y]=1
#frame[tip_2_x, np.array([tip_2[1]]*len(tip_2_x))]= 1
#frame[[tip_2[0]]*len(tip_2_y), tip_2_y.tolist()]= 1

#fig01= plt.imshow(frame, cmap='gray', interpolation='nearest')
#plt.xticks([]), plt.yticks([])
#plt.tight_layout()
#fig01.figure.savefig(samLoc + samName + 'barTipAdded.png', dpi=200)
   
img[tip_1_x, np.array([tip_1[1]]*len(tip_1_x))]= [0, 0, 255]
img[np.array([tip_1[0]]*len(tip_1_y)), tip_1_y]= [0, 0, 255]

img[tip_2_x, np.array([tip_2[1]]*len(tip_2_x))]= [255, 0, 0]
img[np.array([tip_2[0]]*len(tip_2_y)), tip_2_y]= [255, 0, 0]


fig01= plt.imshow(img)
#plt.xticks([]), plt.yticks([])
plt.tight_layout()
fig01.figure.savefig(samLoc + samName + 'barTipAdded.png', dpi=200)











