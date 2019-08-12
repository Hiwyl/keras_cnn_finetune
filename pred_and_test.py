# -*- coding: utf-8 -*-
'''
@Author :      lance
@Email :   wangyl306@163.com
 '''
from sklearn import metrics
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import keras
import keras.backend as K



#%%加载模型
#model_cx=load_model("myweights/model_cx_bin_255_224x224_0.7.h5")
#model_x23=load_model("myweights/model_x23_bin_255_224x224_0.8.h5")

K.clear_session()
K.set_learning_phase(0) 
model=load_model("weights/resnet_0014.h5")

#%%单张展示
#区分组别 binary
test_gen=ImageDataGenerator(rescale=1/255).flow_from_directory("ceshi",
                           target_size=(224,224),
                           class_mode="binary",
                           batch_size=1,
                           shuffle=False)
pred = model_cx.predict_generator(test_gen, steps=110, verbose=1)
pred=pred.ravel()
pred[pred<0.7]=0
pred[pred>=0.7]=1
print("组别(cx)是:", pred)  
#准确率
n=0
for i in pred:
    if i==1:  #类别标签
        n+=1
print(n/110)  
#%%单张展示
#区分组别 binary
test_gen=ImageDataGenerator(rescale=1/255).flow_from_directory("ceshi",
                           target_size=(224,224),
                           class_mode="binary",
                           batch_size=1,
                           shuffle=False)
pred = model_x23.predict_generator(test_gen, steps=10, verbose=1)
pred=pred.ravel()
pred[pred<0.8]=0
pred[pred>=0.8]=1
print("级别是(x23):", pred) 
#准确率
n=0
for i in pred:
    if i==1:  #类别标签
        n+=1
print(n/110)  

#%%4分级
def salt(img, n=10000):
#    循环添加n个椒盐
#    for k in range(n):
#       # 随机选择椒盐的坐标
#       i = int(np.random.random() * img.shape[1])
#       j = int(np.random.random() * img.shape[0])
#       # 如果是灰度图
#       if img.ndim == 2:
#           img[j,i] = 255
#       # 如果是RBG图片
#       elif img.ndim == 3:
#           img[j,i,0]= 255
#           img[j,i,1]= 255
#           img[j,i,2]= 255
    noise = np.random.rand(448,448, 3)*0.05-0.025  
    img = img + noise
    return img
test_gen=ImageDataGenerator(rescale=1/255,preprocessing_function=None).flow_from_directory("ceshi",
                           target_size=(448,448),
                           batch_size=1,
                           shuffle=False)

pred= model.predict_generator(test_gen, steps=120, verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)
print("组别是(softmax):", predicted_class_indices)

#准确率
n=0
for i in predicted_class_indices:
    if i==1:  #类别标签
        n+=1
print(n/120)


#plot：×255后显示
import matplotlib.pyplot as plt
def plots(ims,figsize=(10,5),rows=1,interp=False,titles=None):
      if type(ims[0]) is np.ndarray:
          ims=np.array(ims).astype(np.uint8)
          if (ims.shape[-1] != 3):
              ims=ims.transpose((0,2,3,1))
      f=plt.figure(figsize=figsize)
      cols=len(ims)//rows if len(ims)%2 ==0 else len(ims)//rows+1
      for i in range(len(ims)):
          sp=f.add_subplot(rows,cols,i+1)
          sp.axis('off')
          if titles is not None:
              sp.set_title(titles[i],fontsize=9)
          plt.imshow(ims[i],interpolation=None if interp else "none")



imgs,labels=next(test_gen)
plots(imgs)
plt.show()


#%%读图测试
#skimage预测测试
import  os
from skimage import io,transform
path="新建文件夹/t1_c41"
files=os.listdir(path)
Tmp_Img=[]
for i in range(len(files)):
     print(i)
     tmp=io.imread(path+'/'+files[i])
     tmp_img=transform.resize(tmp,[448,448])
     Tmp_Img.append(tmp_img)
Tmp_Img=np.array(Tmp_Img)
pred=model.predict(Tmp_Img)
pred=np.argmax(pred, axis=1)
print(pred)
#准确率
n=0
for i in pred:
    if i==1:  #类别标签
        n+=1
print(n/120)



#画图
io.imshow(tmp_img)
tmp_img=tmp_img*255

#keras 同ImageDataGenerator
import os
from keras.preprocessing import image
path="ceshi/t1_x31"
file_names = os.listdir(path)
i=0
for file_name in file_names:
     img_path=os.path.join(path, file_name)
     img = image.load_img(img_path, target_size=(448,448))
     x = image.img_to_array(img)
     x = x*(1/255)
     x = np.expand_dims(x, axis=0)
     pred = model.predict(x)
     predicted_class_indices=np.argmax(pred, axis=1)
     print(predicted_class_indices)
     if predicted_class_indices ==3:
         i+=1
print(i/110)

plots(x)
plt.show()
 

#%%加载数据
# train_gen=ImageDataGenerator(1/255).flow_from_directory("re_cx/train",
#                             target_size=(224,224),
#
#                             class_mode="binary",
#                             batch_size=10,
#                             shuffle=False)
#
#
# valid_gen=ImageDataGenerator(1/255).flow_from_directory("re_cx/valid",
#                             target_size=(224,224),
#
#                             class_mode="binary",
#                             batch_size=10,
#                             shuffle=False)

#test_gen=ImageDataGenerator(rescale=1/255).flow_from_directory("ceshi",
#                           target_size=(448,448),
#                           class_mode="binary",
#                           batch_size=50,
#                           shuffle=False)


#%%测试
steps=6
#test_class=np.array([])

#for i in range(steps):
#    test_imgs, test_lables = next(test_gen)
#    test_class=np.hstack((test_class,test_lables ))
#print("真实类别：",test_class)

pred = model_cx.predict_generator(test_gen, steps=steps, verbose=1)
pred=pred.ravel()
pred=list(pred)
for i in range(len(pred)):
    if pred[i]<0.7:
        pred[i]=0
    else:
        pred[i]=1
print("预测结果:", pred)


# 打印混淆矩阵
#cm = metrics.confusion_matrix(test_class, pred)
#
#
#print(cm)



#%%特征模型
model_feather = keras.Model(inputs=model_cx.input,
                                     outputs=model_cx.layers[-2].output)
model_feather.summary()


#%%特征提取
# Labels=[]
# Predicts=[]
#
# for i in range(63):#111
#     print(i)
#     temp=next(train_gen)
#     temp1=temp[0]
#
#     Labels.append(temp[1])
#     Predicts.append(model_feather.predict(temp1))
#
# train_features=np.array(Predicts).reshape([630,1024]) #[1110,1024]
# train_labels=np.array(Labels).reshape([630,1]) #[1110,1]
#
# Labels=[]
# Predicts=[]
#
# for i in range(27):#47
#     print(i)
#     temp=next(valid_gen)
#     temp1=temp[0]
#
#     Labels.append(temp[1])
#     Predicts.append(model_feather.predict(temp1))
#
# valid_features=np.array(Predicts).reshape([270,1024]) #[470,1024]
# valid_labels=np.array(Labels).reshape([270,1]) #[470,1]
#
# Labels=[]
# Predicts=[]
#
# for i in range(3):
#     print(i)
#     temp=next(test_gen)
#     temp1=temp[0]
#
#     Labels.append(temp[1])
#     Predicts.append(model_feather.predict(temp1))
#
# pred_features=np.array(Predicts).reshape([30,1024])
# real_labels=np.array(Labels).reshape([30,1])



#%%误差图
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
valid_datagen = ImageDataGenerator(rescale=1./255)
#valid_datagen = ImageDataGenerator()
valid_gen = valid_datagen.flow_from_directory( 'ceshi/c1',
                                                    target_size=(224,224),batch_size=10,class_mode='binary')


from keras.preprocessing.image import ImageDataGenerator
valid_datagen = ImageDataGenerator()
#valid_datagen = ImageDataGenerator()
valid_gen2 = valid_datagen.flow_from_directory( 'ceshi/c1',
                                                    target_size=(224,224),batch_size=10,class_mode='binary')



temp1=next(valid_gen)
image1=temp1[0]

temp2=next(valid_gen2)
image2=temp2[0]

diff=image1*255-image2

print(np.mean(diff),np.max(diff),np.min(diff))

plt.imshow(diff[1])
plt.show()





#配置：因特尔E5系列金牌处理器、两块总共44核88线程的CPU、四块2080Ti的显卡 - 10万





