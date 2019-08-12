# -*- encoding: utf-8 -*-
'''
@File    :   load_data
@Author :      lance
@Email :   wangyl306@163.com
 '''
from keras_preprocessing.image import ImageDataGenerator


def load_data(input_shape):
    train_path='../cx/train'
    valid_path='../cx/test'

    train_generator = ImageDataGenerator(rescale=1 / 255,
                                       rotation_range=360,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       brightness_range=[0.95, 1.05],
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='nearest')
    valid_generator = ImageDataGenerator(rescale=1 / 255,    rotation_range=360,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       brightness_range=[0.95, 1.05])


    train_batches=train_generator.flow_from_directory(train_path,
                                                      target_size=input_shape,
                                                      classes=["C2F", "X2F"],
                                                      class_mode='binary',
                                                      batch_size=10)



    valid_batches=valid_generator.flow_from_directory(valid_path,
                                                    target_size=input_shape,
                                                    classes=["C2F", "X2F"],
                                                    class_mode='binary',
                                                    batch_size=10
                                                  )
    print(train_batches.class_indices)
    print(valid_batches.class_indices)
    return train_batches,valid_batches
if __name__=="__main__":
    load_data((224,224))


# from keras_preprocessing.image import ImageDataGenerator
# import numpy as np
# import matplotlib.pyplot as plt
#
# train_path='../cx/train'
# valid_path='../cx/test'
#
# train_batches=ImageDataGenerator(rotation_range=360,
#                                        width_shift_range=0.15,
#                                        height_shift_range=0.15,
#                                        brightness_range=[0.95, 1.05],
#                                        shear_range=0.2,
#                                        zoom_range=0.2,
#                                        horizontal_flip=True,
#                                         vertical_flip=True,
#                                        fill_mode='nearest').flow_from_directory(train_path,target_size=(2000,2000),classes=["C2F","X2F"],class_mode = 'binary',batch_size=2)
# valid_batches=ImageDataGenerator(rescale=1/255).flow_from_directory(valid_path,target_size=(224,224), classes=["C4F","X2F"],class_mode = 'binary',batch_size=16)
#
#
# def plots(ims,figsize=(20,10),rows=1,interp=False,titles=None):
#     if type(ims[0]) is np.ndarray:
#         ims=np.array(ims).astype(np.uint8)
#         if (ims.shape[-1] != 3):
#             ims=ims.transpose((0,2,3,1))
#     f=plt.figure(figsize=figsize)
#     cols=len(ims)//rows if len(ims)%2 ==0 else len(ims)//rows+1
#     for i in range(len(ims)):
#         sp=f.add_subplot(rows,cols,i+1)
#         sp.axis('off')
#         if titles is not None:
#             sp.set_title(titles[i],fontsize=9)
#         plt.imshow(ims[i],interpolation=None if interp else "none")
# if __name__=='__main__':
#     imgs,labels=next(train_batches)
#     plots(imgs,titles=labels)
#     plt.show()