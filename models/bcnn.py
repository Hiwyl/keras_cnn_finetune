# -*- encoding: utf-8 -*-
'''
@Author :      lance  
@Email :   wangyl306@163.com
 '''

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import  Flatten, Dense, Dropout, Input, Reshape, Lambda
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
from model_cx.load_data import load_data
import keras

def sign_sqrt(x):
    return K.sign(x) * K.sqrt(K.abs(x) + 1e-10)


def l2_norm(x):
    return K.l2_normalize(x, axis=-1)


def batch_dot(cnn_ab):
    return K.batch_dot(cnn_ab[0], cnn_ab[1], axes=[1, 1])


def bilinearnet(classes,epochs, steps_per_epoch, validation_steps,input_shape):
    # 加载数据
    train_batches, valid_batches= load_data(input_shape)

    input_shape += (3,)

    input_tensor = Input(shape=input_shape)
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
    model_vgg16 = Model(inputs=input_tensor, outputs=vgg16.output)
    model_resnet50 = Model(inputs=input_tensor, outputs=resnet50.output)
    model_vgg16.compile(loss='categorical_crossentropy', optimizer='adam')
    model_resnet50.compile(loss='categorical_crossentropy', optimizer='adam')

    resnet50_x = Reshape([model_resnet50.layers[-6].output_shape[1] * model_resnet50.layers[-6].output_shape[2],
                          model_resnet50.layers[-6].output_shape[3]])(model_resnet50.layers[-6].output)
    vgg16_x = Reshape([model_vgg16.layers[-1].output_shape[1] * model_vgg16.layers[-1].output_shape[2],
                       model_vgg16.layers[-1].output_shape[3]])(model_vgg16.layers[-1].output)

    cnn_dot_out = Lambda(batch_dot)([vgg16_x, resnet50_x])

    sign_sqrt_out = Lambda(sign_sqrt)(cnn_dot_out)
    l2_norm_out = Lambda(l2_norm)(sign_sqrt_out)
    flatten = Flatten()(l2_norm_out)
    dropout = Dropout(0.5)(flatten)
    if classes==1:
        print("执行output = Dense(classes, activation='sigmoid')(dropout)")
        output = Dense(classes, activation='sigmoid')(dropout)
    else:
        print("执行output = Dense(classes, activation='softmax')(dropout)")
        output = Dense(classes, activation='softmax')(dropout)

    model = Model(input_tensor, output)
    if classes==1:
        print("binary")
        model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-3, momentum=0.9, decay=1e-6),
                  metrics=['accuracy'])
    else:
        print("categotical")
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-3, momentum=0.9, decay=1e-6),
                  metrics=['accuracy'])


    # 保存模型
    out_dir = "../weights/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filepath = "../weights/bcnn_{epoch:04d}.h5"
    # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False,
                                 mode='max')
    # 学习率调整
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1,
                                  min_lr=0.000005, mode="min")
    #早停
    earlystopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    # 保存训练过程
    log_dir = "../logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logfile = "../logs/bcnn.csv"
    log = keras.callbacks.CSVLogger(logfile, separator=',', append=False)
    callbacks_list = [checkpoint, lr_reduce, log]

    # 训练
    model.fit_generator(train_batches, steps_per_epoch=steps_per_epoch, validation_data=valid_batches,
                        validation_steps=validation_steps, epochs=epochs, verbose=2,
                        callbacks=callbacks_list,workers=16,max_queue_size=20)


if  __name__=="__main__":
    bilinearnet(1,100,130,32,(224,224))