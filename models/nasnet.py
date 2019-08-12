# -*- encoding: utf-8 -*-
'''
@Author :      lance
@Email :   wangyl306@163.com
 '''

import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Flatten, Dense, regularizers, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
import numpy as np
from load_data import load_data
# 固定随机种子
import random as rn
import tensorflow as tf
from keras import backend as K, Model, Input
import os



def nasnet(classes,epochs, steps_per_epoch, validation_steps,input_shape):
    # 加载数据
    train_batches, valid_batches= load_data(input_shape)

    input_shape += (3,)

    temp_model = keras.applications.nasnet.NASNetMobile(include_top=False,input_shape=input_shape)
    x = temp_model.output
    x = GlobalAveragePooling2D()(x)  # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
    x = Dropout(0.5)(x)

    if classes==1:
        print("二元分类")
        outputs = Dense(classes, activation='sigmoid')(x)
        model = Model(temp_model.inputs, outputs)
        model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    else:
        print("多分类")
        outputs = Dense(classes, activation='softmax')(x)
        model = Model(temp_model.inputs, outputs)
        model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # model_cx.summary()

    print('the number of layers in this model_cx:' + str(len(model.layers)))
    # 保存模型
    out_dir = "weights/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filepath = "weights/nasnet_{epoch:04d}.h5"
    # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True,
                                 mode='max',period=2)
    # 学习率调整
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1,
                                  min_lr=0.000005, mode="min")
    # 早停
    earlystopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')
    # 保存训练过程
    log_dir = "logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logfile = "logs/nasnet.csv"
    log = keras.callbacks.CSVLogger(logfile, separator=',', append=False)
    loggraph = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    callbacks_list = [checkpoint, lr_reduce, log]

    # 训练
    model.fit_generator(train_batches, steps_per_epoch=steps_per_epoch, validation_data=valid_batches,
                        validation_steps=validation_steps, epochs=epochs, verbose=2,
                        callbacks=callbacks_list,workers=16,max_queue_size=20)

if __name__ == "__main__":
    nasnet(1,50,20,5,(224,224))