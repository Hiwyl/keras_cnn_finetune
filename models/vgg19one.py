# -*- encoding: utf-8 -*-
'''
@Author :      lance
@Email :   wangyl306@163.com
 '''

import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Flatten, Dense,regularizers
from keras.optimizers import Adam
import numpy as np
from model.load_data import load_data
# 固定随机种子
import random as rn
import tensorflow as tf
from keras import backend as K
import os




def vgg19_lr(classes,epochs, steps_per_epoch, validation_steps,input_shape):
    # 加载数据
    train_batches, valid_batches= load_data(input_shape)

    input_shape += (3,)

    temp_model = keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    x = temp_model.layers[-1].output
    predictions = Flatten()(x)
    predictions = Dense(256, activation='sigmoid', kernel_regularizer = regularizers.l2(0.001))(predictions)
    if classes==1:
        print("二元分类")
        predictions = Dense(classes, activation='sigmoid')(predictions)
    else:
        print("多分类")
        predictions = Dense(classes, activation='softmax')(predictions)

    model = keras.Model(inputs=temp_model.inputs, outputs=predictions)
    # 训练最后三层
    for layer in model.layers[:-3]:
        layer.trainable = False
    asdl = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    if classes==1:
        print("二元分类")
        model.compile(loss='binary_crossentropy', optimizer=asdl, metrics=['accuracy'])
    else:
        print("多分类")
        model.compile(loss='categorical_crossentropy', optimizer=asdl, metrics=['accuracy'])

    # 保存模型
    out_dir = "../weights/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filepath = "../weights/vgg19one_{epoch:04d}.h5"
    # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True,
                                 mode='max')
    # 学习率调整
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1,
                                  min_lr=0.00000001, mode="min")
    # 早停
    earlystopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')
    # 保存训练过程
    log_dir = "../logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logfile = "../logs/vgg19one.csv"
    log = keras.callbacks.CSVLogger(logfile, separator=',', append=False)
    loggraph = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    callbacks_list = [checkpoint, lr_reduce, log,earlystopping]

    # 训练
    model.fit_generator(train_batches, steps_per_epoch=steps_per_epoch, validation_data=valid_batches,
                        validation_steps=validation_steps, epochs=epochs, verbose=2,
                        callbacks=callbacks_list,workers=16,max_queue_size=20)

if __name__ == "__main__":
    vgg19_lr(1,50, 14,3,(224,224))