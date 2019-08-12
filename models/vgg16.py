# -*- encoding: utf-8 -*-
'''
@Author :      lance  
@Email :   wangyl306@163.com
 '''
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from model.load_data import load_data

# 固定初始参数，确保结果复现
import random as rn
import tensorflow as tf
from keras import backend as K, Model
import os
import numpy as np



def vgg16(classes,epochs,steps_per_epoch,validation_steps,input_shape):
    # 加载数据
    train_batches, valid_batches= load_data(input_shape)

    input_shape += (3,)

    model_vgg = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    for layer in model_vgg.layers:
        layer.trainable = False  # 别去调整之前的卷积层的参数
    model = Flatten(name='flatten')(model_vgg.output)  # 去掉全连接层，前面都是卷积层
    model = Dense(256, activation='relu', name='fc1')(model)
    model = Dense(256, activation='relu', name='fc2')(model)
    model = Dropout(0.5)(model)
    # model = Dropout(0.6)(model)
    if classes==1:
        print("二元分类")
        model = Dense(classes, activation='sigmoid')(model)  # model就是最后的y
        model = Model(inputs=model_vgg.input, outputs=model, name='vgg16')
        ada = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='binary_crossentropy', optimizer=ada, metrics=['accuracy'])
    else:
        print("多分类")
        model = Dense(classes, activation='softmax')(model)  # model就是最后的y
        model= Model(inputs=model_vgg.input, outputs=model, name='vgg16')
        ada = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['accuracy'])

    #保存模型
    out_dir = "../weights/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filepath ="../weights/vgg16_{epoch:04d}.h5"
    # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False,
                                 mode='max')
    #学习率调整
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1,
                                  min_lr=0.00000001, mode="min")
    # 早停
    earlystopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')
    #保存训练过程
    log_dir = "../logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logfile="../logs/vgg16.csv"
    log=keras.callbacks.CSVLogger(logfile, separator=',', append=False)
    loggraph=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    callbacks_list = [checkpoint,lr_reduce,log, earlystopping]
    # 训练
    model.fit_generator(train_batches, steps_per_epoch=steps_per_epoch, validation_data=valid_batches,
                        validation_steps=validation_steps, epochs=epochs, verbose=2,
                        callbacks=callbacks_list,workers=16,max_queue_size=20)

if  __name__=="__main__":
    vgg16(1,200,210,15,(224,224))







