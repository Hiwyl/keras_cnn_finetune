# -*- encoding: utf-8 -*-
'''
@Author :      lance
@Email :   wangyl306@163.com
 '''

import keras
from keras import Input, Model
from keras.applications import  DenseNet121, Xception
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dropout, Dense, GlobalAveragePooling2D

import numpy as np
from keras.optimizers import Adam
from load_data import load_data
# 固定随机种子
import random as rn
import tensorflow as tf
from keras import backend as K
import os


def merge(classes,epochs,steps_per_epoch,validation_steps,input_shape):
    # 加载数据
    train_batches, valid_batches = load_data(input_shape)

    input_shape += (3,)

    input_layer = Input(shape=input_shape)
    dense = DenseNet121(include_top=False, pooling='avg', input_shape=input_shape)
    xception = Xception(include_top=False, pooling='avg', input_shape=input_shape)

    dense = dense(input_layer)
    xception = xception(input_layer)

    # 把top1_model和top2_model连接起来
    t = keras.layers.Concatenate(axis=1)([dense, xception])
    if classes==1:
        print("二元分类")
        top_model = Dense(units=classes, activation="sigmoid")(t)
        model = Model(inputs=input_layer, outputs=top_model)
        asdl = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss='binary_crossentropy', optimizer=asdl, metrics=['accuracy'])
    else:
        print("多分类")
        top_model = Dense(units=classes, activation="softmax")(t)
        model = Model(inputs=input_layer, outputs=top_model)
        asdl = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss='categorical_crossentropy', optimizer=asdl, metrics=['accuracy'])
    # 保存模型
    out_dir = "weights/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filepath = "weights/merge_{epoch:04d}.h5"
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
    logfile = "logs/merge.csv"
    log = keras.callbacks.CSVLogger(logfile, separator=',', append=False)
    loggraph = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    callbacks_list = [checkpoint, lr_reduce, log]

    # 训练
    model.fit_generator(train_batches, steps_per_epoch=steps_per_epoch, validation_data=valid_batches,
                        validation_steps=validation_steps, epochs=epochs, verbose=2,
                        callbacks=callbacks_list,workers=16,max_queue_size=20)


if __name__ == "__main__":
    merge(1, 50, 16,4, (224, 224))