# -*- encoding: utf-8 -*-
'''
@Author :      lance
@Email :   wangyl306@163.com
 '''
from __future__ import print_function
from __future__ import division

import keras
import os
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from model.load_data import load_data

from model.myresnext import MyResNext


def train_resnext(classes,epochs, steps_per_epoch, validation_steps,input_shape):
    # 加载数据
    train_batches, valid_batches= load_data(input_shape)

    input_shape += (3,)

    # img_rows, img_cols = 56,56
    # img_channels = 3
    #
    # img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
    depth = 29
    cardinality = 16#原始为8，效果不明显
    width = 16

    model = MyResNext(input_shape, depth=depth, cardinality=cardinality, width=width, weights=None, classes=classes)
    print("Model created")

    model.summary()
    print('the number of layers in this model:' + str(len(model.layers)))

    optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    print("Finished compiling")
    print("Building model...")

    # 保存模型
    out_dir = "../weights/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filepath = "../weights/resnext.h5"
    # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True,
                                 mode='max')
    # 学习率调整
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1,
                                  min_lr=0.00000001, mode="min")
    # 保存训练过程
    log_dir = "../logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logfile = "../logs/resnext.csv"
    log = keras.callbacks.CSVLogger(logfile, separator=',', append=False)
    loggraph = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    callbacks_list = [checkpoint, lr_reduce, log]

    # 训练
    model.fit_generator(train_batches, steps_per_epoch=steps_per_epoch, validation_data=valid_batches,
                        validation_steps=validation_steps, epochs=epochs, verbose=2,
                        callbacks=callbacks_list,workers=16,max_queue_size=20)


if __name__ == "__main__":
   train_resnext(3, 100,20,5,(56,56))

