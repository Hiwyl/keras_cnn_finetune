# -*- encoding: utf-8 -*-
'''
@Author :      lance
@Email :   wangyl306@163.com
 '''

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam
import keras
from model.load_data import load_data
from model.octconv_config.models import *
import pickle, os, time



def octconv(alpha,classes,epochs, steps_per_epoch, validation_steps,input_shape):
    # 加载数据
    train_batches, valid_batches= load_data(input_shape)

    input_shape += (3,)


    tf.logging.set_verbosity(tf.logging.ERROR)

    if alpha <= 0:
        model = create_normal_wide_resnet(classes,input_shape)
    else:
        model = create_octconv_wide_resnet(alpha,classes,input_shape)
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print('the number of layers in this model:' + str(len(model.layers)))

    out_dir = "../weights/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filepath = "../weights/octconv.h5"
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
    logfile = "../logs/octconv.csv"
    log = keras.callbacks.CSVLogger(logfile, separator=',', append=False)
    loggraph = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    callbacks_list = [checkpoint, lr_reduce, log]
    # 训练
    model.fit_generator(train_batches, steps_per_epoch=steps_per_epoch, validation_data=valid_batches,
                        validation_steps=validation_steps, epochs=epochs, verbose=2,
                        callbacks=callbacks_list,workers=16,max_queue_size=20)


if __name__ == "__main__":
    octconv(0.2,3,200,16,5,(56,56))
#alpha需要自己确定
