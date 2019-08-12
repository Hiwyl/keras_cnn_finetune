# -*- encoding: utf-8 -*-
'''
@Author :      lance  
@Email :   wangyl306@163.com
 '''
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Flatten, Dense, regularizers, GlobalAveragePooling2D, Dropout
from keras.optimizers import SGD, Adagrad, Adam
from keras import backend as K, Model
import os
from model_cx.load_data import load_data


def densenet(classes,epochs,steps_per_epoch,validation_steps,input_shape):
    #加载数据
    train_batches,valid_batches=load_data(input_shape)

    input_shape+=(3,)
    #DenseNet121, DenseNet169, DenseNet201
    temp_model= keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
    x = temp_model.output
    x = GlobalAveragePooling2D()(x)  # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
    x = Dense(1024, activation='relu')(x)
    x=Dropout(0.2)(x)
    if classes==1:
        print("sigmoid")
        predictions = Dense(classes, activation='sigmoid')(x)
    else:
        print("softmax")
        predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=temp_model.input, outputs=predictions)



    if classes==1:
        print("二元分类")
        model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                      loss='binary_crossentropy', metrics=['accuracy'])
    else:
        print("多分类")
        model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='categorical_crossentropy', metrics=['accuracy'])

    # model_cx.summary()

    print('the number of layers in this model_cx:' + str(len(model.layers)))


    #保存模型
    out_dir = "../weights/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filepath ="../weights/densenet_{epoch:04d}.h5"
    # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False,
                                 mode='max')
    #学习率调整
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1,
                                  min_lr=0.000005, mode="min")
    # 早停
    earlystopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')
    #保存训练过程
    log_dir = "../logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logfile="../logs/densenet.csv"
    log=keras.callbacks.CSVLogger(logfile, separator=',', append=False)
    loggraph=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    callbacks_list = [checkpoint,lr_reduce,log]
    # 训练
    model.fit_generator(train_batches, steps_per_epoch=steps_per_epoch, validation_data=valid_batches,
                        validation_steps=validation_steps, epochs=epochs, verbose=2,
                        callbacks=callbacks_list,workers=16,max_queue_size=20)

if  __name__=="__main__":
    densenet(1,200,210,15,(224,224))


#densenet121  batch=16 acc:93.3    densenet(3,50,20,5 ,(224,224))  SGD

#factor:0.5 pat:10







