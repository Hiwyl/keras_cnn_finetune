import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


import os
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
import keras

from model.load_data import load_data
def continue_train(path,epochs,steps_per_epoch,validation_steps,input_shape):
    # 加载数据
    train_batches, valid_batches= load_data(input_shape)

    model = load_model(path)
    model.summary()

    print('the number of layers in this model:' + str(len(model.layers)))

    # 保存模型
    out_dir = "../weights/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filepath = "../weights/continue_train.h5"
    # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True,
                                 mode='max')
    # 学习率调整
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1,
                                  min_lr=0.00000001, mode="min")
    # 早停
    earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    # 保存训练过程
    log_dir = "../logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logfile = "../logs/continue_train.csv"
    log = keras.callbacks.CSVLogger(logfile, separator=',', append=False)
    loggraph = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    callbacks_list = [checkpoint, lr_reduce, log,earlystopping]
    # 训练
    model.fit_generator(train_batches, steps_per_epoch=steps_per_epoch, validation_data=valid_batches,
                        validation_steps=validation_steps, epochs=epochs, verbose=2,
                        callbacks=callbacks_list,workers=16,max_queue_size=20)


if __name__ == "__main__":
    continue_train("../weights/bcnn.h5", 50, 15,3, (224, 224))



