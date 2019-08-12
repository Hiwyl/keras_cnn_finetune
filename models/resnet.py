from keras import backend as K, Model
import os
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import SGD, Adam
import numpy as np
from load_data import load_data







def ResNet50(classes,epochs, steps_per_epoch, validation_steps,input_shape):
    # 加载数据
    train_batches, valid_batches = load_data(input_shape)

    input_shape += (3,)
    base_model = keras.applications.resnet50.ResNet50(include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
    x = Dropout(0.5)(x)

    if classes==1:
        print("二元分类")
        outputs = Dense(classes, activation='sigmoid')(x)
        model = Model(base_model.inputs, outputs)

        sgd = Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    else:
        print("多分类")
        outputs = Dense(classes, activation='softmax')(x)
        model = Model(base_model.inputs, outputs)

        sgd =Adam(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # model_cx.summary()

    print('the number of layers in this model_cx:' + str(len(model.layers)))

    # 保存模型
    out_dir = "weights/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filepath = "weights/resnet_{epoch:04d}.h5"
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
    logfile = "logs/resnet.csv"
    log = keras.callbacks.CSVLogger(logfile, separator=',', append=False)
    loggraph = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)


    callbacks_list = [checkpoint, lr_reduce, log]

    # 训练
    model.fit_generator(train_batches, steps_per_epoch=steps_per_epoch, validation_data=valid_batches,
                        validation_steps=validation_steps, epochs=epochs, verbose=2,
                        callbacks=callbacks_list,workers=16,max_queue_size=20)


if __name__ == "__main__":
    ResNet50(1, 200, 50,7, (224, 224))
