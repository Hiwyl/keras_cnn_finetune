# -*- encoding: utf-8 -*-
'''
@Author :      lance  
@Email :   wangyl306@163.com
 '''
import time
from model_cx.inceptionresnet import inceptionresnet
from model_cx.vgg19two import vgg19_all_lr
from model_cx.inceptionv3 import inceptionv3
from model_cx.densenet import densenet
from model_cx.nasnet import nasnet
from model_cx.merge import merge
from model_cx.bcnn import  bilinearnet
from model_cx.resnet import ResNet50
from model_cx.mobilenetv2 import mobilenetv2
from model_cx.senet import senet





if __name__=="__main__":
    classes = 1
    epochs = 100
    steps_per_epoch = 113
    validation_steps = 48
    shape=(224,224)
    print("开始训练...")
    start = time.time()
    #
    # try:
    #     print("densenet")
    #     densenet(classes, epochs, steps_per_epoch, validation_steps, shape)
    # except Exception as e:
    #     print(e)
    # try:
    #     print("bcnn")
    #     bilinearnet(classes, epochs, steps_per_epoch, validation_steps, shape)
    #
    # except Exception as e:
    #     print(e)
    # try:
    #     print("resnet")
    #     ResNet50(classes, epochs, steps_per_epoch, validation_steps, shape)
    # except Exception as e:
    #     print(e)
    try:
        print("merge")
        merge(classes, epochs, steps_per_epoch, validation_steps, shape)
    except Exception as e:
        print(e)
    # try:
    #     print("ince_res")
    #     inceptionresnet(classes, epochs, steps_per_epoch, validation_steps, (299, 299))
    #     # inceptionresnet(classes, epochs, steps_per_epoch, validation_steps, shape)
    # except Exception as e:
    #     print(e)
    # try:
    #     print("mobilenetv2")
    #     mobilenetv2(classes, epochs, steps_per_epoch, validation_steps, shape)
    # except Exception as e:
    #     print(e)
    # try:
    #     print("inceptionv3")
    #     inceptionv3(classes, epochs, steps_per_epoch, validation_steps, (299, 299))
    #     # inceptionv3(classes, epochs, steps_per_epoch, validation_steps, shape)
    # except Exception as e:
    #     print(e)
    try:
        print("nasnet")
        nasnet(classes, epochs, steps_per_epoch, validation_steps, shape)
    except Exception as e:
        print(e)
    try:
        print("vgg19two")
        vgg19_all_lr(classes, epochs, steps_per_epoch, validation_steps, shape)
    except Exception as e:
        print(e)
    try:
        print("senet")
        vgg19_all_lr(classes, epochs, steps_per_epoch, validation_steps, (100,100))
    except Exception as e:
        print(e)
    end = time.time()
    print("ETA:", (end - start) / 3600)