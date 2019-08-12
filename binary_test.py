'''
@Author :      lance
@Email :   wangyl306@163.com
 '''


from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np



def mytest(path,steps,input_shape):
    #导入数据
    test_path = '测试'
    test_batches = ImageDataGenerator(rescale=1/255).flow_from_directory(test_path,
                                                                         target_size=input_shape,
                                                                         classes=["C2F","X2F"],
                                                                         class_mode="binary",
                                                                         batch_size=10,                                                                                                               shuffle=False)

    model = load_model(path)
    # 测试
    steps=steps
    test_class=np.array([])

    for i in range(steps):
        test_imgs, test_lables = next(test_batches)
        test_class=np.hstack((test_class,test_lables ))
    print("真实类别：",test_class)

    pred = model.predict_generator(test_batches, steps=steps, verbose=1)
    pred=pred.ravel()
    pred=list(pred)
    for i in range(len(pred)):
        if pred[i]<0.5:
            pred[i]=0
        else:
            pred[i]=1
    print("预测结果:", pred)


    # 打印混淆矩阵
    cm = confusion_matrix(test_class, pred)


    print(cm)

    tmp = 0
    for i in range(len(cm[0, :])):
        tmp += cm[i][i]
    accuracy = tmp / np.sum(cm)
    print("acc:", accuracy)

    return path, accuracy



if __name__=="__main__":
    mytest("weights/bcnn_0033.h5",25,(224,224))#0.77
    # mytest("weights/densenet_0023.h5",25,(224,224)) #0.87
    # mytest("weights/ince_res_0021.h5",25,(299,299)) #0.85
    # mytest("weights/inceptionv3_0033.h5",25,(299,299)) #0.80
    # mytest("weights/merge_0022.h5",25,(224,224)) #0.81
    # mytest("weights/mobilenetv2_0032.h5",25,(224,224)) #0.87
    # mytest("weights/nasnet_0017.h5",25,(224,224)) #0.87
    # mytest("weights/resnet_0018.h5",25,(224,224)) #0.79
    # mytest("weights/vgg19two_0022.h5",25,(224,224)) #0.82




