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
                                                                         batch_size=10,                                                                                                               shuffle=False)

    model = load_model(path)
    # 测试
    steps=steps
    test_class=np.array([])

    for i in range(steps):
        test_imgs, test_lables = next(test_batches)
        test_lables = np.argmax(test_lables, axis=1)
        test_lables = np.argmax(test_lables)
        test_class=np.hstack((test_class,test_lables ))
    print("真实类别：",test_class)

    pred = model.predict_generator(test_batches, steps=steps, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)
    print("预测结果:", predicted_class_indices)

    # 打印混淆矩阵
    cm = confusion_matrix(test_class, predicted_class_indices)


    print(cm)

    tmp = 0
    for i in range(len(cm[0, :])):
        tmp += cm[i][i]
    accuracy = tmp / np.sum(cm)
    print("acc:", accuracy)

    return path, accuracy



if __name__=="__main__":
    # mytest("weights/mobilenetv2_0025.h5",30,(224,224)) #0.73
    # mytest("weights/vgg19one_0032h5",30,(224,224))  #0.75
    # mytest("weights/vgg19two_0024.h5",30,(224,224))  #0.84
    mytest("weights/vgg16_0007.h5",30,(224,224))  #0.84

