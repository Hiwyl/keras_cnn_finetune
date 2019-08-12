'''
@Author :      lance
@Email :   wangyl306@163.com
  分级代码
 '''


from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
import numpy as np




def mytest1(path,steps,input_shape):
    #导入数据

    test_path = 'testimgs/t'
    test_batches = ImageDataGenerator(rotation_range=360,rescale=1/255).flow_from_directory(test_path, target_size=input_shape,
                                                                            classes=["C4F","X2F"], batch_size=10,                                                                          shuffle=False)

    model = load_model(path)
    # 测试
    steps=steps
    pred = model.predict_generator(test_batches, steps=steps, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)
    # print("预测结果:", predicted_class_indices)


    return predicted_class_indices


def mytest2(path,steps,input_shape):
    #导入数据

    test_path = 'testimgs'
    test_batches = ImageDataGenerator(rescale=1/255).flow_from_directory(test_path, target_size=input_shape,
                                                                            class_mode=None,classes=["X2F","X3F"], batch_size=10,                                                                          shuffle=False)

    model = load_model(path)
    # 测试
    steps=steps
    pred = model.predict_generator(test_batches, steps=steps, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)
    # print("预测结果:", predicted_class_indices)


    return predicted_class_indices

if __name__=="__main__":
    # pred1=mytest1("weight/model11.h5",2,(224,224))
    # pred1=list(pred1)
    # for i in range(len(pred1)):
    #     if pred1[i]==0:
    #         pred1[i]="C4F"
    #     else:
    #         pred1[i]="X2F"
    # print(pred1)
    #
    pred2=mytest2("x23/测试6之后的模型/bcnn_0141.h5",3,(224,224))
    # pred2=mytest2("x23/测试6之后的模型/resnet_0093.h5",2,(224,224))
    pred2=list(pred2)

    for i in range(len(pred2)):
        if pred2[i]==0:
            pred2[i]="X2F"
        else:
            pred2[i]="X3F"
    #
    #
    # for i in range(len(pred1)):
    #     if pred1[i]=="X2F":
    #         pred1[i]=pred2[i]
    # print(pred2)

    for i in enumerate(pred2):
        print(i)

