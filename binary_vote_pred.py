'''
@Author :      lance
@Email :   wangyl306@163.com
    多模型投票对抗过拟合
 '''
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix


def pred(path,steps,input_shape):
    #导入数据
    test_path = '测试'
    test_batches = ImageDataGenerator(rescale=1/255).flow_from_directory(test_path,
                                                                         target_size=input_shape,
                                                                         classes=["C2F","X2F"],
                                                                         class_mode="binary",
                                                                         batch_size=10,                                                                                                                shuffle=False)
    model = load_model(path)
    # 测试
    steps = steps
    pred = model.predict_generator(test_batches, steps=steps, verbose=1)
    pred = pred.ravel()
    pred = list(pred)
    for i in range(len(pred)):
        if pred[i] < 0.5:
            pred[i] = 0
        else:
            pred[i] = 1
    # print("预测结果:", pred)
    return pred


#投票选出最多的
def vote(lt):
	index1 = 0
	max = 0
	for i in range(len(lt)):
		flag = 0
		for j in range(i+1,len(lt)):
			if lt[j] == lt[i]:
				flag += 1
		if flag > max:
			max = flag
			index1 = i
	return index1

def Ensemble():
    ans = []
    pred1=list(pred("weights/nasnet_0039.h5",25,(224,224)))
    pred2=list(pred("weights/vgg19two_0027.h5",25,(224,224)))
    pred3=list(pred("weights/inceptionv3_0016.h5",25,(299,299)))
    pred4=list(pred("weights/mobilenetv2_0029.h5",25,(224,224)))
    pred5=list(pred("weights/densenet_0025.h5",25,(224,224)))
    pred6=list(pred("weights/bcnn_0020.h5",25,(224,224)))
    pred7=list(pred("weights/resnet_0025.h5",25,(224,224)))
    for i in range(len(pred1)):
        ls = []
        ls.append(pred1[i])
        ls.append(pred2[i])
        ls.append(pred3[i])
        ls.append(pred4[i])
        ls.append(pred5[i])
        ls.append(pred6[i])
        ls.append(pred7[i])


        ans.append(ls[vote(ls)])
    return ans


if __name__=="__main__":
    predicts=Ensemble()
    # for i in enumerate(predicts):
    #     print(i)

    test_path = '测试'
    test_batches = ImageDataGenerator(rescale=1 / 255).flow_from_directory(test_path,
                                                                           target_size=(224,224),
                                                                           classes=["C2F", "X2F"],
                                                                           class_mode="binary",
                                                                           batch_size=10, shuffle=False)
    test_class = np.array([])

    for i in range(25):
        test_imgs, test_lables = next(test_batches)
        test_class = np.hstack((test_class, test_lables))
    print("真实类别：", test_class)

    # 打印混淆矩阵
    cm = confusion_matrix(test_class, predicts)

    print(cm)

    tmp = 0
    for i in range(len(cm[0, :])):
        tmp += cm[i][i]
    accuracy = tmp / np.sum(cm)
    print("acc:", accuracy)

