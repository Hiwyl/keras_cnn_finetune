'''
@Author :      lance
@Email :   wangyl306@163.com
    多模型投票对抗过拟合
 '''
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

#预测模块
def pred(path,steps,input_shape):
    #导入数据
    test_path = '测试'
    test_batches = ImageDataGenerator(rescale=1/255).flow_from_directory(test_path,
                                                                         target_size=input_shape,
                                                                         classes=["C2F","X2F"],
                                                                         batch_size=10,                                                                                                                shuffle=False)
    model = load_model(path)
    # 测试
    steps = steps
    pred = model.predict_generator(test_batches, steps=steps, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)
    # print("预测结果:", predicted_class_indices)

    return predicted_class_indices


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
    pred1=list(pred(path,steps,input_shape))
    for i in range(len(pred1)):
        ls = []
        ls.append(pred1[i])


        ans.append(ls[vote(ls)])
    return ans


if __name__=="__main__":
    predicts=Ensemble()
    for i in enumerate(predicts):
        print(i)
