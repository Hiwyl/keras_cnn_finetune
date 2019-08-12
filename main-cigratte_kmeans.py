# -*- encoding: utf-8 -*-
'''
@Author :      lance  
@Email :   wangyl306@163.com
 '''
import numpy as np
import keras
from keras.models import Sequential 
from sklearn import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import Activation, Dense

valid_datagen = ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory( 'cx/train',
                                                    target_size=(224,224),batch_size=10,class_mode='binary')

valid_gen = valid_datagen.flow_from_directory( 'cx/test',
                                                    target_size=(224,224),batch_size=10,class_mode='binary')


test_gen = test_datagen.flow_from_directory( 'cx/测试cx',
                                                    target_size=(224,224),batch_size=10,class_mode='binary')


#os.chdir('E:/Python/ANN/Projects/烟草')
model=keras.models.load_model("cx_weights/weights/densenet_0017.h5")
model.summary()    

Labels=[]
Predicts=[]

for i in range(3):
    print(i)
    temp=next(test_gen)
    temp1=temp[0]
    
    Labels.append(temp[1])
    Predicts.append(model.predict(temp1))
    
    
Labels=np.array(Labels).reshape([30,1])   

Predicts=np.array(Predicts).reshape([30,1])
    
temp=np.zeros([30,1])
temp[Predicts>0.5]=1    

cf_metrics=metrics.confusion_matrix(Labels,temp)

    
    
    
model_inter = keras.Model(inputs=model.input,
                                     outputs=model.get_layer('dense_1').output)

model_inter.summary()

 
Labels=[]
Predicts=[]

for i in range(111):
    print(i)
    temp=next(train_gen)
    temp1=temp[0]
    
    Labels.append(temp[1])
    Predicts.append(model_inter.predict(temp1))
 

Train_X=np.array(Predicts).reshape([1110,1024])

Train_Y=np.array(Labels).reshape([1110,1])

Labels=[]
Predicts=[]

for i in range(47):
    print(i)
    temp=next(valid_gen)
    temp1=temp[0]
    
    Labels.append(temp[1])
    Predicts.append(model_inter.predict(temp1))
 

Valid_X=np.array(Predicts).reshape([470,1024])

Valid_Y=np.array(Labels).reshape([470,1])

Labels=[]
Predicts=[]

for i in range(3):
    print(i)
    temp=next(test_gen)
    temp1=temp[0]
    
    Labels.append(temp[1])
    Predicts.append(model_inter.predict(temp1))
 

Test_X=np.array(Predicts).reshape([30,1024])

Test_Y=np.array(Labels).reshape([30,1])

      
# In[ ]:
 
    
model_dense= Sequential([
        Dense(16,input_dim=1024),
        Activation('tanh'),
        Dense(1),
        Activation('sigmoid')
        ])

adam=Adam(lr=0.01)


model_dense.compile(
        optimizer=adam,
        loss='mean_squared_error',
#        loss='binary_crossentropy',
        metrics=['accuracy'],
        )

model_dense.fit(Train_X,Train_Y,nb_epoch=1000,batch_size=1120)    
    
    
 # In[ ]:
    

Y_predict = model_dense.predict(Train_X) 

Y_predict[Y_predict>=0.5]=1

Y_predict[Y_predict<0.5]=0

cf_metrics=metrics.confusion_matrix(Train_Y, Y_predict)
accuracy = metrics.accuracy_score(Train_Y, Y_predict)
print(accuracy)    
    
    
#cx的阈值为0.005区分 >0.005=1 X2F    
Y_predict = model_dense.predict(Valid_X) 

Y_predict[Y_predict>=0.5]=1

Y_predict[Y_predict<0.5]=0

cf_metrics=metrics.confusion_matrix(Valid_Y, Y_predict)
accuracy = metrics.accuracy_score(Valid_Y, Y_predict)
print(accuracy)    
      
    
Y_predict = model_dense.predict(Test_X) 

Y_predict[Y_predict>=0.005]=1

Y_predict[Y_predict<0.005]=0

cf_metrics=metrics.confusion_matrix(Test_Y, Y_predict)
accuracy = metrics.accuracy_score(Test_Y, Y_predict)
print(accuracy)    
    
 
# In[ ]:
    
model_dense.summary()    
    
model_feather = keras.Model(inputs=model_dense.input,
                                     outputs=model_dense.get_layer('dense_1').output)

model_feather.summary()
#predicts_inter = model_inter.predict(temp1)    
    
X_train=model_feather.predict(Train_X)
X_valid=model_feather.predict(Valid_X)
X_test=model_feather.predict(Test_X)



 
# In[ ]:
    
    

from sklearn.cluster import KMeans
clf=KMeans(n_clusters=2)

#attention, the cluster sequence will be generated randomly!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
clf.fit(X_train)
#attention, the cluster sequence will be generated randomly!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

predicts=clf.predict(X_train)

cf_metrics=metrics.confusion_matrix(Train_Y, predicts)
accuracy = metrics.accuracy_score(Train_Y, predicts)
print(accuracy)   


predicts=clf.predict(X_valid)

cf_metrics=metrics.confusion_matrix(Valid_Y, predicts)
accuracy = metrics.accuracy_score(Valid_Y, predicts)
print(accuracy)  


predicts=clf.predict(X_test)

cf_metrics=metrics.confusion_matrix(Test_Y, predicts)
accuracy = metrics.accuracy_score(Test_Y, predicts)
print(accuracy)  


threshold=3000

clf_clusters=clf.cluster_centers_

dis_1=X_test-clf_clusters[0,:]
dis_1=np.sum(dis_1**2,axis=1)

dis_2=X_test-clf_clusters[1,:] 
dis_2=np.sum(dis_2**2,axis=1)
predicts_t=np.array(dis_2<threshold)+0

cf_metrics=metrics.confusion_matrix(Test_Y, predicts_t)
accuracy = metrics.accuracy_score(Test_Y, predicts_t)
print(accuracy)



