from skimage import io,transform
import numpy as np
import os
from keras.models import load_model
from keras import backend as K
K.clear_session

def predict(model_path,data_path,shape):

    new_model=load_model(model_path)
    print(dir(new_model))

    # path=data_path
    # files=os.listdir(path)
    # # print(files)
    #
    # Tmp_Img=[]
    # for i in range(len(files)):
    #    tmp=io.imread(path+'/'+files[i])
    #    tmp_img=transform.resize(tmp,shape)
    #    Tmp_Img.append(tmp_img)
    # Tmp_Img=np.array(Tmp_Img)
    # Tmp_Img=Tmp_Img*255
    # predicts=new_model.predict(Tmp_Img)
    # predicts=np.argmax(predicts, axis=1).astype(np.str)
    # predicts[predicts=="0"]="C4F"
    # predicts[predicts=="1"]="X2F"
    # for i in zip(files,predicts):
    #    print("\n",i)

if __name__=="__main__":
    predict("weights/all_resnet2.h5","data/test/X2F",[224,224])

