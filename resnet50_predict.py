import IPython
import json, os, re, sys, time
import numpy as np

from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

class ResNetP:
    def __init__(self,model_path):
        t0 = time.time()
        self.model = load_model(model_path)
        t1 = time.time()
        print('Loaded in:', t1 - t0)

    def predict(self,img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds = self.model.predict(x)
        return preds

if __name__ == '__main__':
    model_path = "resnet50_best.h5"
    # print('Loading model:', model_path)
    resnet=ResNetP(model_path)
    path = '../white_mouse/test/1/'
    files = os.listdir(path)

    aaa=0
    index=0
    for file in files:
    # print('Generating predictions on image:', sys.argv[2])
        t0 = time.time()
        preds = resnet.predict(path + file)
        t1 = time.time()
        print('Loaded in:', t1 - t0)
        result=np.argmax(preds,1)[0]
        aaa+=result
        print(result)
        index+=1
    print(aaa,index)
