# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 19:49:30 2019

@author: Asjad
"""

# load json and create model
from keras.models import model_from_json
import numpy as np
from glob import glob
import os

from keras.preprocessing import image

def something():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
     
    # evaluate loaded model on test data
    loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    print("Samples Prediction of PNEUMONIA X-Rays")
    path = './input/**'
    path_content = glob(path)
    print(len(path_content))
    print(path.split('/')[-2])
    predctr = 0
    
    for p in path_content:
    #     img = random.sample(p, 1)
        
    
        test_image = image.load_img(p, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        if result[0][0] == 1:
            prediction = 'PNEUMONIA'
            predctr +=1
        else:
            prediction = 'NORMAL'
            
        print("Filename = "+os.path.basename(p)+"\t\t Prediction = "+prediction)

something()