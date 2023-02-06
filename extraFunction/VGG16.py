# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 01:01:53 2022

@author: Judy
"""

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Model
Img_Height=224
Img_weight=224
Img_channel=3

def VGG16Model():
    VGG16Model=VGG16(weights='imagenet',include_top=False,input_shape=(Img_Height,Img_weight,Img_channel))
    for layer in VGG16Model.layers:
        layer.trainable=False
    x = Flatten()(VGG16Model.output)
    x = Dense(512,activation='relu')(x)
    # x = Dense(512,activation='relu')(x)
    # x = Dense(512,activation='relu')(x)
    predictions=Dense(2,activation='softmax')(x)
    full_model=Model(inputs=VGG16Model.input,outputs=predictions)
    return full_model  