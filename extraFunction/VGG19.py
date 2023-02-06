# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:14:02 2022

@author: Judy
"""

from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Model

def VGG19Model():
    Img_Height=224
    Img_weight=224
    Img_channel=3
    
    Vgg19Model=VGG19(weights='imagenet',include_top=False,input_shape=(Img_Height,Img_weight,Img_channel))
    for layer in Vgg19Model.layers:
        layer.trainable=False
    x = Flatten()(Vgg19Model.output)
    x = Dense(512,activation='relu')(x)
    # x = Dense(512,activation='relu')(x)
    # x = Dense(512,activation='relu')(x)
    predictions=Dense(2,activation='softmax')(x)
    full_model=Model(inputs=Vgg19Model.input,outputs=predictions)
    return full_model