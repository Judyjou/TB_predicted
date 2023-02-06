# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:05:39 2022

@author: Judy
""" 
from tensorflow.keras import models  
from tensorflow.keras import layers 
import tensorflow as tf

Img_Height=256
Img_weight=256
Img_channel=3

def alexNet():
    full_model = models.Sequential()
    full_model.add(layers.experimental.preprocessing.Resizing(256, 256, interpolation="bilinear", input_shape=(Img_Height,Img_weight,Img_channel)))
    full_model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
    full_model.add(layers.Lambda(tf.nn.local_response_normalization))
    full_model.add(layers.Activation('relu'))
    full_model.add(layers.MaxPooling2D(3, strides=2))
    full_model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
    full_model.add(layers.Lambda(tf.nn.local_response_normalization))
    full_model.add(layers.Activation('relu'))
    full_model.add(layers.MaxPooling2D(3, strides=2))
    full_model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    full_model.add(layers.Activation('relu'))
    full_model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    full_model.add(layers.Activation('relu'))
    full_model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
    full_model.add(layers.Activation('relu'))
    full_model.add(layers.Flatten())
    full_model.add(layers.Dense(4096, activation='relu'))
    full_model.add(layers.Dropout(0.5))
    full_model.add(layers.Dense(4096, activation='relu'))
    full_model.add(layers.Dropout(0.5))
    full_model.add(layers.Dense(2, activation='softmax'))
    return full_model