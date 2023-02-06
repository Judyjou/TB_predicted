# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 17:47:40 2022

@author: Judy
"""

# import h5 model result and print result in jaon 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import numpy as np
import os,sys
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from flask import request
from flask import jsonify
from PIL import Image
import cv2
import shutil
from pydicom import dcmread #pip install pydicom
from extraFunction.ImageProcessing import cut_letterboxing,squImg,resizeImg,avgBlur,BLUE_EDGE,resizeImg,ORI_EDGE
import matplotlib.pyplot as plt


def make_file(Folder):
    if not os.path.isdir(Folder):  #if file not exist made file
        os.mkdir(Folder)

def predict_mainPart():
    #base file path
    staticPath=os.path.join(os.getcwd(),"static")
    model = tf.keras.models.load_model('GoogLeNet_STTI_o2s8_exn10416Pre_SGD_SHZ_correct_5.h5')
    #def image file path & makenew file store image processing file
    imagePath=os.path.join(staticPath,'image')
    make_file(imagePath)
    store_reOri=os.path.join(staticPath,'store_reOri')
    make_file(store_reOri)
    reSavePath = os.path.join(staticPath,'reSave')
    make_file(reSavePath)
    tempPath=os.path.join(staticPath,'temp')
    make_file(tempPath)
    processingPath=os.path.join(staticPath,'processing')
    make_file(processingPath)
    
    allowFile=['png','jpg','jpeg','dcm']
    for i in os.listdir(imagePath):
        target=os.path.join(imagePath, i)
    
    EXT=target.split('.')[1]
    if EXT not in allowFile:
        os.remove(target)
        shutil.rmtree(reSavePath)
        shutil.rmtree(processingPath)
        shutil.rmtree(store_reOri)
        shutil.rmtree(tempPath)
        print("File extension error!! Please upload extension png, jpg, jpeg.")
        sys.exit(0)
    
    if EXT == 'dcm':
        Read_dcm=dcmread(target)
        image_arr= Read_dcm.pixel_array
        os.remove(target)
        cv2.imwrite(target[:-3]+'png', image_arr)
        
        
        cut_letterboxing(imagePath,tempPath)
        squImg(tempPath,tempPath)
        resizeImg(tempPath, store_reOri)
        avgBlur(tempPath,(5,5),processingPath)
        BLUE_EDGE(processingPath,processingPath)
        resizeImg(processingPath, processingPath)
        ORI_EDGE(store_reOri,processingPath,0.2,0.8,processingPath)
        for img in os.listdir(processingPath):
            prImg=os.path.join(processingPath, img)
        shutil.copy(prImg,reSavePath)
        
        
        for i in os.listdir(reSavePath):   
            file_path=os.path.join(reSavePath, i)
        
        #load target image and transform to array
        img=image.load_img(file_path)
        x=image.img_to_array(img)
        
        nor_x=x.reshape((1, 224, 224, 3)).astype('float32') /255
        predict_result=model.predict(nor_x)
        predict_classResult=predict_result[2]
        print("Normal probability: ",predict_classResult[0,0],"Tuberculosis probability: ",predict_classResult[0,1])
        
    
    else:
        #import image need reshape to 224*224 square image
        cut_letterboxing(imagePath,tempPath)
        squImg(tempPath,tempPath)
        resizeImg(tempPath, store_reOri)
        # resizeImg(tempPath, reSavePath)
        avgBlur(tempPath,(5,5),processingPath)
        BLUE_EDGE(processingPath,processingPath)
        resizeImg(processingPath, processingPath)
        ORI_EDGE(store_reOri,processingPath,0.2,0.8,processingPath)
        for img in os.listdir(processingPath):
            prImg=os.path.join(processingPath, img)
        shutil.copy(prImg,reSavePath)
        
        
        for i in os.listdir(reSavePath):   
            file_path=os.path.join(reSavePath, i)
        
        #load target image and transform to array
        img=image.load_img(file_path)
        x=image.img_to_array(img)
        
        nor_x=x.reshape((1, 224, 224, 3)).astype('float32') /255
        predict_result=model.predict(nor_x)
        predict_classResult=predict_result[2]
        # print("Normal probability: ",predict_classResult[0,0],"Tuberculosis probability: ",predict_classResult[0,1])
    
    wResult=os.path.join(os.getcwd(),"Result.txt")
    owResult=open(wResult,'w',encoding="utf-8")    
    owResult.write("Normal probability: ")
    owResult.write(str(predict_classResult[0,0]))
    owResult.write('\n')
    owResult.write("Tuberculosis probability: ")
    owResult.write(str(predict_classResult[0,1]))
    
    owResult.close()
        
    for i in os.listdir(imagePath):
        target=os.path.join(imagePath, i)
    # finall remove all processing images
    os.remove(target)
    # shutil.rmtree(imagePath)
    shutil.rmtree(reSavePath)
    shutil.rmtree(processingPath)
    shutil.rmtree(store_reOri)
    shutil.rmtree(tempPath)
    FinallResult="Normal probability: "+str(predict_classResult[0,0])+"Tuberculosis probability: "+str(predict_classResult[0,1])
    return FinallResult

if __name__=='__main__':
    print(predict_mainPart())



 