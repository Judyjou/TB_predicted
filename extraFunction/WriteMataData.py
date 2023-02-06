# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:12:44 2022

@author: Judy
"""

import cv2
import os
import pandas as pd 
import numpy as np

def write_csv(img_filePath,save_csvFileName,storage_path):
    Number=[]
    Label=[]
    Shape=[]
    for img in os.listdir(img_filePath):
        r_img=cv2.imread(os.path.join(img_filePath,img))
        arr=np.array(r_img)
        Number.append(img[0:-4])
        Label.append(img[-5])
        Shape.append(r_img.shape)
    def_dict={"Number":Number,"Label":Label,"Shape":Shape}
    df=pd.DataFrame(def_dict)        
    df.to_csv(os.path.join(storage_path,str(save_csvFileName+".csv")))

def addData(addDataSource,Oritrain_testData,storagePath,train_test_csv):
    Number=[]
    Label=[]
    Shape=[]
    for img in os.listdir(addDataSource):
        r_img=cv2.imread(os.path.join(addDataSource,img))
        Number.append(img[0:-4])
        Label.append(img[-5])
        Shape.append(r_img.shape)
    def_dict={"Number":Number,"Label":Label,"Shape":Shape}
    df=pd.DataFrame(def_dict,columns=['Number','Label','Shape'])
    OriData=pd.DataFrame(Oritrain_testData,columns=['Number','Label','Shape'])
    NewData=OriData.append(df,ignore_index=True)
    NewData.to_csv(os.path.join(storagePath,str(train_test_csv)+".csv"))
    

def CSVAdd_ArrImage(oriImage,oriMatadata,savePath,NewCSVName):
    ImgArr=[]
    resizeShape=[]
    for j in os.listdir(oriImage):
        for k in oriMatadata["Number"]:
            if j[:-4] == k:                
                rImg=cv2.imread(os.path.join(oriImage,j))
                resizeShape.append(rImg.shape)
                arr=np.array(rImg)
                oriMatadata["resizeShape"]=oriMatadata["Number"].map(lambda x:resizeShape)
                oriMatadata["Image_array"]=oriMatadata["Number"].map(lambda x:arr)
                ImgArr.append(arr)  
    oriMatadata["resizeShape"]=resizeShape            
    oriMatadata["Image_array"]=ImgArr
    oriMatadata.to_csv(os.path.join(savePath,str(NewCSVName)+".csv"))    
    NewCSVName=pd.read_csv(os.path.join(savePath,str(NewCSVName)+".csv"),encoding= 'utf-8')

