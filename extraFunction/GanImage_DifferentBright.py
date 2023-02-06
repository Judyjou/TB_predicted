# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:38:59 2022

@author: Judy
"""
import numpy as np
import cv2
import os

# trainFileName=['Gantest1','Gantest2','Gantest3']
# BlurFileName=['Gantest1_8_BK3','Gantest1_9_BK3','Gantest1_10_BK3']
# GanImage=os.path.join(base_path,'GanImage')
# make_file(GanImage)
# brightNum=[0.8,0.9,1.0,1.1,1.2]

def make_file(Folder):
    if not os.path.isdir(Folder):  #如果資料夾不存在就建立
        os.mkdir(Folder)
        
def imgBrightness(img, c, b):
    rows, cols, channels = img.shape
    blank = np.zeros([rows, cols, channels], img.dtype)
    rst = cv2.addWeighted(img, c, blank, 1-c, b)
    return rst

def var_GanFilePath(trainFileName,GanImage,brightNum,BlurFileName):
    for i in range(len(trainFileName)):
        globals()[trainFileName[i]]=os.path.join(GanImage,trainFileName[i])
        make_file(globals()[trainFileName[i]])
        for j in range(len(brightNum)):
            globals()[trainFileName[i]+"_"+str(brightNum[j])]=os.path.join(globals()[trainFileName[i]],trainFileName[i]+"_"+str(brightNum[j]))
            make_file(globals()[trainFileName[i]+"_"+str(brightNum[j])])
        #原始亮度、條輛、調暗+不同模糊各資料夾
        for k in range(len(BlurFileName)):
            globals()[BlurFileName[k]]=os.path.join(globals()[trainFileName[i]],BlurFileName[k])
            make_file(globals()[BlurFileName[k]])

def Gan_files(trainFileName,GanImage,OritrainImgPath,BlurFileName,brightNum):
    for i in range(len(trainFileName)):
        globals()[trainFileName[i]]=os.path.join(GanImage,trainFileName[i])  
        #原始亮度、條輛、調暗各資料夾
        img_idx=0  
        for j in range(len(brightNum)):
            globals()[trainFileName[i]+"_"+str(brightNum[j])]=os.path.join(globals()[trainFileName[i]],trainFileName[i]+"_"+str(brightNum[j]))
            for fi in os.listdir(OritrainImgPath):
                for img in os.listdir(os.path.join(OritrainImgPath,fi)):
                    path=os.path.join(OritrainImgPath,fi)
                    r_img=cv2.imread(os.path.join(path,img))
                    
                    for BV in range(len(brightNum)):
                        rst = imgBrightness(r_img, brightNum[BV], 3)
                        if str(int(brightNum[BV]*10))==str(trainFileName[i]+"_"+str(brightNum[j])).split('_')[1]:
                            # print(os.path.join(trainFileName[i]+"_"+str(j),img.split('_')[0]+"_"+str(img_idx)+"_"+img.split('_')[1]))
                            cv2.imwrite(os.path.join(globals()[trainFileName[i]+"_"+str(brightNum[j])],img.split('_')[0]+"_"+str(img_idx)+"_"+img.split('_')[1]),rst)
            img_idx+=1
        #原始亮度、條輛、調暗+不同模糊各資料夾
        for k in range(len(BlurFileName)):
            globals()[BlurFileName[k]]=os.path.join(globals()[trainFileName[i]],BlurFileName[k])
        
def aveBlur(oriPath,kernel_size,brightNum,savePath):       
    for i in os.listdir(oriPath):            
        rimg=cv2.imread(os.path.join(oriPath,i))
        avg_img=cv2.blur(rimg,kernel_size)
        inx=int(i.split('_')[1])+len(brightNum) #依序加0-->4  3-->7
        # print(i.split('_')[0],i.split('_')[1],inx)
        cv2.imwrite(os.path.join(savePath,i.split('_')[0]+"_"+str(inx)+"_"+i.split('_')[2]), avg_img)
       
        
def GaussBlur(oriPath,kernel_size,sigma,brightNum,savePath):
    for i in os.listdir(oriPath):
        rimg=cv2.imread(os.path.join(oriPath,i))
        GB_img=cv2.GaussianBlur(rimg,kernel_size,sigma)
        inx=int(i.split('_')[1])+len(brightNum)
        cv2.imwrite(os.path.join(savePath,i.split('_')[0]+"_"+str(inx)+"_"+i.split('_')[2]), GB_img)        
        