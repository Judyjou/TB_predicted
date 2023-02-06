# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 13:43:48 2022

@author: Judy
"""

#GanImage
#新增4種調整好的亮度，並重新命名 原名稱_擴增張數_判斷有無TB標籤
import numpy as np
import cv2
import os

def imgBrightness(img, c, b):
    rows, cols, channels = img.shape
    blank = np.zeros([rows, cols, channels], img.dtype)
    rst = cv2.addWeighted(img, c, blank, 1-c, b)
    return rst

def setBright(oriPath,value,img_idx,savePath): #value=0.8~1.2
    for i in os.listdir(oriPath):  
        r_img=cv2.imread(os.path.join(oriPath,i))
        rst = imgBrightness(r_img, value, 3)
        # print(i.split('_')[0]+"_"+str(img_idx)+"_"+i.split('_')[1])
        cv2.imwrite(os.path.join(savePath,i.split('_')[0]+"_"+str(img_idx)+"_"+i.split('_')[1]),rst)

brightNum=[0.8,0.9,1.0,1.1,1.2] #(擴增影像的亮度決定)5張為一命名級距
        
def aveBlur(oriPath,kernel_size,savePath):       
    for i in os.listdir(oriPath):            
        rimg=cv2.imread(os.path.join(oriPath,i))
        avg_img=cv2.blur(rimg,kernel_size)
        inx=int(i.split('_')[1])+len(brightNum) #依序加0-->5  3-->7
        # print(i.split('_')[0],i.split('_')[1],inx)
        cv2.imwrite(os.path.join(savePath,i.split('_')[0]+"_"+str(inx)+"_"+i.split('_')[2]), avg_img)
        
def aveBlur_(oriPath,kernel_size,savePath):       
    for i in os.listdir(oriPath):            
        rimg=cv2.imread(os.path.join(oriPath,i))
        avg_img=cv2.blur(rimg,kernel_size)
        inx=int(i.split('_')[1])+(len(brightNum)*2) #依序加0-->10  
        # print(i.split('_')[0],i.split('_')[1],inx)
        cv2.imwrite(os.path.join(savePath,i.split('_')[0]+"_"+str(inx)+"_"+i.split('_')[2]), avg_img)
       
        
def GaussBlur(oriPath,kernel_size,sigma,savePath):
    for i in os.listdir(oriPath):
        rimg=cv2.imread(os.path.join(oriPath,i))
        GB_img=cv2.GaussianBlur(rimg,kernel_size,sigma) #依序加0-->15
        inx=int(i.split('_')[1])+(len(brightNum)*3)
        # print(i.split('_')[0],i.split('_')[1],inx)
        cv2.imwrite(os.path.join(savePath,i.split('_')[0]+"_"+str(inx)+"_"+i.split('_')[2]), GB_img)
        
        
        