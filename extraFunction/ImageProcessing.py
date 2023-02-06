# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 15:47:46 2022

@author: Judy
"""

import os
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
from PIL import Image

def cut_letterboxing(mainPath,storePath):
    for i in os.listdir(mainPath):
        ReadImg=cv2.imread(os.path.join(mainPath,i))#讀圖
        img=cv2.medianBlur(ReadImg,5)#中值濾波，kernel size=5 去掉黑邊可能會有躁點干擾
        edit=cv2.threshold(img,5,255,cv2.THRESH_BINARY)
        binary_img=edit[1]
        binary_image= cv2.cvtColor(binary_img,cv2.COLOR_BGR2GRAY)
        edges_y, edges_x = np.where(binary_image==255)
        bottom=min(edges_y)
        top=max(edges_y)
        left = min(edges_x)
        right = max(edges_x)
        height = top - bottom
        width = right - left
        res_image = ReadImg[bottom:bottom+height, left:left+width]
        cv2.imwrite(os.path.join(storePath, i), res_image)
        

 
def squImg(mainPath,storePath):
    for i in os.listdir(mainPath):
        image=Image.open(os.path.join(mainPath,i))
        image=image.convert('RGB')
        w,h=image.size
        background=Image.new('RGB',size=(max(w,h),max(w,h)),color=(0,0,0)) #創建背景圖w,h比較後最大的那者
        # length=int(abs(w-h)//2)#726 寬的中間點
        length=0  
        box=(length,0) if w <h else(0,-length)
        background.paste(image,box)
        image_data=background.resize((224,224))
        background.save(os.path.join(storePath,i))
        
#除擴增資料其餘皆圖片前處理 (CLAHE)vs(Blur) 邊緣提取

def avgBlur(oriPath,kernel_size,savePath):
    for i in os.listdir(oriPath):
        rimg=cv2.imread(os.path.join(oriPath,i))
        avg_img=cv2.blur(rimg,kernel_size)
        cv2.imwrite(os.path.join(savePath,i),avg_img)
        
def CLAHE(mainPath,savePalce):
    for i in os.listdir(mainPath):
        Gimg=cv2.imread(os.path.join(mainPath,i),cv2.IMREAD_GRAYSCALE)
        CLAHE=cv2.createCLAHE(clipLimit=0.01)
        CLAHE_img=CLAHE.apply(Gimg)
        cv2.imwrite(os.path.join(savePalce,i),CLAHE_img)
        
def GaBlur(mainPath,kernel_size,sigma,savePath):
    for i in os.listdir(mainPath):
        Gimg=cv2.imread(os.path.join(mainPath,i))       
        GB_img=cv2.GaussianBlur(Gimg,kernel_size,sigma)
        cv2.imwrite(os.path.join(savePath,i),GB_img)
        
def BLUE_EDGE(oriFile,savePalce):
    for ori in os.listdir(oriFile):
        rimg=cv2.imread(os.path.join(oriFile,ori),cv2.IMREAD_GRAYSCALE)
        sobelx=cv2.Sobel(rimg,cv2.CV_64F,dx=1,dy=0,ksize=3)
        sobely=cv2.Sobel(rimg,cv2.CV_64F,dx=0,dy=1,ksize=3)
        sobelx=cv2.convertScaleAbs(sobelx)
        sobely=cv2.convertScaleAbs(sobely)
        addimgs=cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)            
        cv2.imwrite(os.path.join(savePalce,ori),addimgs)
        
def CLAHE_EDGE(oriFile,savePalce):
    for ori in os.listdir(oriFile):
        rimg=cv2.imread(os.path.join(oriFile,ori),cv2.IMREAD_GRAYSCALE)
        canny=cv2.Canny(rimg,30, 130)
        cv2.imwrite(os.path.join(savePalce,ori),canny)

height=224
width=224
def resizeImg(oriFile,saveFile):
    for ori in os.listdir(oriFile):
        rimg=cv2.imread(os.path.join(oriFile,ori))
        R_img=cv2.resize(rimg,(height,width))
        cv2.imwrite(os.path.join(saveFile,ori),R_img)
        
#疊圖
def ORI_EDGE(oriFile,edgeFile,oriWe,edgeWe,savePalce):
    for ori in os.listdir(oriFile):
        r_ori=cv2.imread(os.path.join(oriFile,ori))
        for edge in os.listdir(edgeFile):
            edge_img=cv2.imread(os.path.join(edgeFile,edge))
            if ori==edge:
                addimgs=cv2.addWeighted(r_ori, oriWe, edge_img, edgeWe, 0)
                cv2.imwrite(os.path.join(savePalce,ori),addimgs)
print("success sobel_ori addweight")  

