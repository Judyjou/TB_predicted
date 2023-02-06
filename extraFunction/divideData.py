# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:47:32 2022

@author: Judy
"""
import os
import shutil
import random

#取0.8為train ;剩下的0.2為test
def divide_data(oriFilePath,trainPath,testPath):
    random.seed(10)
    FileNum=round(len(os.listdir(oriFilePath))*0.8) 
    select_data=random.sample(os.listdir(oriFilePath),k=FileNum)
    for i in select_data:
        shutil.copy(os.path.join(oriFilePath,i),trainPath)
    testLi=[x for x in os.listdir(oriFilePath) if x not in select_data]  
    for j in testLi:
        shutil.copy(os.path.join(oriFilePath,j),testPath)
        
def valid_data(oriFilePath,validPath):
    for i in os.listdir(oriFilePath):
        shutil.copy(os.path.join(oriFilePath,i),validPath)
        
def switch_data(oriPassFilePath,mainPath):
    for i in os.listdir(oriPassFilePath):
        shutil.copy(os.path.join(oriPassFilePath,i),mainPath)