# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 20:12:48 2022

@author: Judy
"""

import os
import shutil
import random

def divideIndia_data(oriFilePath,trainPath,testPath):
    random.seed(10)
    FileNum=round(len(os.listdir(oriFilePath))*0.8) 
    select_data=random.sample(os.listdir(oriFilePath),k=FileNum)
    for i in select_data:
        shutil.copy(os.path.join(oriFilePath,i),trainPath)
    testLi=[x for x in os.listdir(oriFilePath) if x not in select_data]  
    for j in testLi:
        shutil.copy(os.path.join(oriFilePath,j),testPath)
        
def dividData_NorTrain_test(oriFilePath,trainPath,tempPath,testPath):
    random.seed(10)
    Num_NorTrain=62
    Num_NorTest=15
    # Num_NorTrain=62
    # Num_NorTest=16
    select_TrainData=random.sample(os.listdir(oriFilePath),k=Num_NorTrain)
    for i in select_TrainData:
        shutil.copy(os.path.join(oriFilePath,i),trainPath)
    TempLi=[x for x in os.listdir(oriFilePath) if x not in select_TrainData]
    for j in TempLi:
        shutil.copy(os.path.join(oriFilePath,j),tempPath)
    # Num_NorTest=15
    select_TestData=random.sample(os.listdir(tempPath),k=Num_NorTest)
    for i in select_TestData:
        shutil.copy(os.path.join(tempPath,i),testPath)
        
def dividData_TBTrain_test(oriFilePath,trainPath,tempPath,testPath):
    random.seed(10)
    Num_TBTrain=62
    Num_TBTest=16
    select_TrainData=random.sample(os.listdir(oriFilePath),k=Num_TBTrain)
    for i in select_TrainData:
        shutil.copy(os.path.join(oriFilePath,i),trainPath)
    TempLi=[x for x in os.listdir(oriFilePath) if x not in select_TrainData]
    for j in TempLi:
        shutil.copy(os.path.join(oriFilePath,j),tempPath)
    # Num_NorTest=15
    select_TestData=random.sample(os.listdir(tempPath),k=Num_TBTest)
    for i in select_TestData:
        shutil.copy(os.path.join(tempPath,i),testPath)

def dividData_NorTrain_test__(oriFilePath,trainPath,tempPath,testPath):
    random.seed(10)
    Num_NorTrain=246
    Num_NorTest=61
    # Num_NorTrain=62
    # Num_NorTest=16
    select_TrainData=random.sample(os.listdir(oriFilePath),k=Num_NorTrain)
    for i in select_TrainData:
        shutil.copy(os.path.join(oriFilePath,i),trainPath)
    TempLi=[x for x in os.listdir(oriFilePath) if x not in select_TrainData]
    for j in TempLi:
        shutil.copy(os.path.join(oriFilePath,j),tempPath)
    # Num_NorTest=15
    select_TestData=random.sample(os.listdir(tempPath),k=Num_NorTest)
    for i in select_TestData:
        shutil.copy(os.path.join(tempPath,i),testPath)
        
def dividData_TBTrain_test__(oriFilePath,trainPath,tempPath,testPath):
    random.seed(10)
    Num_TBTrain=246
    Num_TBTest=61
    select_TrainData=random.sample(os.listdir(oriFilePath),k=Num_TBTrain)
    for i in select_TrainData:
        shutil.copy(os.path.join(oriFilePath,i),trainPath)
    TempLi=[x for x in os.listdir(oriFilePath) if x not in select_TrainData]
    for j in TempLi:
        shutil.copy(os.path.join(oriFilePath,j),tempPath)
    # Num_NorTest=15
    select_TestData=random.sample(os.listdir(tempPath),k=Num_TBTest)
    for i in select_TestData:
        shutil.copy(os.path.join(tempPath,i),testPath)
