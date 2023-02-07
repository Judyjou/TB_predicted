# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 01:54:12 2022

@author: Judy
"""

modelName="GoogLeNet_STTI_o2s8_SGD_TBX11K_5" #224
#圖片做模糊、(平均模糊、高斯模糊0.5，1，1.5，2.0)
#圖片調整亮暗(利用addweighted)
#check Use GPU
import tensorflow as tf
print(tf.__version__)
print("Use GPU devices:")
print(tf.config.list_physical_devices('GPU'))

  
#count model training needed time
import time 
start=time.process_time()
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import to_categorical
import shutil
import math,os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img #資料增強   ImageDataGenerator : 利用現有的資料經過旋轉、翻轉、縮放…等方式增加更多的訓練資料。
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal #He 正態分布初始化器。 he正態分布初始化方法，参数由0均值，標準差為sqrt(2 / fan_in) 的正態分布產生，其中fan_in權重張量的扇入
from tensorflow.keras.layers import Dense,Input,add,Activation,Lambda,concatenate,Flatten
from tensorflow.keras.layers import Conv2D,AveragePooling2D,GlobalAveragePooling2D,Dropout,MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import optimizers,regularizers
from tensorflow.keras.callbacks import LearningRateScheduler,TensorBoard #TensorBoard 視覺化呈現
from sklearn.metrics import accuracy_score,recall_score,precision_score,multilabel_confusion_matrix,roc_auc_score
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import cv2
import pandas as pd   
from tensorflow.keras.models import Sequential 
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from extraFunction.divideData import divide_data,valid_data
from extraFunction.GanImages import setBright,aveBlur,GaussBlur,aveBlur_,GaussBlur_
from extraFunction.ImageProcessing import cut_letterboxing,squImg,avgBlur,BLUE_EDGE,resizeImg,ORI_EDGE
from extraFunction.WriteMataData import write_csv,addData,CSVAdd_ArrImage
from extraFunction.googLeNet import GoogLeNet
from extraFunction.plotProcessing import show_train_history_acc,show_train_history_loss
from extraFunction.ClfMetrics import ClfMetrics
import random


def make_file(Folder):
    if not os.path.isdir(Folder):  #如果資料夾不存在就建立
        os.mkdir(Folder)


if __name__=="__main__":
    # print("start!")   
    # print(os.getcwd())
    #選擇哪一顆GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="0" 
    #定義起始檔案路徑位置
    basePath="D:\TB_py"    
    #=================================== 定義圖片起始路徑=================================== 
    #Make new file for SHZ file
    DataBase=os.path.join(basePath,"stti_DataBase_divide")
    make_file(DataBase)
    
    save_trainingProcessImg=os.path.join(DataBase,'save_trainingProcessImg')
    make_file(save_trainingProcessImg)
    
    #define path way
    dataset=os.path.join(basePath,"Alldataset")   
    
    # szData=os.path.join(dataset,"ChinaSet_AllFiles")
    # szData_normal=os.path.join(szData,'Normal')
    # szData_TB=os.path.join(szData,'TB')
    
    # Tung_Data=os.path.join(dataset,"Tung_XRay")
    # Tung_Data_normal=os.path.join(Tung_Data,'Normal')
    # Tung_Data_TB=os.path.join(Tung_Data,'TB')
    
    TBX11KData=os.path.join(dataset,"TBX11K")
    TBX11KData_normal=os.path.join(TBX11KData,'Normal')
    TBX11KData_TB=os.path.join(TBX11KData,'TB')
    
    # IndiaData=os.path.join(dataset,"India_DA")
    # IndiaData_normal=os.path.join(IndiaData,'Normal')
    # IndiaData_TB=os.path.join(IndiaData,'TB')
    
    save_trainingProcessImg=os.path.join(DataBase,'save_trainingProcessImg')
    make_file(save_trainingProcessImg)
    print("顯示原始圖片路徑:")
    # print(szData_normal,szData_TB,Tung_Data_normal,Tung_Data_TB,TBX11KData_normal,TBX11KData_TB,IndiaData_normal,IndiaData_TB)
    print(TBX11KData_normal,TBX11KData_TB)
    
    #===================================分割檔案 train/test/valid ===================================
    #定義分割檔資料路徑
    #divide normal shz,tung  data in Train80% Test20%  Normal and TB
    
    #---------------trainDataPath------------------------
    trainData=os.path.join(DataBase,"trainData")
    make_file(trainData)
       
    # shzTrain_nor=os.path.join(trainData,"shzTrain_nor")
    # shzTrain_TB=os.path.join(trainData,"shzTrain_TB")
    # tungTrain_nor=os.path.join(trainData,"tungTrain_nor")
    # tungTrain_TB=os.path.join(trainData,"tungTrain_TB")
    TBX11KTrain_nor=os.path.join(trainData,"TBX11KTrain_nor")
    TBX11KTrain_TB=os.path.join(trainData,"TBX11KTrain_TB")
    # indiaTrain_nor=os.path.join(trainData,"indiaTrain_nor")
    # indiaTrain_TB=os.path.join(trainData,"indiaTrain_TB")
    # trainDataLi=[shzTrain_nor,shzTrain_TB,tungTrain_nor,tungTrain_TB,TBX11KTrain_nor,TBX11KTrain_TB,indiaTrain_nor,indiaTrain_TB]
    trainDataLi=[TBX11KTrain_nor,TBX11KTrain_TB]
    
    
    # for i in trainDataLi:
    #     make_file(i)
       
    #---------------testDataPath------------------------
    testData=os.path.join(DataBase,"testData")
    make_file(testData)   
       
    # shzTest_nor=os.path.join(testData,"shzTest_nor")
    # shzTest_TB=os.path.join(testData,"shzTest_TB")
    # tungTest_nor=os.path.join(testData,"tungTest_nor")
    # tungTest_TB=os.path.join(testData,"tungTest_TB")
    TBX11KTest_nor=os.path.join(testData,"TBX11KTest_nor")
    TBX11KTest_TB=os.path.join(testData,"TBX11KTest_TB")
    # indiaTest_nor=os.path.join(testData,"indiaTest_nor")
    # indiaTest_TB=os.path.join(testData,"indiaTest_TB")
    # testDataLi=[shzTest_nor,shzTest_TB,tungTest_nor,tungTest_TB,TBX11KTest_nor,TBX11KTest_TB,indiaTest_nor,indiaTest_TB]
    testDataLi=[TBX11KTest_nor,TBX11KTest_TB]
    
    # for i in testDataLi:
    #     make_file(i)
    
    
    # #divide data train 80% / test 20% (換模型跑時需註解)
    # divide_data(szData_normal,shzTrain_nor,shzTest_nor)
    # divide_data(szData_TB,shzTrain_TB,shzTest_TB)
    # divide_data(Tung_Data_normal,tungTrain_nor,tungTest_nor)
    # divide_data(Tung_Data_TB,tungTrain_TB,tungTest_TB)
    # divide_data(TBX11KData_normal,TBX11KTrain_nor,TBX11KTest_nor)
    # divide_data(TBX11KData_TB,TBX11KTrain_TB,TBX11KTest_TB)
    # divide_data(IndiaData_normal,indiaTrain_nor,indiaTest_nor)
    # divide_data(IndiaData_TB,indiaTrain_TB,indiaTest_TB)
    

    #=================================== 訓練資料加入擴增影像 ===================================
    # GanImage_shz=os.path.join(trainData,'GanImage_shz')
    # make_file(GanImage_shz)
    
    # GanImage_Tung=os.path.join(trainData,'GanImage_Tung')
    # make_file(GanImage_Tung) 
    
    GanImage_TBX11K=os.path.join(trainData,'GanImage_TBX11K')
    # make_file(GanImage_TBX11K) 
    
    # GanImage_India=os.path.join(trainData,'GanImage_India')
    # make_file(GanImage_India) 
        
    # brightNum_shz=[0.8,1.0,1.2]
    # SHZli=[shzTrain_nor,shzTrain_TB]
    
    # idx=0
    # for i in brightNum_shz:
    #     for j in SHZli:
    #         setBright(j,i,idx,GanImage_shz)    
    #     idx+=1
            
    # brightNum_tung=[0.9,1.0,1.2]
    # Tungli=[tungTrain_nor,tungTrain_TB]
    
    # idx=0
    # for i in brightNum_tung:
    #     for j in Tungli:
    #         setBright(j,i,idx,GanImage_Tung)    
    #     idx+=1
        
    # brightNum_TBX11K=[0.9,1.0,1.1]
    # TBX11Kli=[TBX11KTrain_TB]
    
    # idx=0
    # for i in brightNum_TBX11K:
    #     for j in TBX11Kli:
    #         setBright(j,i,idx,GanImage_TBX11K)    
    #     idx+=1
        
    # brightNum_India=[0.8,0.9,1.0,1.1,1.2]
    # Indiali=[indiaTrain_nor,indiaTrain_TB]
    
    # idx=0
    # for i in brightNum_India:
    #     for j in Indiali:
    #         setBright(j,i,idx,GanImage_India)    
    #     idx+=1
    
    
    #=========================== 10種模糊挑1種使用 ===========================
    # shzGan=os.path.join(trainData,'shzGan')
    # make_file(shzGan)    
    # aveBlur(GanImage_shz,(3,3),brightNum_shz,shzGan)      
    
    # TungGan=os.path.join(trainData,'TungGan')
    # make_file(TungGan)    
    # aveBlur(GanImage_Tung,(3,3),brightNum_tung,TungGan) 

    TBX11KGan=os.path.join(trainData,'TBX11KGan')
    # make_file(TBX11KGan)    
    # aveBlur(GanImage_TBX11K,(3,3),brightNum_TBX11K,TBX11KGan)

    # indiaGan=os.path.join(trainData,'indiaGan')
    # make_file(indiaGan)    
    # aveBlur(GanImage_India,(3,3),brightNum_India,indiaGan)
    # aveBlur_(GanImage_India,(5,5),brightNum_India,indiaGan)
    # GaussBlur(GanImage_India,(5,5),0.5,brightNum_India,indiaGan)
    # GaussBlur_(GanImage_India,(3,3),2.0,brightNum_India,indiaGan)
    
    #將確定需要的圖併在同一資料夾
    # SHZGanTrainImg=os.path.join(trainData,"SHZGanTrainImg")
    # make_file(SHZGanTrainImg)
    # for i in os.listdir(GanImage_shz):
    #     shutil.copy(os.path.join(GanImage_shz,i),SHZGanTrainImg)
    # random.seed(10)
    # select_data=random.sample(os.listdir(shzGan),k=1128)
    # for i in select_data:
    #     shutil.copy(os.path.join(shzGan,i),SHZGanTrainImg)   
    
    # TungGanTrainImg=os.path.join(trainData,"TungGanTrainImg")
    # make_file(TungGanTrainImg)
    # for i in os.listdir(GanImage_Tung):
    #     shutil.copy(os.path.join(GanImage_Tung,i),TungGanTrainImg)
    # random.seed(10)
    # select_data1=random.sample(os.listdir(TungGan),k=1014)
    # for i in select_data1:
    #     shutil.copy(os.path.join(TungGan,i),TungGanTrainImg)
        
    TBX11KGanTrainImg=os.path.join(trainData,"TBX11KGanTrainImg")
    # make_file(TBX11KGanTrainImg)
    # random.seed(10)
    # select_data2=random.sample(os.listdir(TBX11KData_normal),k=1302)
    # for i in select_data2:
    #     shutil.copy(os.path.join(TBX11KData_normal,i),TBX11KGanTrainImg)
    # for i in os.listdir(TBX11KTrain_TB):
    #     shutil.copy(os.path.join(TBX11KTrain_TB,i),TBX11KGanTrainImg)        
    # select_data3=random.sample(os.listdir(TBX11KGan),k=662)
    # for i in select_data3:
    #     shutil.copy(os.path.join(TBX11KGan,i),TBX11KGanTrainImg)
        
    # IndiaGanTrainImg=os.path.join(trainData,"IndiaGanTrainImg")
    # make_file(IndiaGanTrainImg)
    # for i in os.listdir(GanImage_India):
    #     shutil.copy(os.path.join(GanImage_India,i),IndiaGanTrainImg)
    # random.seed(10)
    # select_data4=random.sample(os.listdir(indiaGan),k=1984)
    # for i in select_data4:
    #     shutil.copy(os.path.join(indiaGan,i),IndiaGanTrainImg)

   #  #===================================CSV mataData檔(檔名;標籤;原始圖片大小)===================================
    #train
    # write_csv(SHZGanTrainImg,"GanTrainMataData_Gan_SHZ",trainData)
    # TrainData=pd.read_csv(os.path.join(trainData,"GanTrainMataData_Gan_SHZ.csv"),encoding= 'utf-8')
    # write_csv(TBX11KGanTrainImg,"GanTrainMataData_Gan_TBX11K",trainData)
    TrainData=pd.read_csv(os.path.join(trainData,"GanTrainMataData_Gan_TBX11K.csv"),encoding= 'utf-8')
    # addData(TungGanTrainImg,TrainData,trainData,"GanTrainMataData_Gan_ST")
    # TrainData=pd.read_csv(os.path.join(trainData,"GanTrainMataData_Gan_ST.csv"),encoding= 'utf-8')
    # addData(TBX11KGanTrainImg,TrainData,trainData,"GanTrainMataData_Gan")
    # TrainData=pd.read_csv(os.path.join(trainData,"GanTrainMataData_Gan.csv"),encoding= 'utf-8')
    # addData(IndiaGanTrainImg,TrainData,trainData,"GanTrainMataData_Gan")
    # TrainData=pd.read_csv(os.path.join(trainData,"GanTrainMataData_Gan.csv"),encoding= 'utf-8')
    print(TrainData.head())
        
    #test
    # write_csv(shzTest_nor,"TestMataData_SHZ",testData)
    # TestData=pd.read_csv(os.path.join(testData,"TestMataData_SHZ.csv"),encoding= 'utf-8')
    # addData(shzTest_TB,TestData,testData,"TestMataData_SHZ")
    # TestData=pd.read_csv(os.path.join(testData,"TestMataData_SHZ.csv"),encoding= 'utf-8')
    # write_csv(TBX11KTest_nor,"TestMataData_TBX11K",testData)
    # TestData=pd.read_csv(os.path.join(testData,"TestMataData_TBX11K.csv"),encoding= 'utf-8')
    # addData(TBX11KTest_TB,TestData,testData,"TestMataData_TBX11K")
    TestData=pd.read_csv(os.path.join(testData,"TestMataData_TBX11K.csv"),encoding= 'utf-8')
    # addData(tungTest_nor,TestData,testData,"TestMataData_ST")
    # TestData=pd.read_csv(os.path.join(testData,"TestMataData_ST.csv"),encoding= 'utf-8')
    # addData(TBX11KTest_nor,TestData,testData,"TestMataData")
    # TestData=pd.read_csv(os.path.join(testData,"TestMataData.csv"),encoding= 'utf-8')
    # addData(TBX11KTest_TB,TestData,testData,"TestMataData")
    # TestData=pd.read_csv(os.path.join(testData,"TestMataData.csv"),encoding= 'utf-8')
    # addData(indiaTest_nor,TestData,testData,"TestMataData")
    # TestData=pd.read_csv(os.path.join(testData,"TestMataData.csv"),encoding= 'utf-8')
    # addData(indiaTest_TB,TestData,testData,"TestMataData")
    # TestData=pd.read_csv(os.path.join(testData,"TestMataData.csv"),encoding= 'utf-8')
    print(TestData.head()) 
    
   #  #========================== 開始將圖片前處理 1.先將多餘的邊裁切===============
    CL_allTrainImage=os.path.join(trainData,"CL_allTrainImage_TBX11K")
    # make_file(CL_allTrainImage)   
    CL_allTestImage=os.path.join(testData,"CL_allTrainImage_TBX11K")
    # make_file(CL_allTestImage)
    
    # allGanTrainLi=[SHZGanTrainImg,TungGanTrainImg,TBX11KGanTrainImg,IndiaGanTrainImg]
    allGanTrainLi=[TBX11KGanTrainImg]
    
    # for i in allGanTrainLi:
    #     cut_letterboxing(i, CL_allTrainImage) 
   
    # for j in testDataLi:
    #     cut_letterboxing(j, CL_allTestImage)

   #  #===================================#開始將圖片前處理 2.補邊變正方形===================================
    squ_allTrainImage=os.path.join(trainData,"squ_allTrainImage_TBX11K")
    # make_file(squ_allTrainImage) 
    
    squ_allTestImage=os.path.join(testData,"squ_allTestImage_TBX11K")
    # make_file(squ_allTestImage)    
       
    # squImg(CL_allTrainImage,squ_allTrainImage)      
    # squImg(CL_allTestImage, squ_allTestImage)
    
   #  #===================================#除擴增資料其餘皆圖片前處理 (邊緣提取/CLAHE)vs(Blur)=================================== 
    BlurTrain=os.path.join(trainData,"BlurTrain_BK5_TBX11K")
    make_file(BlurTrain)
    BlurTest=os.path.join(testData,"BlurTest_BK5_TBX11K")
    make_file(BlurTest)
   
    # avgBlur(squ_allTrainImage,(5,5),BlurTrain)
    # avgBlur(squ_allTestImage,(5,5),BlurTest)
   
    # CLAHETrain=os.path.join(trainData,"CLAHETrain")
    # make_file(CLAHETrain)
    # CLAHETest=os.path.join(testData,"CLAHETest")
    # make_file(CLAHETest)
   
    # CLAHE(squ_TrainImage,CLAHETrain)
    # CLAHE(squ_allTestImage,CLAHETest)
    
   #  #==================================將CLAHE/Blur 做 sobel/scharr/laplace/canny ==================================
    edgeTrain=os.path.join(trainData,"edgeTrain_Sb3_TBX11K")
    # make_file(edgeTrain)
    edgeTest=os.path.join(testData,"edgeTest_Sb3_TBX11K")
    # make_file(edgeTest)
   

    # BLUE_EDGE(squ_allTrainImage,edgeTrain)
    # BLUE_EDGE(BlurTest,edgeTest)
    
    #===================================#開始將圖片前處理 3.轉成相同大小===================================
    edit_train=os.path.join(trainData,"edit_train_TBX11K")
    # make_file(edit_train)    
    edit_test=os.path.join(testData,"edit_test_TBX11K")
    # make_file(edit_test)    
   
    #原圖也需resize
    oriImage_train=os.path.join(trainData,"oriImage_train_TBX11K")   
    # make_file(oriImage_train)  
    oriImage_test=os.path.join(testData,"oriImage_test_TBX11K")   
    # make_file(oriImage_test)      
  
    # resizeImg(edgeTrain, edit_train)
    # resizeImg(edgeTest, edit_test)

    # resizeImg(squ_allTrainImage, oriImage_train)
    # resizeImg(squ_allTestImage, oriImage_test)
    
    #===================================疊圖===================================
    ResAddweTrain=os.path.join(trainData,"ResAddweTrain_sobel_o2s8_TBX11K")
    make_file(ResAddweTrain)
    ResAddweTest=os.path.join(testData,"ResAddweTest_sobel_o2s8_TBX11K")
    make_file(ResAddweTest)

   
    # ORI_EDGE(oriImage_train,edit_train,0.2,0.8,ResAddweTrain)
    # ORI_EDGE(oriImage_test,edit_test,0.2,0.8,ResAddweTest)
   
   #  #=================================== 開始將圖片前處理結束===================
   #  #=================================== 加入矩陣到mataData(圖片矩陣&處理完後圖片大小)==========================
    CSVAdd_ArrImage(ResAddweTrain,TrainData,trainData,"TrainArr_So2s8_Tung")
    CSVAdd_ArrImage(ResAddweTest,TestData,testData,"TestArr_So2s8_Tung")
    
   #  #===================================定義訓練資料===================================
    img_train,img_test,label_train,label_test=TrainData["Image_array"].values,TestData["Image_array"].values,TrainData["Label"].values,TestData["Label"].values
    img_train=np.asarray(np.asarray(img_train.tolist()).tolist())
    img_test=np.asarray(np.asarray(img_test.tolist()).tolist())
    print("img_train: ",img_train.shape)
    print("img_test: ",img_test.shape)
  
    print("label_train: ",label_train.shape)
    print("label_test: ",label_test.shape)
  
    #256*256轉乘1維數字向量 且儲存為 float /255 做正規劃 直落在0~1之間
    nimg_train=img_train.reshape((2604, 224, 224, 3)).astype('float32')/255
    nimg_test=img_test.reshape((920, 224, 224, 3)).astype('float32') /255
    label_train_oneHot=to_categorical(label_train)
    label_test_oneHot=to_categorical(label_test)
  
    print("train: ",nimg_train.shape,label_train_oneHot.shape)
    print("test: ",nimg_test.shape,label_test_oneHot.shape)    

    # ===================================建立模型 googLeNet ===================================
    SaveModel=os.path.join(DataBase,"SaveModel")
    make_file(SaveModel)
    
    full_model = GoogLeNet()

    # full_model.summary()
    # epochs=120
    # batch_size = 82
    # initial_lrate=0.001
    
    # def decay(epoch,steps=100):
    #     initial_lrate=0.001
    #     drop=0.96
    #     epochs_drop=8
    #     lrate=initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    #     return lrate
    # lr_sc = LearningRateScheduler(decay, verbose=1)
    # earlyStop=EarlyStopping(monitor='val_loss',patience=10, mode='auto',verbose=2)
    # # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=2, min_lr=0.0001,verbose=2)
    # full_model.compile(loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'],loss_weights=[1, 0.4, 0.6],optimizer=keras.optimizers.SGD(lr=initial_lrate, momentum=0.9, nesterov=False),metrics=['accuracy'])
    # history = full_model.fit(nimg_train,[label_train_oneHot,label_train_oneHot,label_train_oneHot],validation_data = (nimg_test,[label_test_oneHot,label_test_oneHot,label_test_oneHot]),batch_size=batch_size,epochs=epochs,verbose=2,callbacks=[lr_sc,earlyStop])

    full_model.summary()
    epochs=200
    batch_size = 64
    initial_lrate=0.001
    
    def decay(epoch,steps=100):
        initial_lrate=0.001
        drop=0.97
        epochs_drop=16
        lrate=initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    lr_sc = LearningRateScheduler(decay, verbose=1)   
    earlyStop=EarlyStopping(monitor='val_loss',patience=13, mode='auto',verbose=2)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=2, min_lr=0.0001,verbose=2)
    full_model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(lr=initial_lrate),metrics=['accuracy'])
    # full_model.compile(loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'],loss_weights=[1, 0.3, 0.2],optimizer=keras.optimizers.SGD(lr=initial_lrate, momentum=0.9, nesterov=False),metrics=['accuracy'])
    history = full_model.fit(nimg_train,label_train_oneHot,validation_data = (nimg_test,label_test_oneHot),batch_size=batch_size,epochs=epochs,verbose=2,callbacks=[lr_sc,earlyStop])#
 
         
    try:
        full_model.save_weights(os.path.join(SaveModel,modelName+'.h5'))
        print("Model input Success!!")
    except:
        print("Model input fail, create new one.!!")
        
    full_model.save_weights(os.path.join(SaveModel,modelName+'.h5'))
    print("Save model success.")
    
    #繪出訓練過程圖
    show_train_history_acc(history,'dense_4_accuracy','val_dense_4_accuracy',save_trainingProcessImg,modelName) 
    show_train_history_loss(history,'dense_4_loss','val_dense_4_loss',save_trainingProcessImg,modelName)
    
    # print("=================Test acc/loss =====================")
    # scores_test = full_model.evaluate(nimg_test, label_test_oneHot, verbose=1)
    # print('Test loss:', scores_test[0])
    # print('Test accuracy:', scores_test[1])
    
    # print("=================Train acc/loss========================")    
    # scores_train = full_model.evaluate(nimg_train, label_train_oneHot, verbose=1)
    # print('Train loss:', scores_train[0])
    # print('Train accuracy:', scores_train[1])
    
    #====顯示結果=====     
    from sklearn.metrics import matthews_corrcoef
    
    ResultTxt=os.path.join(DataBase,"ResultTxt")
    make_file(ResultTxt)
    #========== train===============
    print("----------- Train ------------")
    wResult=os.path.join(ResultTxt,"Result_"+modelName+".txt")
    owResult=open(wResult,'w')    
    #顯示每一張training預測的類別標籤
    train_predict_result=np.argmax(full_model.predict(img_train)[2],axis=-1)
    # print("Train預測標籤結果: ",train_predict_result)
    # @@@@@@@@@@@@@@@@@@@@@@@ 刪除變數 @@@@@@@@@@@@@@@@@@@@@@@
    del img_train
    
    # 顯示混淆矩陣    
    train_conm=pd.crosstab(label_train.ravel(),train_predict_result,rownames=['label'],colnames=['predict'])
    print("Train confusion matrix: ")
    print(train_conm)
    owResult.write("Train confusion matrix: \n")
    owResult.write(train_conm.to_string())
    owResult.write("\n")
    
    TrainClfMetrics={}    
    AllTrainClfMetrics=["Train Accuracy: ","Train precision: ","Train recall: ","Train F1 Score: ","Train Sensitivity: ","Train Specificity: ","Train MCC: "]
    
   #各項指標值計算
    train_clf_metrics = ClfMetrics(label_train.ravel(), train_predict_result)    
    print("Train 準確度: ",round(train_clf_metrics.accuracy_score(),4))
    print("Train 精確度: ",round(train_clf_metrics.precision_score(),4))
    print("Train 召回率: ",round(train_clf_metrics.recall_score(),4))    
    print("Train F1 Score: ",round(train_clf_metrics.f1_score(),4))
    print("Train Sensitivity:",round(train_clf_metrics.Sensitivity_score(),4))
    print("Train Specificity:",round(train_clf_metrics.Specificity_score(),4))
    print("Train MCC: ",round(matthews_corrcoef(label_train,train_predict_result),4))# matthews_corrcoef(實際類別,預測類別)    
    TrainResultList=[round(train_clf_metrics.accuracy_score(),4),round(train_clf_metrics.precision_score(),4),round(train_clf_metrics.recall_score(),4),round(train_clf_metrics.f1_score(),4),round(train_clf_metrics.Sensitivity_score(),4),round(train_clf_metrics.Specificity_score(),4),round(matthews_corrcoef(label_train,train_predict_result),4)]
    
    for i in range(len(AllTrainClfMetrics)): 
        for j in range(len(TrainResultList)):
            if i==j:
                TrainClfMetrics[AllTrainClfMetrics[i]]=TrainResultList[j]            
    for k,v in TrainClfMetrics.items():
        owResult.write(str(k)+str(v)+'\n')         
    
    
    #==========Test===============
    print("----------- Test ------------")
    #顯示每一張testing預測的類別標籤
    test_predict_result=np.argmax(full_model.predict(img_test)[2],axis=-1)
    # print("Tset預測標籤結果: ",test_predict_result)   
    # @@@@@@@@@@@@@@@@@@@@@@@ 刪除變數 @@@@@@@@@@@@@@@@@@@@@@@
    del img_test     
    
    # 顯示混淆矩陣    
    test_conm=pd.crosstab(label_test.ravel(),test_predict_result,rownames=['label'],colnames=['predict'])
    print("Test confusion matrix: ")
    print(test_conm)
    owResult.write("\n")
    owResult.write("Test confusion matrix: \n")
    owResult.write(test_conm.to_string())
    owResult.write("\n")
    TestClfMetrics={}
    AllTestClfMetrics=["Test Accuracy: ","Test precision: ","Test recall: ","Test F1 Score: ","Test Sensitivity: ","Test Specificity: ","Test MCC: "]
    
    
    #各項指標值計算並取到小數點第2位
    test_clf_metrics = ClfMetrics(label_test.ravel(), test_predict_result)
    print("Test 準確度: ",round(test_clf_metrics.accuracy_score(),4))
    print("Test 精確度: ",round(test_clf_metrics.precision_score(),4))
    print("Test 召回率: ",round(test_clf_metrics.recall_score(),4))
    print("Test F1 Score: ",round(test_clf_metrics.f1_score(),4))  
    print("Test Sensitivity:",round(test_clf_metrics.Sensitivity_score(),4))
    print("Test Specificity:",round(test_clf_metrics.Specificity_score(),4))
    print("Test MCC: ",round(matthews_corrcoef(label_test,test_predict_result),4))
    TestResultList=[round(test_clf_metrics.accuracy_score(),4),round(test_clf_metrics.precision_score(),4),round(test_clf_metrics.recall_score(),4),round(test_clf_metrics.f1_score(),4),round(test_clf_metrics.Sensitivity_score(),4),round(test_clf_metrics.Specificity_score(),4),round(matthews_corrcoef(label_test,test_predict_result),4)]
    
    for i in range(len(AllTestClfMetrics)): 
        for j in range(len(TestResultList)):
            if i==j:
                TestClfMetrics[AllTestClfMetrics[i]]=TestResultList[j]            
    for k,v in TestClfMetrics.items():
        owResult.write(str(k)+str(v)+'\n') 
        
    
    owResult.close()
    
    #close GPU
    from numba import cuda
    cuda.select_device(0)
    cuda.close()
    
    #釋放記憶體
    import gc
    del full_model
    gc.collect() #14982  10913
    
    del history  
    gc.collect()  #10912