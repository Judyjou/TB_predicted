# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 11:47:03 2022

@author: Judy
"""

modelName="VGG16_shz10000_o2s8_Tung" #224
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
# from tensorflow.keras.applications.resnet50 import ResNet50

# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications import VGG19
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
from extraFunction.GanImage_10000 import setBright,aveBlur,GaussBlur,aveBlur_
from extraFunction.ImageProcessing import cut_letterboxing,squImg,avgBlur,BLUE_EDGE,resizeImg,ORI_EDGE
from extraFunction.WriteMataData import write_csv,addData,CSVAdd_ArrImage
from extraFunction.VGG16 import VGG16Model
from extraFunction.plotProcessing import show_train_history_acc,show_train_history_loss
from extraFunction.ClfMetrics import ClfMetrics


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
    DataBase=os.path.join(basePath,"SHZ10000_DataBase")
    make_file(DataBase)
    
    save_trainingProcessImg=os.path.join(DataBase,'save_trainingProcessImg')
    make_file(save_trainingProcessImg)
    
    #define path way
    dataset=os.path.join(basePath,"Alldataset")
     
    szData=os.path.join(dataset,"ChinaSet_AllFiles")
    szData_normal=os.path.join(szData,'Normal')
    szData_TB=os.path.join(szData,'TB')
     
    Tung_Data=os.path.join(dataset,"Tung_XRay")
    Tung_Data_normal=os.path.join(Tung_Data,'Normal')
    Tung_Data_TB=os.path.join(Tung_Data,'TB')
    
    
    
    save_trainingProcessImg=os.path.join(DataBase,'save_trainingProcessImg')
    make_file(save_trainingProcessImg)
    print("顯示原始圖片路徑:")
    print(szData_normal,szData_TB,Tung_Data_normal,Tung_Data_TB)
    
    #===================================分割檔案 train/test/valid ===================================
    #定義分割檔資料路徑
    #divide normal shz,tung  data in Train80% Test20%  Normal and TB
    
    #---------------trainDataPath------------------------
    trainData=os.path.join(DataBase,"trainData")
    make_file(trainData)
   
    shzTrain=os.path.join(trainData,"shzTrain")  
    make_file(shzTrain)
       
    #---------------testDataPath------------------------
    testData=os.path.join(DataBase,"testData")
    make_file(testData)   
    
    shzTest=os.path.join(testData,"shzTest")
    make_file(shzTest)
   
    #---------------validDataPath------------------------
    validData=os.path.join(DataBase,"validData")
    make_file(validData)
    
    TungValid=os.path.join(validData,"TungValid")
    make_file(TungValid)
    
    # #divide data train 80% / test 20% (換模型跑時需註解)
    # divide_data(szData_normal,shzTrain,shzTest)
    # divide_data(szData_TB,shzTrain,shzTest)
    
    # valid_data(Tung_Data_normal,TungValid)
    # valid_data(Tung_Data_TB,TungValid)

    #=================================== 訓練資料加入擴增影像===================================
    GanImage=os.path.join(trainData,'GanImage')
    make_file(GanImage)
        
    trainDataLi=[shzTrain]
    brightNum=[0.8,0.9,1.0,1.1,1.2]
    
    # idx=0
    # for i in brightNum:
    #     for j in trainDataLi:
    #         setBright(j,i,idx,GanImage)    
    #     idx+=1
            
    #=========================== 10種模糊挑3種使用 ===========================
    ShzGan_10000=os.path.join(trainData,'ShzGan_10000')
    make_file(ShzGan_10000)    
    # aveBlur(GanImage,(3,3),ShzGan_10000)        
    # aveBlur_(GanImage,(5,5),ShzGan_10000)   
    # GaussBlur(GanImage,(5,5),0.5,ShzGan_10000)
    #==========================
    #===================================CSV mataData檔(檔名;標籤;原始圖片大小)===================================
    #train
    write_csv(GanImage,"GanTrainMataData_Gan",trainData)
    TrainData=pd.read_csv(os.path.join(trainData,"GanTrainMataData_Gan.csv"),encoding= 'utf-8')
    addData(ShzGan_10000,TrainData,trainData,"GanTrainMataData_Gan")
    TrainData=pd.read_csv(os.path.join(trainData,"GanTrainMataData_Gan.csv"),encoding= 'utf-8')
    print(TrainData.head())
    
    #test
    write_csv(shzTest,"TestMataData",testData)
    TestData=pd.read_csv(os.path.join(testData,"TestMataData.csv"),encoding= 'utf-8')
    print(TestData.head())
    
    #valid
    write_csv(TungValid, "ValidMataData", validData)
    ValidData=pd.read_csv(os.path.join(validData,"ValidMataData.csv"),encoding= 'utf-8')      
    print(ValidData.head())  
    
    #========================== 開始將圖片前處理 1.先將多餘的邊裁切===============
    CL_TrainImage=os.path.join(trainData,"CL_TrainImage")
    make_file(CL_TrainImage)   
    CL_trainGanImage=os.path.join(trainData,"GanImage_Gan") #不動
    make_file(CL_trainGanImage)
    
    CL_allTestImage=os.path.join(testData,"CL_allTestImage")
    make_file(CL_allTestImage)
    
    CL_allValidImage=os.path.join(validData,"CL_allValidImage")
    make_file(CL_allValidImage)
    
    
    # cut_letterboxing(GanImage, CL_TrainImage)
    # cut_letterboxing(ShzGan_10000, CL_trainGanImage) 
    
    # cut_letterboxing(shzTest, CL_allTestImage)    

    
    # ValidOriList=[Tung_Data_normal,Tung_Data_TB]
    # for i in range(len(ValidOriList)):
    #     cut_letterboxing(ValidOriList[i], CL_allValidImage) 
    
    #===================================#開始將圖片前處理 2.補邊變正方形===================================
    squ_TrainImage=os.path.join(trainData,"squ_TrainImage")
    make_file(squ_TrainImage) 
    squ_GanTrainImage=os.path.join(trainData,"squ_GanTrainImage_Gan")
    make_file(squ_GanTrainImage)    
    squ_allTestImage=os.path.join(testData,"squ_allTestImage")
    make_file(squ_allTestImage)    
    squ_allValidImage=os.path.join(validData,"squ_allValidImage")
    make_file(squ_allValidImage)
        
    # squImg(CL_TrainImage,squ_TrainImage)   
    # squImg(CL_trainGanImage,squ_GanTrainImage)   
    # squImg(CL_allTestImage, squ_allTestImage)
    # squImg(CL_allValidImage, squ_allValidImage)
    
    #===================================#除擴增資料其餘皆圖片前處理 (邊緣提取/CLAHE)vs(Blur)=================================== 
    BlurTrain=os.path.join(trainData,"BlurTrain_BK5")
    make_file(BlurTrain)
    BlurTest=os.path.join(testData,"BlurTest_BK5")
    make_file(BlurTest)
    BlurValid=os.path.join(validData,"BlurValid_BK5")
    make_file(BlurValid)
    
    # avgBlur(squ_TrainImage,(5,5),BlurTrain)
    # avgBlur(squ_allTestImage,(5,5),BlurTest)
    # avgBlur(squ_allValidImage,(5,5),BlurValid)
    
    # CLAHETrain=os.path.join(trainData,"CLAHETrain")
    # make_file(CLAHETrain)
    # CLAHETest=os.path.join(testData,"CLAHETest")
    # make_file(CLAHETest)
    # CLAHEValid=os.path.join(validData,"CLAHEValid")
    # make_file(CLAHEValid)
    
    # CLAHE(squ_TrainImage,CLAHETrain)
    # CLAHE(squ_allTestImage,CLAHETest)
    # CLAHE(squ_allValidImage,CLAHEValid)
    
    #==================================將CLAHE/Blur 做 sobel/scharr/laplace/canny ==================================
    edgeTrain=os.path.join(trainData,"edgeTrain_Sb3")
    make_file(edgeTrain)
    edgeTest=os.path.join(testData,"edgeTest_Sb3")
    make_file(edgeTest)
    edgeValid=os.path.join(validData,"edgeValid_Sb3")
    make_file(edgeValid)
    
    # BLUE_EDGE(BlurTrain,edgeTrain)
    # BLUE_EDGE(squ_GanTrainImage,edgeTrain)
    # BLUE_EDGE(BlurTest,edgeTest)
    # BLUE_EDGE(BlurValid,edgeValid)
    
    #===================================#開始將圖片前處理 3.轉成相同大小===================================
    edit_train=os.path.join(trainData,"edit_train")
    make_file(edit_train)    
    edit_test=os.path.join(testData,"edit_test")
    make_file(edit_test)    
    edit_valid=os.path.join(validData,"edit_valid")
    make_file(edit_valid)
    
    #原圖也需resize
    oriImage_train=os.path.join(trainData,"oriImage_train")   
    make_file(oriImage_train)  
    oriImage_test=os.path.join(testData,"oriImage_test")   
    make_file(oriImage_test)  
    oriImage_valid=os.path.join(validData,"oriImage_valid")   
    make_file(oriImage_valid)      
    
    # resizeImg(edgeTrain, edit_train)
    # resizeImg(edgeTest, edit_test)
    # resizeImg(edgeValid, edit_valid)

    # resizeImg(squ_TrainImage, oriImage_train)
    # resizeImg(squ_GanTrainImage, oriImage_train)
    # resizeImg(squ_allTestImage, oriImage_test)
    # resizeImg(squ_allValidImage, oriImage_valid) 
    
    #===================================疊圖===================================
    ResAddweTrain=os.path.join(trainData,"ResAddweTrain_sobel_o2s8")
    make_file(ResAddweTrain)
    ResAddweTest=os.path.join(testData,"ResAddweTest_sobel_o2s8")
    make_file(ResAddweTest)
    ResAddweValid=os.path.join(validData,"ResAddweValid_sobel_o2s8")
    make_file(ResAddweValid)
    
    # ORI_EDGE(oriImage_train,edit_train,0.2,0.8,ResAddweTrain)
    # ORI_EDGE(oriImage_test,edit_test,0.2,0.8,ResAddweTest)
    # ORI_EDGE(oriImage_valid,edit_valid,0.2,0.8,ResAddweValid)    
    #=================================== 開始將圖片前處理結束===================
    #=================================== 加入矩陣到mataData(圖片矩陣&處理完後圖片大小)==========================
    CSVAdd_ArrImage(ResAddweTrain,TrainData,trainData,"TrainArr_So2s8")
    CSVAdd_ArrImage(ResAddweTest,TestData,testData,"TestArr_So2s8")
    CSVAdd_ArrImage(ResAddweValid,ValidData,validData,"ValidArr_So2s8")
    
    #===================================定義訓練資料===================================
    img_train,img_test,label_train,label_test=TrainData["Image_array"].values,TestData["Image_array"].values,TrainData["Label"].values,TestData["Label"].values
    img_train=np.asarray(np.asarray(img_train.tolist()).tolist())
    img_test=np.asarray(np.asarray(img_test.tolist()).tolist())
    print("img_train: ",img_train.shape)
    print("img_test: ",img_test.shape)

    print("label_train: ",label_train.shape)
    print("label_test: ",label_test.shape)

    #256*256轉乘1維數字向量 且儲存為 float /255 做正規劃 直落在0~1之間
    nimg_train=img_train.reshape((9540, 224, 224, 3)).astype('float32')/255
    nimg_test=img_test.reshape((132, 224, 224, 3)).astype('float32') /255
    label_train_oneHot=to_categorical(label_train)
    label_test_oneHot=to_categorical(label_test)

    print("train: ",nimg_train.shape,label_train_oneHot.shape)
    print("test: ",nimg_test.shape,label_test_oneHot.shape)

    #===================================定義驗證資料===================================
    img_valid,label_valid=ValidData["Image_array"].values,ValidData["Label"].values
    
    
    img_valid=np.asarray(np.asarray(img_valid.tolist()).tolist())
    print("img_valid: ",img_valid.shape)
    
    nimg_valid=img_valid.reshape((614, 224, 224, 3)).astype('float32') /255
    
    label_valid_oneHot=to_categorical(label_valid)    
    print("valid: ",nimg_valid.shape,label_valid_oneHot.shape)
    
    
    # ===================================建立模型 googLeNet ===================================
    SaveModel=os.path.join(DataBase,"SaveModel")
    make_file(SaveModel)
    
    full_model = VGG16Model()

    full_model.summary()
    epochs=30
    batch_size = 10
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3, min_lr=0.00001,verbose=2)
    # model.fit(X_train, Y_train, callbacks=[reduce_lr])
    full_model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9, beta_2=0.999, epsilon=0.1,amsgrad=False,name="Adam"),metrics=['accuracy'])
    history = full_model.fit(nimg_train,label_train_oneHot,validation_data = (nimg_valid,label_valid_oneHot),batch_size=batch_size,epochs=epochs,verbose=2,callbacks=[reduce_lr])
        
    try:
        full_model.save_weights(os.path.join(SaveModel,modelName+'.h5'))
        print("Model input Success!!")
    except:
        print("Model input fail, create new one.!!")
        
    full_model.save_weights(os.path.join(SaveModel,modelName+'.h5'))
    print("Save model success.")
    
    #繪出訓練過程圖
    show_train_history_acc(history,'accuracy','val_accuracy',save_trainingProcessImg,modelName) 
    show_train_history_loss(history,'loss','val_loss',save_trainingProcessImg,modelName) 
    
    print("=================Test acc/loss =====================")
    scores_test = full_model.evaluate(nimg_test, label_test_oneHot, verbose=1)
    print('Test loss:', scores_test[0])
    print('Test accuracy:', scores_test[1])
    
    print("=================Train acc/loss========================")    
    scores_train = full_model.evaluate(nimg_train, label_train_oneHot, verbose=1)
    print('Train loss:', scores_train[0])
    print('Train accuracy:', scores_train[1])
    
    print("=================valid acc/loss========================")    
    scores_valid = full_model.evaluate(nimg_valid, label_valid_oneHot, verbose=1)
    print('valid loss:', scores_valid[0])
    print('valid accuracy:', scores_valid[1])
    
    #====顯示結果=====     
    from sklearn.metrics import matthews_corrcoef
    
    ResultTxt=os.path.join(DataBase,"ResultTxt")
    make_file(ResultTxt)
    #========== train===============
    print("----------- Train ------------")
    wResult=os.path.join(ResultTxt,"Result_"+modelName+".txt")
    owResult=open(wResult,'w')    
    #顯示每一張training預測的類別標籤
    train_predict_result=np.argmax(full_model.predict(img_train),axis=-1)
    # print("Train預測標籤結果: ",train_predict_result)
    
    # 顯示混淆矩陣    
    train_conm=pd.crosstab(label_train.ravel(),train_predict_result,rownames=['label'],colnames=['predict'])
    print("Train confusion matrix: ")
    print(train_conm)
    owResult.write("Train confusion matrix: \n")
    owResult.write(train_conm.to_string())
    owResult.write("\n")
    
    TrainClfMetrics={}    
    AllTrainClfMetrics=['Train loss: ','Train accuracy: ',"Train Accuracy: ","Train precision: ","Train recall: ","Train F1 Score: ","Train Sensitivity: ","Train Specificity: ","Train MCC: "]
    
   #各項指標值計算
    train_clf_metrics = ClfMetrics(label_train.ravel(), train_predict_result)    
    print("Train 準確度: ",round(train_clf_metrics.accuracy_score(),4))
    print("Train 精確度: ",round(train_clf_metrics.precision_score(),4))
    print("Train 召回率: ",round(train_clf_metrics.recall_score(),4))    
    print("Train F1 Score: ",round(train_clf_metrics.f1_score(),4))
    print("Train Sensitivity:",round(train_clf_metrics.Sensitivity_score(),4))
    print("Train Specificity:",round(train_clf_metrics.Specificity_score(),4))
    print("Train MCC: ",round(matthews_corrcoef(label_train,train_predict_result),4))# matthews_corrcoef(實際類別,預測類別)    
    TrainResultList=[scores_train[0],scores_train[1],round(train_clf_metrics.accuracy_score(),4),round(train_clf_metrics.precision_score(),4),round(train_clf_metrics.recall_score(),4),round(train_clf_metrics.f1_score(),4),round(train_clf_metrics.Sensitivity_score(),4),round(train_clf_metrics.Specificity_score(),4),round(matthews_corrcoef(label_train,train_predict_result),4)]
    
    for i in range(len(AllTrainClfMetrics)): 
        for j in range(len(TrainResultList)):
            if i==j:
                TrainClfMetrics[AllTrainClfMetrics[i]]=TrainResultList[j]            
    for k,v in TrainClfMetrics.items():
        owResult.write(str(k)+str(v)+'\n')         
    
    
    #==========Test===============
    print("----------- Test ------------")
    #顯示每一張testing預測的類別標籤
    test_predict_result=np.argmax(full_model.predict(img_test),axis=-1)
    # print("Tset預測標籤結果: ",test_predict_result)        
    
    # 顯示混淆矩陣    
    test_conm=pd.crosstab(label_test.ravel(),test_predict_result,rownames=['label'],colnames=['predict'])
    print("Test confusion matrix: ")
    print(test_conm)
    owResult.write("\n")
    owResult.write("Test confusion matrix: \n")
    owResult.write(test_conm.to_string())
    owResult.write("\n")
    TestClfMetrics={}
    AllTestClfMetrics=['Test loss: ','Test accuracy: ',"Test Accuracy: ","Test precision: ","Test recall: ","Test F1 Score: ","Test Sensitivity: ","Test Specificity: ","Test MCC: "]
    
    
    #各項指標值計算並取到小數點第2位
    test_clf_metrics = ClfMetrics(label_test.ravel(), test_predict_result)
    print("Test 準確度: ",round(test_clf_metrics.accuracy_score(),4))
    print("Test 精確度: ",round(test_clf_metrics.precision_score(),4))
    print("Test 召回率: ",round(test_clf_metrics.recall_score(),4))
    print("Test F1 Score: ",round(test_clf_metrics.f1_score(),4))  
    print("Test Sensitivity:",round(test_clf_metrics.Sensitivity_score(),4))
    print("Test Specificity:",round(test_clf_metrics.Specificity_score(),4))
    print("Test MCC: ",round(matthews_corrcoef(label_test,test_predict_result),4))
    TestResultList=[scores_test[0],scores_test[1],round(test_clf_metrics.accuracy_score(),4),round(test_clf_metrics.precision_score(),4),round(test_clf_metrics.recall_score(),4),round(test_clf_metrics.f1_score(),4),round(test_clf_metrics.Sensitivity_score(),4),round(test_clf_metrics.Specificity_score(),4),round(matthews_corrcoef(label_test,test_predict_result),4)]
    
    for i in range(len(AllTestClfMetrics)): 
        for j in range(len(TestResultList)):
            if i==j:
                TestClfMetrics[AllTestClfMetrics[i]]=TestResultList[j]            
    for k,v in TestClfMetrics.items():
        owResult.write(str(k)+str(v)+'\n') 
        
    #==========valid=============== 
    print("----------- validation ------------")
    #顯示每一張valid預測的類別標籤
    valid_predict_result=np.argmax(full_model.predict(img_valid),axis=-1)
    # print("valid預測標籤結果: ",valid_predict_result)
    
    # 顯示混淆矩陣    
    V_conm=pd.crosstab(label_valid.ravel(),valid_predict_result,rownames=['label'],colnames=['predict'])
    print("Valid confusion matrix: ")
    print(V_conm)
    owResult.write("\n")
    owResult.write("Valid confusion matrix: \n")
    owResult.write(V_conm.to_string())
    owResult.write("\n")
    
    ValidClfMetrics={}
    AllValidClfMetrics=['valid loss: ',"valid accuracy: ","Valid Accuracy: ","Valid precision: ","Valid recall: ","Valid F1 Score: ","Valid Sensitivity: ","Valid Specificity: ","Valid MCC: "]
    # 
    #各項指標值計算
    valid_clf_metrics = ClfMetrics(label_valid.ravel(), valid_predict_result)
    print("Valid 準確度: ",round(valid_clf_metrics.accuracy_score(),4))
    print("Valid 精確度: ",round(valid_clf_metrics.precision_score(),4))
    print("Valid 召回率: ",round(valid_clf_metrics.recall_score(),4))
    print("Valid F1 Score: ",round(valid_clf_metrics.f1_score(),4)) 
    print("Valid Sensitivity:",round(valid_clf_metrics.Sensitivity_score(),4))
    print("Valid Specificity:",round(valid_clf_metrics.Specificity_score(),4))
    print("Valid MCC: ",round(matthews_corrcoef(label_valid,valid_predict_result),4))
    ValidResultList=[scores_valid[0],scores_valid[1],round(valid_clf_metrics.accuracy_score(),4),round(valid_clf_metrics.precision_score(),4),round(valid_clf_metrics.recall_score(),4),round(valid_clf_metrics.f1_score(),4),round(valid_clf_metrics.Sensitivity_score(),4),round(valid_clf_metrics.Specificity_score(),4),round(matthews_corrcoef(label_valid,valid_predict_result),4)]
    # 
    for i in range(len(AllValidClfMetrics)): 
        for j in range(len(ValidResultList)):
            if i==j:
                ValidClfMetrics[AllValidClfMetrics[i]]=ValidResultList[j]            
    for k,v in ValidClfMetrics.items():
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
    
    