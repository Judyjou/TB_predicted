# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 21:22:47 2022

@author: Judy
"""

#check Use GPU
modelName="GoogleE200"
import tensorflow as tf
print(tf.__version__)
# print("Use GPU devices:")
# print(tf.config.list_physical_devices('GPU'))


#count model training needed time
import time 
start=time.process_time()
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import to_categorical
import shutil

import datetime
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
# from tensorflow.keras.applications.densenet import DenseNet121
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg import VGG19
from sklearn.metrics import accuracy_score,recall_score,precision_score,multilabel_confusion_matrix,roc_auc_score
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import cv2
import pandas as pd   
from tensorflow.keras.models import Sequential        

if __name__=="__main__":
    # print("start!")   
    # print(os.getcwd())
    #選擇哪一顆GPU
    # os.environ["CUDA_VISIBLE_DEVICES"]="0" 
    #定義起始檔案路徑位置
    basePath="D:\TB_py"    
    
    def make_file(Folder):
        if not os.path.isdir(Folder):  #如果資料夾不存在就建立
            os.mkdir(Folder)
            
    #=================================== 定義起始圖片路徑=================================== 
    #Make new file for easten mix file
    EastenDataBase=os.path.join(basePath,"EastenDataBase")
    make_file(EastenDataBase)
    
    #define path way
    #basePath="D:\TB_py"
    dataset=os.path.join(basePath,"Alldataset")
     
    szData=os.path.join(dataset,"ChinaSet_AllFiles")
    szData_normal=os.path.join(szData,'Normal')
    szData_TB=os.path.join(szData,'TB')
     
    IndiaData=os.path.join(dataset,"India_DA")
    IndiaData_normal=os.path.join(IndiaData,'Normal')
    IndiaData_TB=os.path.join(IndiaData,'TB')
     
    TBX11KData=os.path.join(dataset,"TBX11K")
    TBX11KData_normal=os.path.join(TBX11KData,'Normal')
    TBX11KData_TB=os.path.join(TBX11KData,'TB')
     
    Tung_Data=os.path.join(dataset,"Tung_XRay")
    Tung_Data_normal=os.path.join(Tung_Data,'Normal')
    Tung_Data_TB=os.path.join(Tung_Data,'TB')
     
    MonData=os.path.join(dataset,"MontgomerySet")
    MonData_normal=os.path.join(MonData,'Normal')
    MonData_TB=os.path.join(MonData,'TB')
     
    BelarusData=os.path.join(dataset,"Belarus")
    BelarusData_TB=os.path.join(BelarusData,'TB')
     
     
    save_trainingProcessImg=os.path.join(EastenDataBase,'save_trainingProcessImg')
    make_file(save_trainingProcessImg)
    print("顯示原始圖片路徑:")
    print(szData,szData_normal,IndiaData,IndiaData_normal,TBX11KData,TBX11KData_normal,Tung_Data,Tung_Data_normal,MonData,MonData_normal,BelarusData_TB) 

    #===================================分割檔案 train/test/valid ===================================
    #分割檔案
    #divide normal shz,tung  data in Train80% Test20%  Normal and TB
    #TBX11K Test20%
    #各資料分train/test Data
    import random
    #---------------trainDataPath------------------------
    trainData=os.path.join(EastenDataBase,"trainData")
    make_file(trainData)
   
    shzTrain=os.path.join(trainData,"shzTrain")  
    indiaTrain=os.path.join(trainData,"indiaTrain")  
    tungTrain=os.path.join(trainData,'tungTrain') 
   
    # trainDataLi=[shzTrain,indiaTrain,tungTrain]
    # for i in trainDataLi:
    #     make_file(i)
   
    #---------------testDataPath------------------------
    testData=os.path.join(EastenDataBase,"testData")
    make_file(testData)   
    
    shzTest=os.path.join(testData,"shzTest")
    indiaTest=os.path.join(testData,"indiaTest")
    TBX11KTest=os.path.join(testData,"TBX11KTest")
    tungTest=os.path.join(testData,'tungTest')
    
    # testDataLi=[shzTest,indiaTest,TBX11KTest,tungTest]
    # for i in testDataLi:
    #     make_file(i)   
   
    #---------------validDataPath------------------------
    validData=os.path.join(EastenDataBase,"validData")
    # make_file(validData)
   
    
   #分割檔案
   #取0.8為train ;剩下的0.2為test
    def divide_data(oriFilePath,trainPath,testPath):
        FileNum=round(len(os.listdir(oriFilePath))*0.8) 
        select_data=random.sample(os.listdir(oriFilePath),k=FileNum)
        for i in select_data:
            shutil.copy(os.path.join(oriFilePath,i),trainPath)
        testLi=[x for x in os.listdir(oriFilePath) if x not in select_data]  
        for j in testLi:
            shutil.copy(os.path.join(oriFilePath,j),testPath)
            
    # #執行分割檔案 
    # #第一次分圖，寫入CSV檔 後面檔案可再覆蓋取代
    # divide_data(szData_normal,shzTrain,shzTest)
    # divide_data(szData_TB,shzTrain,shzTest)
    
    # divide_data(IndiaData_normal,indiaTrain,indiaTest)
    # divide_data(IndiaData_TB,indiaTrain,indiaTest)
    
    # divide_data(Tung_Data_normal,tungTrain, tungTest)
    # divide_data(Tung_Data_TB,tungTrain, tungTest)
    
    # #TBX11K Test
    # FileNum=round(len(os.listdir(TBX11KData_normal))*0.21)
    # select_data_798=random.sample(os.listdir(TBX11KData_normal),k=FileNum)
    # PickNum=round(len(select_data_798)*0.2)
    # select_data_160=random.sample(select_data_798,k=PickNum)
    # for i in select_data_160:
    #     shutil.copy(os.path.join(TBX11KData_normal,i),TBX11KTest)
        
    # #validData 
    # #TBX11KValid normal
    TBX11KValid=os.path.join(validData,"TBX11KValid")
    # make_file(TBX11KValid)
    # unpickLi=[x for x in os.listdir(TBX11KData_normal) if x not in select_data_160]
    # for j in unpickLi:
    #         shutil.copy(os.path.join(TBX11KData_normal,j),TBX11KValid)    
    
    # #TBX11KValid TB
    # FileNum_=round(len(os.listdir(TBX11KData_TB))*0.2) 
    # select_dataTB=random.sample(os.listdir(TBX11KData_TB),k=FileNum_)    
    # for k in select_dataTB:
    #     shutil.copy(os.path.join(TBX11KData_TB,k),TBX11KTest)  
        
    # # validData 
    # unpickLi_TB=[x for x in os.listdir(TBX11KData_TB) if x not in select_dataTB]
    # for p in unpickLi_TB:
    #     shutil.copy(os.path.join(TBX11KData_TB,p),TBX11KValid)  
    #===================================將分好的檔案寫成CSV mataData檔===================================
    #寫CSV mataData檔   一個資料夾一個個新增
    def write_csv(img_filePath,save_csvFileName,storage_path):
        Number=[]
        Label=[]
        Shape=[]
        for img in os.listdir(img_filePath):
            r_img=cv2.imread(os.path.join(img_filePath,img))
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
    
    ## TrainMataData
    # write_csv(shzTrain,"TrainMataData",trainData)
    # TrainData=pd.read_csv(os.path.join(trainData,"TrainMataData.csv"),encoding= 'utf-8')
    # addData(indiaTrain,TrainData,trainData,"TrainMataData")
    # TrainData=pd.read_csv(os.path.join(trainData,"TrainMataData.csv"),encoding= 'utf-8')
    # addData(tungTrain,TrainData,trainData,"TrainMataData")
    TrainData=pd.read_csv(os.path.join(trainData,"TrainMataData.csv"),encoding= 'utf-8')
    print(TrainData)
    
    #TestMataData
    # write_csv(shzTest,"TestMataData",testData)
    # TestData=pd.read_csv(os.path.join(testData,"TestMataData.csv"),encoding= 'utf-8')
    # addData(indiaTest,TestData,testData,"TestMataData")
    # TestData=pd.read_csv(os.path.join(testData,"TestMataData.csv"),encoding= 'utf-8')
    # addData(tungTest,TestData,testData,"TestMataData")
    # TestData=pd.read_csv(os.path.join(testData,"TestMataData.csv"),encoding= 'utf-8')
    # addData(TBX11KTest,TestData,testData, "TestMataData")
    TestData=pd.read_csv(os.path.join(testData,"TestMataData.csv"),encoding= 'utf-8')
    print(TestData)
    
    #ValidMataData
    # write_csv(TBX11KValid, "ValidMataData", validData)
    ValidData=pd.read_csv(os.path.join(validData,"ValidMataData.csv"),encoding= 'utf-8')      
    print(ValidData)   
    #===================================CSV mataData檔(檔名;標籤;原始圖片大小)===================================   
    #===================================開始將圖片前處理 1.先將多餘的邊裁切===================================
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
            
    CL_allTrainImage=os.path.join(trainData,"CL_allTrainImage")
    # make_file(CL_allTrainImage)
    
    CL_allTestImage=os.path.join(testData,"CL_allTestImage")
    # make_file(CL_allTestImage)
    
    CL_allValidImage=os.path.join(validData,"CL_allValidImage")
    # make_file(CL_allValidImage)
    
    TrainOriList=[shzTrain,indiaTrain,tungTrain]
    TestOriList=[shzTest,indiaTest,tungTest,TBX11KTest]
    # 執行裁邊
    # for i in range(len(TrainOriList)):
    #     cut_letterboxing(TrainOriList[i], CL_allTrainImage)       
    
    # for t in range(len(TestOriList)):
    #     cut_letterboxing(TestOriList[t], CL_allTestImage) 
    
    # cut_letterboxing(TBX11KValid,CL_allValidImage)
    #===================================#開始將圖片前處理 2.補邊變正方形===================================        
    # 將每一張圖依照大小，比較後最大的一邊為主去將圖補邊變正方形
    import matplotlib.pyplot as plt # plt 用於顯示圖片
    import matplotlib.image as mpimg # mpimg 用於讀取圖片
    from PIL import Image
     
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
            # image_data=background.resize((224,224))
            background.save(os.path.join(storePath,i))
     
     #執行轉成正方形
    squ_allTrainImage=os.path.join(trainData,"squ_allTrainImage")
    # make_file(squ_allTrainImage)
     
    squ_allTestImage=os.path.join(testData,"squ_allTestImage")
    # make_file(squ_allTestImage)
    
    squ_allValidImage=os.path.join(validData,"squ_allValidImage")
    # make_file(squ_allValidImage)
    
    # squImg(CL_allTrainImage,squ_allTrainImage)
    # squImg(CL_allTestImage, squ_allTestImage)
    # squImg(CL_allValidImage, squ_allValidImage)
    #===================================#開始將圖片前處理 3.轉成相同大小===================================
    height=224
    width=224
    def resizeImg(oriFile,saveFile):
        for ori in os.listdir(oriFile):
            rimg=cv2.imread(os.path.join(oriFile,ori))
            R_img=cv2.resize(rimg,(height,width))
            cv2.imwrite(os.path.join(saveFile,ori),R_img)
            
    res_allTrainImage=os.path.join(trainData,"res_allTrainImage")
    # make_file(res_allTrainImage)
    
    res_allTestImage=os.path.join(testData,"res_allTestImage")
    # make_file(res_allTestImage)
    
    res_allValidImage=os.path.join(validData,"res_allValidImage")
    # make_file(res_allValidImage)
    
    # resizeImg(squ_allTrainImage, res_allTrainImage)
    # resizeImg(squ_allTestImage, res_allTestImage)
    # resizeImg(squ_allValidImage, res_allValidImage)
    #===================================#開始將圖片前處理結束===================================
    #===================================加入圖片矩陣產生新CSV===================================
    
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
    
    CSVAdd_ArrImage(res_allTrainImage,TrainData,trainData,"TrainArr")
    CSVAdd_ArrImage(res_allTestImage,TestData,testData,"TestArr")
    CSVAdd_ArrImage(res_allValidImage,ValidData,validData,"ValidArr")    
    #===================================加入圖片矩陣產生新CSV結束===================================    
    #===================================定義訓練資料===================================
    img_train,img_test,label_train,label_test=TrainData["Image_array"].values,TestData["Image_array"].values,TrainData["Label"].values,TestData["Label"].values
    # img_train,img_test,label_train,label_test=TrainArr["Image_array"].values,TestArr["Image_array"].values,TrainArr["Label"].values,TestArr["Label"].values
    img_train=np.asarray(np.asarray(img_train.tolist()).tolist())
    img_test=np.asarray(np.asarray(img_test.tolist()).tolist())
    print("img_train: ",img_train.shape)
    print("img_test: ",img_test.shape)

    print("label_train: ",label_train.shape)
    print("label_test: ",label_test.shape)

    #256*256轉乘1維數字向量 且儲存為 float /255 做正規劃 直落在0~1之間
    nimg_train=img_train.reshape((1146, 224, 224, 3)).astype('float32')/255
    nimg_test=img_test.reshape((605, 224, 224, 3)).astype('float32') /255
    label_train_oneHot=to_categorical(label_train)
    label_test_oneHot=to_categorical(label_test)

    print("train: ",nimg_train.shape,label_train_oneHot.shape)
    print("test: ",nimg_test.shape,label_test_oneHot.shape)

    #===================================定義驗證資料===================================
    img_valid,label_valid=ValidData["Image_array"].values,ValidData["Label"].values    
    img_valid=np.asarray(np.asarray(img_valid.tolist()).tolist())
    print("img_valid: ",img_valid.shape)
    
    nimg_valid=img_valid.reshape((4280, 224, 224, 3)).astype('float32') /255
    
    label_valid_oneHot=to_categorical(label_valid)    
    print("valid: ",nimg_valid.shape,label_valid_oneHot.shape)
    
    #=================================== Define VGG16 model epoch200=================================== 
    SaveModel=os.path.join(EastenDataBase,"SaveModel")
    make_file(SaveModel)
    # 建立模型
    Img_Height=224
    Img_weight=224
    Img_channel=3
    # VGG16Model=VGG16(weights='imagenet',include_top=False,input_shape=(Img_Height,Img_weight,Img_channel))
    # for layer in VGG16Model.layers:
    #     layer.trainable=False
    # x = Flatten()(VGG16Model.output)
    # x = Dense(512,activation='relu')(x)
    # # x = Dense(512,activation='relu')(x)
    # # x = Dense(512,activation='relu')(x)
    # predictions=Dense(2,activation='softmax')(x)
    # full_model=Model(inputs=VGG16Model.input,outputs=predictions)
    # full_model.summary()
    def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4): 
        # Input: 
        # - f1: number of filters of the 1x1 convolutional layer in the first path
        # - f2_conv1, f2_conv3 are number of filters corresponding to the 1x1 and 3x3 convolutional layers in the second path
        # - f3_conv1, f3_conv5 are the number of filters corresponding to the 1x1 and 5x5  convolutional layer in the third path
        # - f4: number of filters of the 1x1 convolutional layer in the fourth path
          
        # 1st path:
        path1 = Conv2D(filters=f1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
          
        # 2nd path
        path2 = Conv2D(filters = f2_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
        path2 = Conv2D(filters = f2_conv3, kernel_size = (3,3), padding = 'same', activation = 'relu')(path2)
          
        # 3rd path
        path3 = Conv2D(filters = f3_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
        path3 = Conv2D(filters = f3_conv5, kernel_size = (5,5), padding = 'same', activation = 'relu')(path3)
          
        # 4th path
        path4 = MaxPooling2D((3,3), strides= (1,1), padding = 'same')(input_layer)
        path4 = Conv2D(filters = f4, kernel_size = (1,1), padding = 'same', activation = 'relu')(path4)
          
        output_layer = concatenate([path1, path2, path3, path4], axis = -1)
          
        return output_layer
    def GoogLeNet():
        # input layer 
        input_layer = Input(shape = (Img_Height, Img_weight, Img_channel))
      
        # convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
        X = Conv2D(filters = 64, kernel_size = (7,7), strides = 2, padding = 'valid', activation = 'relu')(input_layer)
      
        # max-pooling layer: pool_size = (3,3), strides = 2
        X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)
      
        # convolutional layer: filters = 64, strides = 1
        X = Conv2D(filters = 64, kernel_size = (1,1), strides = 1, padding = 'same', activation = 'relu')(X)
      
        # convolutional layer: filters = 192, kernel_size = (3,3)
        X = Conv2D(filters = 192, kernel_size = (3,3), padding = 'same', activation = 'relu')(X)
      
        # max-pooling layer: pool_size = (3,3), strides = 2
        X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)
      
        # 1st Inception block
        X = Inception_block(X, f1 = 64, f2_conv1 = 96, f2_conv3 = 128, f3_conv1 = 16, f3_conv5 = 32, f4 = 32)
      
        # 2nd Inception block
        X = Inception_block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 192, f3_conv1 = 32, f3_conv5 = 96, f4 = 64)
      
        # max-pooling layer: pool_size = (3,3), strides = 2
        X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)
      
        # 3rd Inception block
        X = Inception_block(X, f1 = 192, f2_conv1 = 96, f2_conv3 = 208, f3_conv1 = 16, f3_conv5 = 48, f4 = 64)
      
        # Extra network 1:
        X1 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
        X1 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X1)
        X1 = Flatten()(X1)
        X1 = Dense(1024, activation = 'relu')(X1)
        X1 = Dropout(0.7)(X1)
        X1 = Dense(2, activation = 'softmax')(X1)
      
        
        # 4th Inception block
        X = Inception_block(X, f1 = 160, f2_conv1 = 112, f2_conv3 = 224, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)
      
        # 5th Inception block
        X = Inception_block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 256, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)
      
        # 6th Inception block
        X = Inception_block(X, f1 = 112, f2_conv1 = 144, f2_conv3 = 288, f3_conv1 = 32, f3_conv5 = 64, f4 = 64)
      
        # Extra network 2:
        X2 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
        X2 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X2)
        X2 = Flatten()(X2)
        X2 = Dense(1024, activation = 'relu')(X2)
        X2 = Dropout(0.7)(X2)
        X2 = Dense(2, activation = 'softmax')(X2)
        
        
        # 7th Inception block
        X = Inception_block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, f3_conv5 = 128, f4 = 128)
      
        # max-pooling layer: pool_size = (3,3), strides = 2
        X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)
      
        # 8th Inception block
        X = Inception_block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, f3_conv5 = 128, f4 = 128)
      
        # 9th Inception block
        X = Inception_block(X, f1 = 384, f2_conv1 = 192, f2_conv3 = 384, f3_conv1 = 48, f3_conv5 = 128, f4 = 128)
      
        # Global Average pooling layer 
        X = GlobalAveragePooling2D(name = 'GAPL')(X)
      
        # Dropoutlayer 
        X = Dropout(0.4)(X)
      
        # output layer 
        X = Dense(2, activation = 'softmax')(X)
        
        # model
        model = Model(input_layer, [X, X1, X2], name = 'GoogLeNet')
      
        return model
    full_model = GoogLeNet()

    full_model.summary()
    epochs=200
    batch_size = 191
    
    # losses.sparse_categorical_crossentropy
    # full_model.compile(loss=[losses.sparse_categorical_crossentropy,losses.sparse_categorical_crossentropy,losses.sparse_categorical_crossentropy]loss_weirht=[1.0.3,0.3],optimizer=keras.optimizers.Adam(learning_rate=0.0001),metrics=['accuracy'])
    full_model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.0001),metrics=['accuracy'])
    history = full_model.fit(nimg_train,label_train_oneHot,validation_data = (nimg_valid,label_valid_oneHot),batch_size=batch_size,epochs=epochs,verbose=2)
    
    try:
        full_model.save_weights(os.path.join(SaveModel,modelName+'.h5'))
        print("Model input Success!!")
    except:
        print("Model input fail, create new one.!!")
        
    full_model.save_weights(os.path.join(SaveModel,modelName+'.h5'))
    print("Save model success.")
        
    #============繪圖顯示訓練過程==============    
    def show_train_history_acc(history,train,validation,modelsName):
        plt.plot(history.history[train])
        plt.plot(history.history[validation])
        plt.title("train accuracy history")
        plt.ylabel(train)
        plt.xlabel("Epoch")
        plt.legend(['train','validation'],loc='upper left')
        plt.savefig(os.path.join(save_trainingProcessImg,modelsName+"_train_acc_history_"+str(datetime.date.today())+time.strftime("_%H%M",time.localtime())+'.png'))
        plt.show()
        
    
    def show_train_history_loss(history,train,validation,modelsName):
        plt.plot(history.history[train])
        plt.plot(history.history[validation])
        plt.title("train loss history")
        plt.ylabel(train)
        plt.xlabel("Epoch")
        plt.legend(['train','validation'],loc='upper left')
        plt.savefig(os.path.join(save_trainingProcessImg,modelsName+"_train_loss_history_"+str(datetime.date.today())+time.strftime("_%H%M",time.localtime())+'.png'))
        plt.show()
        
    show_train_history_acc(history,'dense_4_accuracy','val_dense_4_accuracy',modelName) 
    show_train_history_loss(history,'loss','val_loss',modelName) 
    # Epoch 50/50 6/6 - 103s - loss: 1.3316 - dense_4_loss: 0.4043 - dense_1_loss: 0.5137 - dense_3_loss: 0.4136 - dense_4_accuracy: 0.8220 - dense_1_accuracy: 0.7469 - dense_3_accuracy: 0.8115 - val_loss: 3.1328 - val_dense_4_loss: 1.1070 - val_dense_1_loss: 0.9622 - val_dense_3_loss: 1.0636 - val_dense_4_accuracy: 0.3949 - val_dense_1_accuracy: 0.3664 - val_dense_3_accuracy: 0.4388
    
    #============ 繪圖顯示訓練過程end ==============
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
    
    # #顯示每一張testing預測的類別標籤
    # predict_result=np.argmax(full_model.predict(nimg_test),axis=1)
    # print(predict_result)
    
    # #顯示混淆矩陣
    # print("confusion matrix")
    # conm=pd.crosstab(label_test.ravel(),predict_result,rownames=['label'],colnames=['predict'])
    # print(conm)
    
    #===================================各項指標計算 ===================================
    class ClfMetrics:
        """
        This class calculates some of the metrics of classifier including accuracy, precision, recall, f1 according to confusion matrix.
        Args:
            y_true (ndarray): 1d-array for true target vector.
            y_pred (ndarray): 1d-array for predicted target vector.
        """
        def __init__(self, y_true, y_pred):
            self._y_true = y_true
            self._y_pred = y_pred
        def confusion_matrix(self):
            """
            This function returns the confusion matrix given true/predicted target vectors.
            """
            n_unique = np.unique(self._y_true).size
            cm = np.zeros((n_unique, n_unique), dtype=int)
            for i in range(n_unique):
                for j in range(n_unique):
                    n_obs = np.sum(np.logical_and(self._y_true == i, self._y_pred == j))
                    cm[i, j] = n_obs
            self._tn = cm[0, 0]
            self._tp = cm[1, 1]
            self._fn = cm[1, 0]
            self._fp = cm[0, 1]
            return cm
        def accuracy_score(self):
            """
            This function returns the accuracy score given true/predicted target vectors.
            """
            cm = self.confusion_matrix()
            accuracy = (self._tn + self._tp) / np.sum(cm)
            return accuracy
        def precision_score(self):
            """
            This function returns the precision score given true/predicted target vectors.
            """
            precision = self._tp / (self._tp + self._fp)
            return precision
        def recall_score(self):
            """
            This function returns the recall score given true/predicted target vectors.
            """
            recall = self._tp / (self._tp + self._fn)
            return recall
        def f1_score(self, beta=1):
            """
            This function returns the f1 score given true/predicted target vectors.
            Args:
                beta (int, float): Can be used to generalize from f1 score to f score.
            """
            precision = self.precision_score()
            recall = self.recall_score()
            f1 = (1 + beta**2)*precision*recall / ((beta**2 * precision) + recall)
            return f1  
        def Sensitivity_score(self): # TP/(TP+FN)          
            sn=(self._tp / (self._tp+self._fn))
            return sn
        def Specificity_score(self): # TN/(TN+FP)            
            sp=(self._tn / (self._tn+self._fp))
            return sp
        # def MCC_score(self):            
        #     mcc=(self._tp*self._tn-self._fp*self._fn) /( (self._tp + self._fp)*(self._tp + self._fn)*(self._tn + self._fp)*(self._tn + self._fn))**0.5
        #     # print(mcc)
        #     # MCC = (TP*TN – FP*FN) / √(TP+FP)(TP+FN)(TN+FP)(TN+FN)
        #     return mcc        
        
     #====顯示結果=====     
    from sklearn.metrics import matthews_corrcoef
         
    ResultTxt=os.path.join(EastenDataBase,"ResultTxt")
    make_file(ResultTxt)
    #========== train===============
    print("----------- Train ------------")
    wTrain=os.path.join(ResultTxt,"traintResult_"+modelName+".txt")
    owTrain=open(wTrain,'w')    
    #顯示每一張training預測的類別標籤
    train_predict_result=np.argmax(full_model.predict(img_train)[2],axis=-1)
    # print("Train預測標籤結果: ",train_predict_result)
    
    # 顯示混淆矩陣    
    train_conm=pd.crosstab(label_train.ravel(),train_predict_result,rownames=['label'],colnames=['predict'])
    print("Train confusion matrix: ")
    print(train_conm)
    owTrain.write("Train confusion matrix: \n")
    owTrain.write(train_conm.to_string())
    owTrain.write("\n")
    
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
    TrainResultList=[round(scores_train[0],4),round(scores_train[1],4),round(train_clf_metrics.accuracy_score(),4),round(train_clf_metrics.precision_score(),4),round(train_clf_metrics.recall_score(),4),round(train_clf_metrics.f1_score(),4),round(train_clf_metrics.Sensitivity_score(),4),round(train_clf_metrics.Specificity_score(),4),round(matthews_corrcoef(label_train,train_predict_result),4)]
    
    for i in range(len(AllTrainClfMetrics)): 
        for j in range(len(TrainResultList)):
            if i==j:
                TrainClfMetrics[AllTrainClfMetrics[i]]=TrainResultList[j]            
    for k,v in TrainClfMetrics.items():
        owTrain.write(str(k)+str(v)+'\n')         
    owTrain.close()
    
    
    #==========Test===============
    print("----------- Test ------------")
    wTest=os.path.join(ResultTxt,"testResult_"+modelName+".txt")
    owTest=open(wTest,'w')
    #顯示每一張testing預測的類別標籤
    test_predict_result=np.argmax(full_model.predict(img_test)[2],axis=-1)
    # print("Tset預測標籤結果: ",test_predict_result)        
    
    # 顯示混淆矩陣    
    test_conm=pd.crosstab(label_test.ravel(),test_predict_result,rownames=['label'],colnames=['predict'])
    print("Test confusion matrix: ")
    print(test_conm)
    owTest.write("Test confusion matrix: \n")
    owTest.write(test_conm.to_string())
    owTest.write("\n")
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
    TestResultList=[round(scores_test[0],4),round(scores_test[1],4),round(test_clf_metrics.accuracy_score(),4),round(test_clf_metrics.precision_score(),4),round(test_clf_metrics.recall_score(),4),round(test_clf_metrics.f1_score(),4),round(test_clf_metrics.Sensitivity_score(),4),round(test_clf_metrics.Specificity_score(),4),round(matthews_corrcoef(label_test,test_predict_result),4)]
    
    for i in range(len(AllTestClfMetrics)): 
        for j in range(len(TestResultList)):
            if i==j:
                TestClfMetrics[AllTestClfMetrics[i]]=TestResultList[j]            
    for k,v in TestClfMetrics.items():
        owTest.write(str(k)+str(v)+'\n') 
    owTest.close()
    #==========valid===============     
    print("----------- validation ------------")
    wValid=os.path.join(ResultTxt,"validResult_"+modelName+".txt")    
    owValid=open(wValid,'w')
    #顯示每一張valid預測的類別標籤
    valid_predict_result=np.argmax(full_model.predict(img_valid)[2],axis=-1)
    # print("valid預測標籤結果: ",valid_predict_result)
    
    # 顯示混淆矩陣    
    V_conm=pd.crosstab(label_valid.ravel(),valid_predict_result,rownames=['label'],colnames=['predict'])
    print("Valid confusion matrix: ")
    print(V_conm)
    owValid.write("Valid confusion matrix: \n")
    owValid.write(V_conm.to_string())
    owValid.write("\n")
    
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
    ValidResultList=[round(scores_valid[0],4),round(scores_valid[1],4),round(valid_clf_metrics.accuracy_score(),4),round(valid_clf_metrics.precision_score(),4),round(valid_clf_metrics.recall_score(),4),round(valid_clf_metrics.f1_score(),4),round(valid_clf_metrics.Sensitivity_score(),4),round(valid_clf_metrics.Specificity_score(),4),round(matthews_corrcoef(label_valid,valid_predict_result),4)]
    # 
    for i in range(len(AllValidClfMetrics)): 
        for j in range(len(ValidResultList)):
            if i==j:
                ValidClfMetrics[AllValidClfMetrics[i]]=ValidResultList[j]            
    for k,v in ValidClfMetrics.items():
        owValid.write(str(k)+str(v)+'\n')  
    
    owValid.close()
    

    # # #close GPU
    # from numba import cuda
    # cuda.select_device(0)
    # cuda.close()
    
    # #釋放記憶體
    # import gc
    # del full_model
    # gc.collect() #14982  10913
    
    # del history  
    # gc.collect()  #10912

     