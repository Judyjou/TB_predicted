# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 10:57:18 2022

@author: Judy
"""

import matplotlib.pyplot as plt # plt 用於顯示圖片
import datetime
import os
import time 
start=time.process_time()

def show_train_history_acc(history,train,validation,savebase,modelsName):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title("Training and testing accuracy")
    plt.ylabel(train)
    plt.xlabel("Epoch")
    plt.legend(['Train_accuracy','Test_accuracy'],loc='upper right')
    plt.savefig(os.path.join(savebase,modelsName+"_train_acc_history_"+str(datetime.date.today())+time.strftime("_%H%M",time.localtime())+'.png'))
    plt.show()
    

def show_train_history_loss(history,train,validation,savebase,modelsName):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title("Training and testing losses")
    plt.ylabel(train)
    plt.xlabel("Epoch")
    plt.legend(['Train_loss','Test_loss'],loc='upper right')
    plt.savefig(os.path.join(savebase,modelsName+"_train_loss_history_"+str(datetime.date.today())+time.strftime("_%H%M",time.localtime())+'.png'))
    plt.show()