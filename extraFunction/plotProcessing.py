# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 18:13:56 2022

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
    plt.title("train accuracy history")
    plt.ylabel(train)
    plt.xlabel("Epoch")
    plt.legend(['train','validation'],loc='upper left')
    plt.savefig(os.path.join(savebase,modelsName+"_train_acc_history_"+str(datetime.date.today())+time.strftime("_%H%M",time.localtime())+'.png'))
    plt.show()
    

def show_train_history_loss(history,train,validation,savebase,modelsName):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title("train loss history")
    plt.ylabel(train)
    plt.xlabel("Epoch")
    plt.legend(['train','validation'],loc='upper left')
    plt.savefig(os.path.join(savebase,modelsName+"_train_loss_history_"+str(datetime.date.today())+time.strftime("_%H%M",time.localtime())+'.png'))
    plt.show()