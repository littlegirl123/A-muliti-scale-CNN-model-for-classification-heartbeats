#coding:utf-8
import numpy as np
import keras
import keras.layers as layers
from keras.models import Model
from keras.layers import (Input, 
                          Dense,                          
                          Flatten,                        
                          Dropout, 
                          BatchNormalization,
                          Conv1D, 
                          MaxPooling1D, 
                          Activation)

from keras.optimizers import adam
from keras.layers.merge import Concatenate
from keras import regularizers

import random
import os
from os import listdir
from os.path import isfile, join
import gc


ksize1=3
ksize2=5
ksize3=7
#��������ʽ
def layoutConv1D(strides = 1, nFilters=None,  kSize = 5):
    
    return Conv1D(
        filters = nFilters, #����˸���
        kernel_size = kSize,  #����˵Ŀ����ʱ�򴰳��ȡ��������˵ĳ��Ϳ���ȣ���Ҫ��kernel_h��kernel_w�ֱ��趨
        strides = strides,   #����               
        padding = 'same',  #�������ͼ��Ĵ�Сһ��  
        activation='relu',  #�����
        use_bias=True,    #���ƫ��
        #kernel_regularizer=regularizers.l2(0.01),
        kernel_initializer = 'random_normal')  #����˳�ʼ����ʽ


#ʵ�ֿ�ͨ���Ľ�������Ϣ���ϣ�ʹ�òв���ǰһ��������С��ͬ��������������ά
def layoutIdentity1D(strides = 1, nFilters=None):
    
    return Conv1D(
                filters = nFilters, 
                kernel_size = 1,   #����˴�С1*1
                strides = 1, 
                padding = 'same',       
                activation='linear',   #��g(z)=z
                kernel_initializer = 'one', 
                use_bias = False,
                trainable=False)

#����в�飬����Ϊ�в���������ξ���õ���feature-map�Ͳв������
def layoutResidualBlock(cnt,flow,shortcut,ksize):
    
        DropoutRate = 0.5
        strides = 2 if cnt%2==0 else 1       
        flow = BatchNormalization()(flow)
        flow = Activation(activation='relu')(flow)
        flow = Dropout(DropoutRate)(flow)
        #��һ������C1�����������Сһ�� 
        flow = layoutConv1D(nFilters = 8, strides = 1,kSize=ksize)(flow)
        flow = BatchNormalization()(flow)
        flow = Activation(activation='relu')(flow)
        flow = Dropout(DropoutRate)(flow)

        #�ڶ�������C2������Ĵ�СΪ
        flow = layoutConv1D(nFilters = 16,  strides = strides,kSize=ksize)(flow)              
        
        nFiltersInput, nFiltersOutput = int(shortcut.get_shape()[-1]), int(flow.get_shape()[-1])
        if  nFiltersInput != nFiltersOutput:                    
            #Identity convlution             
            shortcut = layoutIdentity1D(nFilters = nFiltersOutput)(shortcut)                   
            shortcut = MaxPooling1D(pool_size=strides, padding = 'same')(shortcut)
            pass
        else:
            shortcut = MaxPooling1D(pool_size=strides, padding = 'same')(shortcut)
                    
        flow=keras.layers.concatenate([flow,shortcut])	      
        # print(flow.get_shape())
        return flow,shortcut
   

#input_shape=(300,2),
def resnet_model(input_shape):
    
    input1 = Input(shape=(300,2))
    input2 =Input(shape=(3,1))
    DropoutRate = 0.3
    
###ksize=3
    #C1_1
    flow1 = layoutConv1D(strides = 1, kSize=ksize1, nFilters=8)(input1)               #None*300*8
    flow1 = BatchNormalization()(flow1)   #�淶��������������������ֹ�����
    flow1 = Activation(activation='relu')(flow1)  #�����	

###ksize=5
    #C1_2
    flow2 = layoutConv1D(strides = 1, kSize=ksize2, nFilters=8)(input1)               #None*300*8
    flow2 = BatchNormalization()(flow2)   #�淶��������������������ֹ�����
    flow2 = Activation(activation='relu')(flow2)  #�����


###ksize=7
    #C1_3
    flow3 = layoutConv1D(strides = 1, kSize=ksize3, nFilters=8)(input1)  			 #None*300*8             
    flow3 = BatchNormalization()(flow3)   #�淶��������������������ֹ�����
    flow3 = Activation(activation='relu')(flow3)  #�����
	
    flow=keras.layers.concatenate([flow1,flow2])
    flow=keras.layers.concatenate([flow,flow3])								#None*300*24
   
    shortcut = flow

	
    #C2
    flow = layoutConv1D(strides = 1, nFilters=8,kSize=ksize1)(flow)             #None*300*8
    flow = BatchNormalization()(flow)   
    flow = Activation(activation='relu')(flow)
    flow = Dropout(DropoutRate)(flow)	
	
    #C3
    flow = layoutConv1D(strides = 2, nFilters=16, kSize=ksize1)(flow)       	#None*150*16
    
    shortcut = layoutIdentity1D(nFilters = int(flow.get_shape()[-1]))(shortcut)   #None*300*16      
    #�²�����S1��pool_size�²������ӣ����˲�����С��strides���������ΪNone,��Ĭ�ϴ�СΪpool_size,paddingΪ���ʽ
    shortcut = MaxPooling1D(pool_size=2, padding='same')(shortcut)      		#None*150*16
    
    flow=keras.layers.concatenate([flow,shortcut])  

    for i in range(2,7):
		
        flow,shortcut = layoutResidualBlock(i,flow,shortcut,ksize1)
        pass

    flow = BatchNormalization()(flow)
    flow = Activation(activation='relu')(flow)
    flow = Dropout(DropoutRate)(flow)
    flow = Flatten()(flow)
	
    flow = Dense(128, activation='relu')(flow)
    flow = Dropout(DropoutRate)(flow)

	#�������߶��µõ��������Լ�RR��������

    flow=keras.layers.concatenate([flow, Flatten()(input2)])
    flow=BatchNormalization()(flow)
	
	
    predictions = Dense(4, activation='softmax')(flow)
   
    adam_lr=0.005
    model = Model(inputs=[input1,input2], outputs=predictions)
    model.compile(optimizer = adam(lr=adam_lr) , loss='categorical_crossentropy', metrics=['accuracy'])
    return model

