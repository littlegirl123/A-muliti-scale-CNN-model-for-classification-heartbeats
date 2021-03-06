#coding:utf-8
from __future__ import print_function 
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py 
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from resnet_model3_2 import resnet_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


n_input =300
n_classes =4

#导入训练集和测试集
traindata = sio.loadmat('train_avg_NSVF_RR_wavelet.mat')
testdata =sio.loadmat('test_NSVF_RR_wavelet.mat')
traindata=traindata['data_wavelet']
testdata=testdata['data_wavelet']


#训练集
train_x=np.reshape(traindata[:,n_classes:604],[traindata.shape[0],n_input,2])
train_x_RR=np.reshape(traindata[:,604:607],[traindata.shape[0],3,1])
train_y=np.reshape(traindata[:,0:n_classes],[traindata.shape[0],n_classes])

#测试集
test_x=np.reshape(testdata[:,n_classes:604],[testdata.shape[0],n_input,2])
test_x_RR=np.reshape(testdata[:,604:607],[testdata.shape[0],3,1])
test_y=np.reshape(testdata[:,0:n_classes],[testdata.shape[0],n_classes])

#从测试集中抽1000个样本作为验证集
validation_x=np.reshape(testdata[:1000,n_classes:604],[1000,n_input,2])
validation_x_RR=np.reshape(testdata[:1000,604:607],[1000,3,1])
validation_y=np.reshape(testdata[:1000,0:n_classes],[1000,n_classes])

model_ng=resnet_model(input_shape=(300,2))
init=tf.initialize_all_variables()

with tf.Session() as sess: 
	 
	 sess.run(init)
	 filepath="resnet_model5.h5"
	 checkpoint=ModelCheckpoint(filepath,monitor='val_acc',save_best_only=True,save_weights_only=False,period=1,mode='max')
	 callbacks_list = [checkpoint]
	 model_ng.fit([train_x,train_x_RR],train_y,batch_size=32,epochs=50,validation_data=([validation_x,validation_x_RR],validation_y),callbacks=callbacks_list)
	 
	 resnet_mod=load_model('resnet_model5.h5')
	 print(resnet_mod.evaluate([test_x,test_x_RR],test_y)[1])
	 y_pred=resnet_mod.predict([test_x,test_x_RR])
	 C=confusion_matrix(test_y.argmax(axis=1),y_pred.argmax(axis=1))
	 print(C)
	 


     # early_stopping=EarlyStopping(monitor='acc',patience=3,mode='max')	 
     #model_ng.fit([train_x,train_x_RR],train_y,batch_size=32,epochs=30,validation_data=([validation_x,validation_x_RR],validation_y))
	 # ,callbacks=[early_stopping]	    	
	
	 

     











   
            
