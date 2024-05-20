#!/usr/bin/env python
# coding: utf-8

# In[1]:


import  numpy as np
import pandas as pd
import os
import tensorflow as tf
#from keras.models import Sequential
from keras.layers import (Input, Embedding, SimpleRNN, Dense, Activation,
                         TimeDistributed)

from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from keras.layers import Dense
from pyneurgen.recurrent import NARXRecurrent
from pyneurgen.neuralnet import NeuralNet
from tensorflow.keras.optimizers import RMSprop


# In[2]:


#pip uninstall tf-agents
#pip install tf-agents-nightly


# In[3]:


#Instances_lowcost_H3
#test_instance
path="C:/Users/razeeee/dataset"
path1="C:/Users/razeeee/dataset1"
sample=[]
annotation=[]
for file in os.listdir(path1):
    s=pd.read_csv(os.path.join(path1,file))
    sample.append(np.array(s))
for file in os.listdir(path):   
    anno=pd.read_csv(os.path.join(path,file))
    annotation.append(np.array(anno))
samples=np.array(sample)
annotations = np.array(annotation) 
for i in range(len(samples)):
    annotations[i] = np.array(annotations[i],dtype='int')
    samples[i] = np.array(samples[i])
    samples[i]=samples[i][1: ,1].astype(float)


# In[4]:


def make_pattern(w,sample,annotation):
    pattern=np.zeros((len(sample)-(w-1),2*w))
    for k in range(pattern.shape[0]):
        #print('k',k)
        sa=sample[k:k+w]
        #print('sa',sa)
        a=annotation[k:k+w]
        patt=[]
        for i in range(len(sa)):
            patt.append(sa[i])
            patt.append(int(a[i]))
        #print('p',len(patt))
        pattern[k]=patt
    return pattern 


# In[5]:


def make_dataset(w,samples,annotations):
    pattern0=make_pattern(w,samples[0],annotations[0])
    t0=int(np.round(0.8*pattern0.shape[0]))
    train0=pattern0[ :t0, : ]
    test0=pattern0[t0: , : ]
    pattern1=make_pattern(w,samples[1],annotations[1])
    t1=int(np.round(0.8*pattern1.shape[0]))
    train1=pattern1[ :t1, : ]
    test1=pattern1[t1: , : ]
    Train=np.concatenate((train0,train1), axis=0)
    Test=np.concatenate((test0,test1), axis=0)
    for i in range(2,len(samples)):
        pattern=make_pattern(w,samples[i],annotations[i])
        t=int(np.round(0.8*pattern.shape[0]))
        train=pattern[ :t, : ]
        test=pattern[t: , : ]
        Train=np.concatenate((Train,train), axis=0)
        Test=np.concatenate((Test,test), axis=0)
    return Train,Test    


# In[6]:


class Narx(keras.Model):

    def __init__(self):
        super(Narx, self).__init__(name='narx')
        self.hiddenlayer = keras.layers.Dense(4, activation='tanh')
        self.outputLayer = keras.layers.Dense(4, activation='softmax')

    def call(self, inputs, flag = False):
        if (flag):
            hidden_nodes = self.hiddenlayer(inputs)
            return self.outputLayer(hidden_nodes)
        else: 
            hidden_nodes = self.hiddenlayer(inputs)
            return self.outputLayer(hidden_nodes)
        
windows=[11,5,21]
acc_train_list=[]
acc_test_list=[]
for w in windows:
    train,test=make_dataset(w,samples,annotations)
    #print('train.shape=',train.shape)
    X_train=train[ : , :-1]
    ytrain=train[ : ,-1].reshape((-1,1))
    one_hot_train = to_categorical(ytrain,num_classes=4)
    X_test=test[ : , :-1]
    ytest=test[ : ,-1].reshape((-1,1))
    one_hot_test = to_categorical(ytest,num_classes=4)
    model = Narx()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss='mse',
              metrics=['accuracy'])
    model.fit(X_train, one_hot_train, epochs=3,verbose=0,batch_size=1)
    eva_train=model.evaluate(X_train,one_hot_train, verbose=0)
    eva_test=model.evaluate(X_test,one_hot_test, verbose=0)
    acc_train_list.append(eva_train[1])
    acc_test_list.append(eva_test[1])
    print('Sliding window of size ',w,' : train accuracy = ',eva_train[1]*100,'test accuracy = ',eva_test[1]*100)


# In[ ]:




