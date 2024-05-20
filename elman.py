#!/usr/bin/env python
# coding: utf-8

# In[1]:


import  numpy as np
import pandas as pd
import os
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (Input, Embedding, SimpleRNN, Dense, Activation,
                          TimeDistributed)

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop


# In[2]:


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


# In[3]:


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


# In[4]:


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


# In[5]:


def elman(X_train,one_hot_train,X_test,one_hot_test):
    #'Adadelta'
    featurs=X_train.shape[1]
    labels=4
    model = keras.Sequential([
        keras.layers.SimpleRNN(labels, return_sequences=False, activation=tf.nn.tanh),
        keras.layers.Dense(labels, activation='softmax')])
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
          
    model.fit(X_train, one_hot_train, epochs=2,verbose=0,batch_size=1)
    eva_train=model.evaluate(X_train,one_hot_train, verbose=0)
    eva_test=model.evaluate(X_test,one_hot_test, verbose=0)
    accuracy_train =eva_train[1]
    accuracy_test =eva_test[1]
    return accuracy_test,accuracy_train


# In[6]:


windows=[11,5,21]
acc_list=[]
for w in windows:
    train,test=make_dataset(w,samples,annotations)
    xtrain=train[ : , :-1]
    ytrain=train[ : ,-1].reshape((-1,1))
    X_train=xtrain.reshape((xtrain.shape[0],xtrain.shape[1],1))
    one_hot_train = to_categorical(ytrain,num_classes=4)
    xtest=test[ : , :-1]
    ytest=test[ : ,-1].reshape((-1,1))
    X_test=xtest.reshape((xtest.shape[0],xtest.shape[1],1))
    one_hot_test = to_categorical(ytest,num_classes=4)
    acc_test,acc_train=elman(X_train,one_hot_train,X_test,one_hot_test)
    acc_list.append((acc_test,acc_train))
    print('Sliding window of size ',w,' : train accuracy = ',acc_train*100,'test accuracy = ',acc_test*100)


# In[ ]:




