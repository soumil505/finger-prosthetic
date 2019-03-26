# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:08:23 2019

@author: soumil
"""

import pandas as pd
import numpy as np
import os
def read(fname):
    df=pd.read_csv(fname)
    X=df.as_matrix()
    return X

def smoothen(l):
    for i in range(5,len(l)):
        l[i]=np.mean(l[i-5:i+1])
    return l

def open_all():
    files=os.listdir()
    files=[file for file in files if ".csv" in file]
    X=[]
    for file in files:
        if np.shape(X)==(0,):
            X=read(file)
        else:
            X=np.concatenate((X,read(file)),axis=0)
    return X

def grad(l,window_size=5):
    l2=[0]+[l[i]-l[i-1] for i in range(1,len(l))]
    for i in range(window_size,len(l2)):
        l2[i]=np.mean(l2[i-window_size:i+1])
    return l2

def ms(l,window_size=5):
    l2=np.asarray(l)**2
    for i in range(window_size,len(l2)):
        l2[i]=np.mean(l2[i-window_size:i+1])
    return l2

def feature_extract(X):
    for i in range(X.shape[1]-1):
        X[:,i]=np.reshape(smoothen(X[:,i]),X[:,i].shape)
        
    Y=np.zeros((X.shape[0],X.shape[1]*2-1))
    
    Y[:,1]=np.reshape(grad(X[:,0]),Y[:,1].shape)
    Y[:,3]=np.reshape(grad(X[:,1]),Y[:,3].shape)
    Y[:,5]=np.reshape(grad(X[:,2]),Y[:,5].shape)
    
    Y[:,0]=np.reshape(ms(X[:,0]),Y[:,0].shape)
    Y[:,2]=np.reshape(ms(X[:,1]),Y[:,2].shape)
    Y[:,4]=np.reshape(ms(X[:,2]),Y[:,4].shape)
    Y[:,6]=X[:,-1]
    
    return Y[50:,:]