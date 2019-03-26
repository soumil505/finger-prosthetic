# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 20:10:54 2019

@author: soumil
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
from misc import *

clf=MLPClassifier((20,20,20))
scaler=StandardScaler()
X=open_all()
Y=feature_extract(X)


x=Y[:,:-1]
y=Y[:,-1:]
x=scaler.fit_transform(x)

y=enc.fit_transform(y).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))

