# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:38:36 2021

@author: Halil İbrahim BAYAT

    Classification Task

"""
import pandas as pd; import numpy as np;import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,accuracy_score,confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


###############################################################################

data=pd.read_csv('weatherAUS.csv')

'''data_info=data.info()
data_null=data.isnull().any()
data_head=data.head();data_corr=data.corr()
data_des=data.describe()
'''
# Data Preprocessing

imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputer2=SimpleImputer(missing_values=np.nan,strategy='mean')

data['RainTomorrow']=imputer.fit_transform(data['RainTomorrow'].values.reshape(-1,1))
data['RainToday']=imputer.fit_transform(data['RainToday'].values.reshape(-1,1))
data['Evaporation']=imputer2.fit_transform(data['Evaporation'].values.reshape(-1,1))
data['Sunshine']=imputer2.fit_transform(data['Sunshine'].values.reshape(-1,1))
data['MinTemp']=imputer2.fit_transform(data['MinTemp'].values.reshape(-1,1))
data['Rainfall']=imputer2.fit_transform(data['Rainfall'].values.reshape(-1,1))
data['WindGustSpeed']=imputer2.fit_transform(data['WindGustSpeed'].values.reshape(-1,1))
data['WindSpeed9am']=imputer2.fit_transform(data['WindSpeed9am'].values.reshape(-1,1))

data['Humidity3pm']=imputer2.fit_transform(data['Humidity3pm'].values.reshape(-1,1))
data['Humidity9am']=imputer2.fit_transform(data['Humidity9am'].values.reshape(-1,1))
data['Pressure9am']=imputer2.fit_transform(data['Pressure9am'].values.reshape(-1,1))
data['Pressure3pm']=imputer2.fit_transform(data['Pressure3pm'].values.reshape(-1,1))
data['Cloud9am']=imputer2.fit_transform(data['Cloud9am'].values.reshape(-1,1))
data['Cloud3pm']=imputer2.fit_transform(data['Cloud3pm'].values.reshape(-1,1))
data['Temp9am']=imputer2.fit_transform(data['Temp9am'].values.reshape(-1,1))
data['Temp3pm']=imputer2.fit_transform(data['Temp3pm'].values.reshape(-1,1))
data['MaxTemp']=imputer2.fit_transform(data['MaxTemp'].values.reshape(-1,1))
data['WindSpeed3pm']=imputer2.fit_transform(data['WindSpeed3pm'].values.reshape(-1,1))

le=LabelEncoder()

data['RainTomorrow']=le.fit_transform(data['RainTomorrow'].values.reshape(-1,1))
data['RainToday']=le.fit_transform(data['RainToday'].values.reshape(-1,1))

#ohe=OneHotEncoder()
#arr=data['Location'].values
#arr=ohe.fit_transform(arr.reshape(-1,1)).toarray()
# To much data my notebook can not  hande it

data=data.drop(['Date','Location','WindDir9am','WindGustDir','WindDir3pm'],axis=1)

''' 
    list of the columns that contain nan value:
    
    'vaporatio' : 'Sunshine' : 'MinTemp' : 'MaxTemp' : 'Rainfall'
    'WindGustSpeed' : 'WindSpeed9am' : 'WindSpeed3am' : 'Humidity9am'
    'Humidity3pm' : 'Pressure9am' : 'Pressure3pm' : 'Cloud9am' : 'Cloud3pm'
    'Temp9am' : 'Temp3pm' : 'RainToday' : 'RainTomorrow'    
'''

'''
     List of Columns that consist of Categoric Data:
     'RainToday' : 'RainTomorrow' : 'Location' : 'Data'**
     
     **Is values of (format=xxxx-xx-xx) 'date' necessary and possitive
     effect for classification ?!  
   
'''

# Specify inputs and targets

x=data.drop(['RainTomorrow'],axis=1).values
y=data['RainTomorrow'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
sc=MinMaxScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# Creating Models

# ANN
'''
model=Sequential()

model.add(Dense(36,activation='relu'))
model.add(Dense(72,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=128,epochs=100)

y_pred=model.predict(x_test)
y_pred=(y_pred>0.5)

cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
acc_sc=accuracy_score(y_test,y_pred)

print(cm)
print(cr)

'''

# SVM

from sklearn.svm import SVC
from sklearn.metrics import roc_curve,auc

svm=SVC(kernel='rbf',random_state=0)
svm.fit(x_train,y_train.ravel())

y_pred=svm.predict(x_test)

cm=confusion_matrix(y_test, y_pred)
cr=classification_report(y_test,y_pred)

FPR,TPR,tresh_hold=roc_curve(y_test,y_pred)
auc_val=auc(FPR,TPR)

# Drawing ROC curve 

plt.figure()
plt.plot(FPR,TPR,label='AUC %0.2f % auc_val')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.ylabel('True Possitive Rate')
plt.xlabel('Flse Possitive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')

from sklearn.model_selection import GridSearchCV

p=[{'kernel':['Linear']},
   {'kernel':['rbf']},
   {'kernel':['sigmoid']},
   {'kernel':['poly']}]

gs=GridSearchCV(estimator=svm,param_grid=p,scoring='accuracy',cv=10,n_job=-1)

grid_s=gs.fit(x_train,y_train)
bs=grid_s.best_score_
bp=grid_s.best_params_
results=grid_s.cv_results_













