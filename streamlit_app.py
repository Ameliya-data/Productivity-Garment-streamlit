import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

df=pd.read_csv('garments_worker_productivity_10000_rows.csv')

df

df.head()

df.tail()

df.shape

df.columns

df.dtypes

df1=df['date'].value_counts()

df2=df['quarter'].value_counts()

df2

plt.bar(df2.index,df2.values,color='m')

df3=df['day'].value_counts()

plt.plot(df3.index,df3.values,marker='o')

df4=df['department'].value_counts()

df4

plt.pie(df4,labels=df4.index,autopct='%.1f%%')

df.isna().sum()

df['wip'].unique()

df['wip']=df['wip'].fillna(df['wip'].mean())

df['wip']

df.isna().sum()

sns.heatmap(df.corr(numeric_only=True))

df.corr(numeric_only=True)

df.drop(['date'],axis=1,inplace=True)

df

dfe=pd.get_dummies(df[['quarter','department','day']],drop_first=True,dtype=int)

dfe

df=pd.concat([df,dfe],axis=1)

df

df.drop(['quarter','department','day'],axis=1,inplace=True)

x=df.iloc[:,:-1]

x

y=df.iloc[:,-1]

y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

x_train

x_test

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(x_train)

x_train=scaler.transform(x_train)

x_test=scaler.transform(x_test)

x_train

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)

y_pred

y_test

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

cm

from sklearn.metrics import accuracy_score

score=accuracy_score(y_test,y_pred)

score

