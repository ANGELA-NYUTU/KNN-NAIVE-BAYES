#Reading libraries
import pandas as pd
import numpy as np
#Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
#Modeling libraries
from sklearn.neighbors import KNeighborsClassifier
#Splitting libraries
from sklearn.model_selection import train_test_split
#Optimization libraries
from sklearn.model_selection import GridSearchCV
#Metrics libraries
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,normalize
#preview data
dftrain=pd.read_csv('dftrainclean')
dftrain.head()
dftrain.shape
dftrain.dtypes
col=['Sex', 'Embarked']
def codes(df):
    for i in df[col]:
        df[i]=df[i].astype('category')
        df[i]=df[i].cat.codes
        return df
codes(dftrain)
#Defining variables
x1=dftrain.iloc[:,1:]
x=dftrain.iloc[:,1:].values
y=dftrain.iloc[:,:1].values
#Spliting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y, test_size=0.3, random_state=42)
x2_train, x2_test, y2_train, y2_test = train_test_split(x, y, test_size=0.4, random_state=42)

#pre processing
#Standardizing
def scale(var1,var2):
  sc = StandardScaler()
  var1 = sc.fit_transform(var1)
  var2 = sc.transform(var2)
  return var1,var2
scale(x_train,x_test)
scale(x1_train,x1_test)
scale(x2_train,x2_test)

#Baseline modeling/ 80-20 split
knn=KNeighborsClassifier(n_neighbors=5)
model=knn.fit(x_train,y_train)
ypred=model.predict(x_test)
print('Accuracy:',round(metrics.accuracy_score(y_test,ypred)*100),'%')
print('matrix:',metrics.confusion_matrix(y_test,ypred))
print(metrics.classification_report(y_test, ypred))

#Baseline modeling/ 70-30 split
knn=KNeighborsClassifier(n_neighbors=5)
model1=knn.fit(x1_train,y1_train)
ypred1=model1.predict(x1_test)
print('Accuracy:',round(metrics.accuracy_score(y1_test,ypred1)*100),'%')
print('matrix:',metrics.confusion_matrix(y1_test,ypred1))
print(metrics.classification_report(y1_test, ypred1))

#Baseline modeling/ 60-40 split
knn=KNeighborsClassifier(n_neighbors=5)
model2=knn.fit(x2_train,y2_train)
ypred2=model2.predict(x2_test)
print('Accuracy:',round(metrics.accuracy_score(y2_test,ypred2)*100),'%')
print('matrix:',metrics.confusion_matrix(y2_test,ypred2))
print(metrics.classification_report(y2_test, ypred2))

#optimized model
param = { 'n_neighbors' : np.arange(1,30),
          'metric' : ['euclidean', 'manhattan', 'minkowski'],
          'weights': ['uniform', 'distance']}
gridknn=GridSearchCV(knn,param,cv=10,scoring='accuracy',verbose=1)
gridknn.fit(x_train,y_train)
gridknn.best_params_

knn1=KNeighborsClassifier(n_neighbors=10,weights='distance', metric='manhattan')
knn1.fit(x_train,y_train)
ypredk1=knn1.predict(x_test)
print(metrics.accuracy_score(y_test,ypredk1))