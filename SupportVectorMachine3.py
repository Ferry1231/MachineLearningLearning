#other kernel for SVM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
iris=datasets.load_iris()
'''
print(iris.data[0:5])
print(iris.feature_names)
print(iris.target[0:5])
print(iris.target_names)
'''
X=iris.data[:,:2]
y=iris.target
colors=['r','g','b']
C=1#C参数越小，边距，即支持向量到超平面的距离越大，但这样会降低精度
clf=svm.SVC(kernel='poly',C=C,gamma='auto',degree=4).fit(X,y)#gamma值越高，拟合精度越高，也容易导致过度拟合#多项式内核(poly)，
#clf=svm.SVC(kernel='rbf',C=C,gamma='auto').fit(X,y)#rbf内核(rbf)是根据每个点到原点的距离赋值分类
title='SVC with linear kernel'
x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
h=(x_max/x_min)/100
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)
plt.contourf(xx,yy,Z,cmap=plt.cm.Accent,alpha=0.8)
for color,i,target in zip(colors,[0,1,2],iris.target_names):
    plt.scatter(X[y==i,0],X[y==i,1],color=color,label=target)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend(loc='best',shadow=False,scatterpoints=1)
plt.title(title)
plt.show()
predictions=clf.predict(X)
print(np.unique(predictions,return_counts=True))