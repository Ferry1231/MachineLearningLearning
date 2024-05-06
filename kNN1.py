import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import make_circles
from sklearn import svm,datasets
from sklearn.neighbors import KNeighborsClassifier#kNN分类
from sklearn.model_selection import cross_val_score#交叉验证，把数据集分成k块，并进行k次验证
from mpl_toolkits.mplot3d import Axes3D
# an important parameter k
k=10
##
iris=datasets.load_iris()
X=iris.data[:,:2]
y=iris.target
colors=['r','g','b']
knn=KNeighborsClassifier(n_neighbors=20)
knn.fit(X,y)
#get the mesh grid
x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
h=(x_max/x_min)/100
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
Z=knn.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)
#plt.contourf(xx,yy,Z,cmap=plt.cm.Accent,alpha=0.8)
##meshgrid不明白
#交叉验证
cv_scores=[]
ks=[k for k in range(135) if k%3!=0]
for k in ks:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')
    mean=scores.mean()
    cv_scores.append(mean)
    #print(k,mean)
#评分列表，综合评估该模型
##
'''for color,i,target in zip(colors,[0,1,2],iris.target_names):
    plt.scatter(X[y==i,0],X[y==i,1],color=color,label=target)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend(loc='best',shadow=False,scatterpoints=1)
plt.title("Sepal")
plt.show()'''
#predictions=knn.predict(X)
#print(np.unique(predictions,return_counts=True))
MSE=[1-i for i in cv_scores]
plt.plot(ks,MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
