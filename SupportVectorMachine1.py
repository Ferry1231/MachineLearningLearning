import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
#
data=pd.read_csv('svm.csv')
x=data.x1
y=data.x2
points=data[['x1','x2']].values
result=data.r
#get the SVM model
clf=svm.SVC(kernel='linear')
clf.fit(points,result)

print('Vector of weights(w)=',clf.coef_[0])
print('b=',clf.intercept_[0])
print('Indices of support vectors=',clf.support_)#支持向量的下标
print('Support vectors=',clf.support_vectors_)#Support vectors
print('Number of support vectors of each class=',clf.n_support_)
print('Coeffiencies of the support vector in decision function=',np.abs(clf.dual_coef_))
#all coefficients
w=clf.coef_[0]
slope=-w[0]/w[1]
b=clf.intercept_[0]
#get the hyperplane
xx=np.linspace(0,4)
yy=slope*xx-(b/x[1])
#every curve of supports vectors
s=clf.support_vectors_[0]
yy_down=slope*xx+(s[1]-slope*s[0])
s=clf.support_vectors_[-1]
yy_up=slope*xx+(s[1]-slope*s[0])
#plot
plt.scatter(x,y,color='r',alpha=0.7,s=10)
plt.xlabel('X1')
plt.ylabel('X2')
#plt.scatter(xx,yy,s=15,alpha=0.7)
plt.plot(xx,yy,linewidth=2,color='g')
plt.plot(xx,yy_down,'k--')
plt.plot(xx,yy_up,'k--')
plt.show()
#test for prediction
print(clf.predict([[3,3]])[0])
'''
#plot
plt.scatter(x,y,color='r',alpha=0.7,s=10)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
'''