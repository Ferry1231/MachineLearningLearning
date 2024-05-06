import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles
from mpl_toolkits.mplot3d import Axes3D
#X:features;c:class labels
X,c=make_circles(n_samples=500,noise=0.09)
rgb=np.array(['r','g'])
'''
#2d graph plotting
plt.scatter(X[:,0],X[:,1],color=rgb[c])
plt.show()
'''
fig=plt.figure(figsize=(18,15))
ax=fig.add_subplot(111,projection='3d')
z=X[:,0]**2+X[:,1]**2
ax.scatter(X[:,0],X[:,1],z,color=rgb[c])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
#plt.show()

features=np.concatenate((X,z.reshape(-1,1)),axis=1)
clf=svm.SVC(kernel='linear')
clf.fit(features,c)
Y=lambda x,y:(-clf.intercept_[0]-clf.coef_[0][0]*x-clf.coef_[0][1]*y)/clf.coef_[0][2]
tmp=np.linspace(-1.5,1.5,100)
x,y=np.meshgrid(tmp,tmp)
ax.plot_surface(x,y,Y(x,y))
plt.show()