import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.datasets import make_regression,make_blobs,make_circles#some magical datasets generators
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import pickle
from mpl_toolkits.mplot3d import Axes3D
bc=datasets.load_breast_cancer()
df=pd.DataFrame(bc.data)
#print(bc.feature_names)
#print(bc.DESCR)
#print(bc.target)
#print(df.info())
corr=df.corr()#快速给出特性两两之间相关性,返回DataFrame类
df.to_csv('breast_cancer.csv')
#print(corr)
corr.to_csv('corr1.csv')
corr1=pd.read_csv('corr1.csv')
top_six=corr1['10'].nlargest(6)
#print(top_six)
'''fig,axes=plt.subplots(1,2)
axes[0].scatter(df[16],df[15],s=10,color='purple',alpha=0.7)
axes[0].set(xlabel='compactness error',ylabel='smoothness error')
axes[1].scatter(df[19],df[15],s=10,alpha=0.7)
axes[1].set(xlabel='symmetry error',ylabel='smoothness error')
'''
'''fig=plt.figure(figsize=(18,15))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(df[15],df[16],df[19],c='b',alpha=0.7,s=5)
ax.set_xlabel('smoothness error')
ax.set_ylabel('compactness error')
ax.set_zlabel('symmetry error')
plt.show()'''
'''fig=plt.figure(figsize=(18,15))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(df[10],df[3],df[22],c='b',alpha=0.7,s=5)
ax.set_xlabel('radius error')
ax.set_ylabel('mean area')
ax.set_zlabel('worst perimeter')
plt.show()'''
x=pd.DataFrame(np.c_[df[12],df[13]],columns=['mean area','perimeter error'])
Y=df[10]
x_train,x_test,Y_train,Y_test=train_test_split(x,Y,test_size=0.3,random_state=5)
polynomialFeatures=PolynomialFeatures(degree=2)
x_train_poly=polynomialFeatures.fit_transform(x_train)
x_test_poly=polynomialFeatures.fit_transform(x_test)
model2=LinearRegression()
model2.fit(x_train_poly,Y_train)
R_2=model2.score(x_test_poly,Y_test)
#print(model2.intercept_)
#print(model2.coef_)
#print(R_2)
z=lambda x,y:(model2.intercept_+model2.coef_[1]*x+model2.coef_[2]*y+model2.coef_[3]*x**2+model2.coef_[4]*x*y+model2.coef_[5]*y**2)
fig=plt.figure(figsize=(18,15))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(x['mean area'],x['perimeter error'],Y,c='b')
ax.set_xlabel("mean area")
ax.set_ylabel("perimeter error")
ax.set_zlabel("radius error")
ax.set_title("Radius Error about MA and PE")
x_surf=np.arange(0,30,1)
y_surf=np.arange(0,500,1)
x_surf,y_surf=np.meshgrid(x_surf,y_surf)
ax.plot_surface(x_surf,y_surf,z(x_surf,y_surf),alpha=0.4)
plt.show()
'''model=LinearRegression()
model.fit(x_train,Y_train)
pre=model.predict(x_test)
R_2=model.score(x_test,Y_test)
#print(R_2)'''
'''plt.scatter(Y_test,pre,s=10,c='purple',alpha=0.7)
plt.xlabel('Actual')
plt.ylabel('predictions')
#plt.show()
print(model.intercept_)
print(model.coef_)'''
'''fig=plt.figure(figsize=(18,15))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(df[10],df[12],df[13],c='b',alpha=0.7,s=5)
ax.set_xlabel('radius error')
ax.set_ylabel('perimeter error')
ax.set_zlabel('area error')
ax.axis([0,3,0,25])
x_surf=np.arange(0,25,1)
Y_surf=np.arange(0,200,1)
x_surf,Y_surf=np.meshgrid(x_surf,Y_surf)
z=lambda x,y:(model.intercept_+model.coef_[0]*x+model.coef_[1]*y)
ax.plot_surface(z(x_surf,Y_surf),x_surf,Y_surf,rstride=1,cstride=1,alpha=0.4)
plt.show()
#print(df[10],df[12],df[13],end='\n')
pickle.dump(model,open('modelForBreastCancerDataset.sav','wb'))'''