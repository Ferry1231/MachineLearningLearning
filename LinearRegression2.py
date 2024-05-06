import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets
import pickle
df=pd.read_csv('./polynomial.csv')
plt.scatter(df.x,df.y,s=10,alpha=0.7)
x=np.array(df.x[0:9])[:,np.newaxis]
y=np.array(df.y[0:9])[:,np.newaxis]
model=LinearRegression()
model.fit(x,y)
R_2=model.score(x,y)
#plt.plot(x,model.predict(x))
#plt.show()
#print(R_2)
polynomialFeatures=PolynomialFeatures(degree=10)
x_poly=polynomialFeatures.fit_transform(x)#return a matrix that includes 1,a,a^2 for a in x
#print(x_poly)
model2=LinearRegression()
model2.fit(x_poly,y)
plt.plot(x,model2.predict(x_poly),c='r')
plt.show()
#print(model2.intercept_)
#print(model2.coef_)
print(model2.score(x_poly,y))
'''for i in range(30):
    polynomialFeatures=PolynomialFeatures(degree=i)
    x_poly=polynomialFeatures.fit_transform(x)
    model2=LinearRegression()
    model2.fit(x_poly,y)
    print(i,":",model2.score(x_poly,y))'''
