import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import pickle
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression,make_blobs,make_circles#some magical datasets generators
heights=[[1.6],[1.65],[1.7],[1.73],[1.8]]
weights=[[60],[65],[72.3],[75],[80]]
model=LinearRegression()
model.fit(X=heights, y=weights)#create a regression model
plt.plot(heights, weights,"*")
plt.plot(heights,model.predict(heights),'r')
plt.title("Weights plotted against heights")
plt.xlabel("Heights/m")
plt.ylabel("Weights/kg")
plt.axis([1.5,1.85,50,95])
plt.grid(True)
plt.show()
weight1=model.predict([[1.85]])[0][0]#to predict(after fit),exactly it's a 2D return value
#print(weight1)
print(model.intercept_)
print(model.coef_)
#test data_1
RSS0=np.sum((np.ravel(weights)-np.ravel(model.predict(heights)))**2)#ravel():2D->1D,continuous array
heights_test=[[1.58],[1.62],[1.69],[1.76],[1.82]]
weights_test=[[58],[63],[72],[73],[85]]
RSS=np.sum((np.ravel(weights_test)-np.ravel(model.predict(heights_test)))**2)
weights_test_mean=np.mean(np.ravel(weights_test))
TSS=np.sum((np.ravel(weights_test)-weights_test_mean)**2)
R_2=1-(RSS/TSS)
#or
R_2_=model.score(heights_test,weights_test)
print("RSS:%.2f"%RSS)
print("TSS:%.2f"%TSS)
print("R_2:%.2f"%R_2)
print("R_2_:%.2f"%R_2_)