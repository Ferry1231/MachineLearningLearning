#加成与销售总量的多元回归
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.datasets import make_regression,make_blobs,make_circles#some magical datasets generators
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import pickle

data1=pd.read_excel(r"C:\Users\D\Documents\Tencent Files\3185220995\FileRecv\附件2_d.xlsx")
data2=pd.read_excel(r"C:\Users\D\Documents\Tencent Files\3185220995\FileRecv\品类-日变化.xlsx",sheet_name="Sheet3")
data3=pd.read_excel(r"C:\Users\D\Documents\Tencent Files\3185220995\FileRecv\品类日利润.xlsx",sheet_name="Sheet3")
data4=pd.read_excel(r"C:\Users\D\Documents\Tencent Files\3185220995\FileRecv\加成定价.xlsx")
data2.columns=[1,2,3,4,5,6,'all']
data3.columns=[1,2,3,4,5,6]
data4.columns=[1,2,3,4,5,6]
list_1=list(data1['单品编码'])
list_all=list(data2.loc[721:727:1,'all'])
list_0=[list(data4.loc[721:727:1,i]) for i in range(1,7)]
#print(list_0)


x=pd.DataFrame(np.c_[list_0[0],list_0[1],list_0[2],list_0[3],list_0[4],list_0[5]],columns=data3.columns)
Y=list_all

polynomialFeatures=PolynomialFeatures(degree=1)
x_poly=polynomialFeatures.fit_transform(x)

model2=LinearRegression()
model2.fit(x_poly,Y)
R_2=model2.score(x_poly,Y)
print(model2.intercept_)
print(model2.coef_)
print(R_2)
#z=lambda x1,x2,x3,x4,x5,x6:(model2.intercept_+model2.coef_[1]*x1+model2.coef_[2]*x2+model2.coef_[3]*x3+model2.coef_[4]*x4+model2.coef_[5]*x5+model2.coef_[6]*x6)
