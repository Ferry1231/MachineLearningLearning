#33种单品与销售总量的多元回归
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.datasets import make_regression,make_blobs,make_circles#some magical datasets generators
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import pickle

data1_1=pd.read_excel(r"C:\Users\D\Documents\Tencent Files\3185220995\FileRecv\附件2 (1).xlsx",sheet_name="Sheet2")
data1_3=pd.read_excel(r"C:\Users\D\Documents\Tencent Files\3185220995\FileRecv\附件2 (1).xlsx",sheet_name="Sheet4")
data1_2=pd.read_excel(r"C:\Users\D\Documents\Tencent Files\3185220995\FileRecv\合并(1).xlsx")
list_all=list(data1_1['单品编码'])
list_new=[0 for i in range(49)]
k=0

for i in list_all:
    for j in range(380):
        if data1_2.loc[j,'单品编码']==i:
            list_new[k]+=data1_2.loc[j,'销量千克']*(data1_2.loc[j,'销售单价元千克']-data1_2.loc[j,'批发价格元千克'])
    k+=1
#print(list_new)
dict_1={i:j for i,j in zip(list_all,list_new)}
sorted_dict = dict(sorted(dict_1.items(), key=lambda x: x[1], reverse=True))
top_33 = dict(list(sorted_dict.items())[:33])

#print(top_33)
list_time=np.array([[m+1 for m in range(7)] for i in range(33)])
list_price=np.array([[0 for n in range(7)] for i in range(33)])
list_value=np.array([[0 for m in range(7)] for i in range(33)])
for key in top_33.keys():
    k=0
    for i in range(272):
        if data1_3.loc[i,'单品编码']==key: 
            for j in range(1,8):
                if data1_3.loc[i,'时间']==j:
                    list_price[k][j-1]=data1_3.loc[i,'销售单价(元/千克)']
                    list_value[k][j-1]=data1_3.loc[i,'销售单价(元/千克)']*data1_3.loc[i,'销量(千克)']
    k+=1

x=pd.DataFrame(np.c_[list_price[i] for i in range(33)])

polynomialFeatures=PolynomialFeatures(degree=1)
x_poly=polynomialFeatures.fit_transform(x)

model2=LinearRegression()
model2.fit(x_poly,Y)
R_2=model2.score(x_poly,Y)
print(model2.intercept_)
print(model2.coef_)
print(R_2)
#z=lambda x1,x2,x3,x4,x5,x6:(model2.intercept_+model2.coef_[1]*x1+model2.coef_[2]*x2+model2.coef_[3]*x3+model2.coef_[4]*x4+model2.coef_[5]*x5+model2.coef_[6]*x6)
