import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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
for key in top_33.keys():
    list_time=np.array([m+1 for m in range(7)]).reshape(-1,1)
    list_price=np.array([0 for n in range(7)]).reshape(-1,1)
    list_value=np.array([0 for m in range(7)]).reshape(-1,1)
    for i in range(272):
        if data1_3.loc[i,'单品编码']==key: 
            for j in range(1,8):
                if data1_3.loc[i,'时间']==j:
                    list_price[j-1]=data1_3.loc[i,'销售单价(元/千克)']
                    list_value[j-1]=data1_3.loc[i,'销售单价(元/千克)']*data1_3.loc[i,'销量(千克)']
    #print(list_value)
    #print(list_price)
    polynomialFeatures=PolynomialFeatures(degree=1)
    price_poly=polynomialFeatures.fit_transform(list_price)
    #print(price_poly.shape)
    #print(len(list_value))
    model=LinearRegression()
    model.fit(price_poly,list_value)
    print(key,'->','intercept:',model.intercept_,'; coef:',model.coef_,'; accuracy:',model.score(price_poly,list_value))
    plt.scatter(list_price,list_value,c='blue',alpha=0.8,s=15)
    plt.plot(list_price,model.predict(price_poly))
    plt.show()
    '''
    best=0
    if(model.score(price_poly,list_value)<0.6):
        if(model.coef_[0][2]<0):
            best=-(model.coef_[0][1]/2*model.coef_[0][2])
        elif(model.coef_[0][2]>=0):
            best=list_price[0][0]
        value=lambda x:model.intercept_+model.coef_[0][1]*x+model.coef_[0][2]*x**2
        if(value(best)/best<2.5):
            print('  ->','price:',list_price[0][0],'; how many:','2.5kg')
        else:
            print('  ->','price:',best,'; how many:',value(best)/best)
    '''
    