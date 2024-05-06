import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics#Confusion matrix
from mpl_toolkits.mplot3d import Axes3D


def logit(x):
    return np.log(x/(1-x))
def sigmond(x):
    return 1/(1+math.exp(-logRegress.intercept_[0]-logRegress.coef_[0][0]*x))


cancer=datasets.load_breast_cancer()#数据集导入
df=pd.DataFrame(cancer.data)

#df.to_csv('breast_cancer.csv')#only type"DataFrame" can be saved as CSV
colors=['r','b']
colours={0:'r',1:'b'}
'''
X=[]
for target in range(2):
    X.append([[],[],[]])#几个参数就几个[]
    for i in range(len(cancer.data)):
        if cancer.target[i]==target:
                X[target][0].append(cancer.data[i][0])
                X[target][1].append(cancer.data[i][1])
                X[target][2].append(cancer.data[i][2])#几个参数就到几
print(df)
fig=plt.figure(figsize=(18,15))
ax=fig.add_subplot(111,projection='3d')
for target in range(2):
    ax.scatter(X[target][0],X[target][1],X[target][2],c=colors[target])
ax.set(xlabel="mean radius",ylabel="mean texture",zlabel="mean perimeter")
plt.show()'''
x=cancer.data[:,0]
y=cancer.target#在该例中，0或1的取值代表良恶性肿瘤
plt.scatter(x,y,edgecolors=pd.DataFrame(cancer.target)[0].apply(lambda x:colours[x]),s=5)
plt.show()

logRegress=LogisticRegression()
logRegress.fit(X=np.array(x).reshape(len(x),1),y=y)

#print(logRegress.intercept_)#截距
#print(logRegress.coef_)#斜率，每一个参数的拟合系数


#print(logRegress.predict_proba([[20]]))#2D array input,maybe there were several inputs before
train_set,test_set,train_labels,test_labels=train_test_split(cancer.data,cancer.target,test_size=0.25,random_state=1)
#train_set=cancer.data;train_labels=cancer.target#把你的初始数据输入
x2=train_set[:,0:30]#如果x个参数，0:30->0:x
y2=train_labels

logRegress2=LogisticRegression()
logRegress2.fit(x2,y2)#生成模型的过程
#print(logRegress2.intercept_)
#print(logRegress2.coef_)

predsProb=pd.DataFrame(logRegress2.predict_proba(X=test_set))#预测0或1的可能性#test_set新的病人的数据集

predsProb.columns=["Malignant","Benign"]

preds=logRegress2.predict(X=test_set)

predsClass=pd.DataFrame(logRegress2.predict(X=test_set))#预测是0还是1

predsClass.columns=["Prediction"]

originalResult=pd.DataFrame(test_labels)#个人学习时候的试验
originalResult.columns=["Original Result"]
result=pd.concat([predsProb,predsClass,originalResult],axis=1)#when axis=1,they will place in rows#按列到一块
print(result.head(20))#the first 10 of the table
#print("---Confusion Matrix---")
#print(pd.crosstab(preds,test_labels))
print(metrics.confusion_matrix(y_true=test_labels,y_pred=preds))#显示有多少0或1预测正确或错误的矩阵
print(metrics.classification_report(y_true=test_labels,y_pred=preds))#精确度