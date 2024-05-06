import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics#Confusion matrix
from sklearn.metrics import roc_curve,auc
from mpl_toolkits.mplot3d import Axes3D
def logit(x):
    return np.log(x/(1-x))
def sigmond(x):
    return 1/(1+math.exp(-logRegress.intercept_[0]-logRegress.coef_[0][0]*x))
'''x=np.arange(0.001,0.999,0.001)
y=[logit(i) for i in x]
x2=np.arange(-10,10,0.001)
y2=[sigmond(i) for i in x2]
fig,axes=plt.subplots(1,2)
axes[0].plot(x,y,alpha=0.7)
axes[0].set_xlabel("Probability")
axes[0].set_ylabel("L-logit")
axes[1].plot(x2,y2,alpha=0.7,c='r')
axes[1].set_xlabel("L-logit")
axes[1].set_ylabel("Probability")
plt.show()'''
cancer=datasets.load_breast_cancer()#数据集导入
df=pd.DataFrame(cancer.data)
X=[]
for target in range(2):
    X.append([[],[],[]])
    for i in range(len(cancer.data)):
        if cancer.target[i]==target:
                X[target][0].append(cancer.data[i][0])
                X[target][1].append(cancer.data[i][1])
                X[target][2].append(cancer.data[i][2])
print(df)
#df.to_csv('breast_cancer.csv')#only type"DataFrame" can be saved as CSV
colors=['r','b']
colours={0:'r',1:'b'}
'''fig=plt.figure(figsize=(18,15))
ax=fig.add_subplot(111,projection='3d')
for target in range(2):
    ax.scatter(X[target][0],X[target][1],X[target][2],c=colors[target])
ax.set(xlabel="mean radius",ylabel="mean texture",zlabel="mean perimeter")
plt.show()'''
x=cancer.data[:,0]
y=cancer.target
plt.scatter(x,y,edgecolors=pd.DataFrame(cancer.target)[0].apply(lambda x:colours[x]),s=5)
#plt.show()
logRegress=LogisticRegression()
logRegress.fit(X=np.array(x).reshape(len(x),1),y=y)
#print(logRegress.intercept_)
#print(logRegress.coef_)
x1=np.arange(0,30,0.001)
y1=[sigmond(n) for n in x1]
plt.plot(x1,y1,alpha=0.7,lw=1.0)#get the curve
#plt.scatter(x1,y1,s=5,c='b',alpha=0.5)
#plt.show()
#print(logRegress.predict_proba([[20]]))#2D array input,maybe there were several inputs before
train_set,test_set,train_labels,test_labels=train_test_split(cancer.data,cancer.target,test_size=0.25,random_state=1)
x2=train_set[:,0:30]
y2=train_labels
logRegress2=LogisticRegression()
logRegress2.fit(x2,y2)
#print(logRegress2.intercept_)
#print(logRegress2.coef_)
predsProb=pd.DataFrame(logRegress2.predict_proba(X=test_set))
predsProb.columns=["Malignant","Benign"]
preds=logRegress2.predict(X=test_set)
predsClass=pd.DataFrame(logRegress2.predict(X=test_set))
predsClass.columns=["Prediction"]
originalResult=pd.DataFrame(test_labels)
originalResult.columns=["Original Result"]
result=pd.concat([predsProb,predsClass,originalResult],axis=1)#when axis=1,they will place in rows
#print(result.head(20))#the first 10 of the table
#print("---Confusion Matrix---")
#print(pd.crosstab(preds,test_labels))
#print(metrics.confusion_matrix(y_true=test_labels,y_pred=preds))
#print(metrics.classification_report(y_true=test_labels,y_pred=preds))

