#k-means聚类，无监督学习,没有标记，即没有target
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn import metrics

df=pd.read_excel("C:\\Users\\D\\Documents\\Tencent Files\\3185220995\\FileRecv\\单品-日变化.xlsx",sheet_name='Sheet2')
k=3#分成3组
X=np.array(list(zip(df.x,df.y)))
kmeans=KMeans(n_clusters=k)
kmeans.fit(X)
labels=kmeans.predict(X)
centroids=kmeans.cluster_centers_

c=['b', 'r', 'y', 'g', 'c', 'm']
colors=[c[i] for i in labels]
silhouette_samples=metrics.silhouette_samples(X,kmeans.labels_)
print(silhouette_samples.mean())

plt.scatter(df.x,df.y,s=15,alpha=0.7,c=colors)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=100,c='black')
plt.show()
#print(labels)
#print(centroids)#k个中心点的坐标