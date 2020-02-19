import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df = pd.read_csv('Customers.csv')
print df.head()
'''
   CustomerID   Genre  Age  Annual Income (k$)  Spending Score (1-100)
0           1    Male   19                  15                      39
1           2    Male   21                  15                      81
2           3  Female   20                  16                       6
3           4  Female   23                  16                      77
4           5  Female   31                  17                      40'''

x= df.iloc[:,3:].values
print x[:6]
'''
[[15 39]
 [15 81]
 [16  6]
 [16 77]
 [17 40]
 [17 76]]'''

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('Elbow_method')
plt.xlabel('no. of clusters')
plt.ylabel('wcss')
plt.savefig('Elbow_curve')
plt.show()

model = KMeans(n_clusters=5,init='k-means++',random_state=0)
pred = model.fit_predict(x)
print pred
''' here are our predicted values :
[4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4
 3 4 3 4 3 4 1 4 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 2 0 2 1 2 0 2 0 2 1 2 0 2 0 2 0 2 0 2 1 2 0 2 0 2
 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0
 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2]'''

# Visualizing clusters

plt.scatter(x[pred==0,0],x[pred==0,1],s=100,c='red',label='Cluster1')
plt.scatter(x[pred==1,0],x[pred==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(x[pred==2,0],x[pred==2,1],s=100,c='green',label='Cluster3')
plt.scatter(x[pred==3,0],x[pred==3,1],s=100,c='cyan',label='Cluster4')
plt.scatter(x[pred==4,0],x[pred==4,1],s=100,c='magenta',label='Cluster5')



#plot cluster Centroids

plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=200,c='yellow',label='Centroids')

plt.title('Customers_clusters')
plt.xlabel('no. of clusters')
plt.ylabel('spending')
plt.legend()
plt.savefig('clusters_of_custemers')
plt.show()
