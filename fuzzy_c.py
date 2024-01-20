
# coding: utf-8

# In[114]:


import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA
from scipy.spatial.distance import cdist
global MAX
MAX = 100.0
#used for end condition
global Epsilon
Epsilon = 0.01


d = pd.read_csv("Data Sets.csv")
X=d.iloc[:, 0].values
Y=d.iloc[:,1].values


# In[133]:


def fcm(data, n_clusters=1, n_init=30, m=2, max_iter=300, tol=1e-16):

    min_cost = np.inf
    for iter_init in range(n_init):

        # Randomly initialize centers
        centers = data[np.random.choice(
            data.shape[0], size=n_clusters, replace=False
            ), :]

        # Compute initial distances
        # Zeros are replaced by eps to avoid division issues
        dist = np.fmax(
            cdist(centers, data, metric='sqeuclidean'),
            np.finfo(np.float64).eps
        )
        itr=0
        for iter1 in range(max_iter):
            
            # Compute memberships       
            u = (1 / dist) ** (1 / (m-1))
            um = (u / u.sum(axis=0))**m

            # Recompute centers
            prev_centers = centers
            centers = um.dot(data) / um.sum(axis=1)[:, None]

            dist = cdist(centers, data, metric='sqeuclidean')
            itr+=1
            if np.linalg.norm(centers - prev_centers) < tol:
                break

        # Compute cost
        cost = np.sum(um * dist)
        if cost < min_cost:
            min_cost = cost
            min_centers = centers
            mem = um.argmax(axis=0)

    return min_centers, mem,cost,itr




# In[134]:


points=np.array(list(zip(X,Y)))
centers, mem ,cost,iteration= fcm(points, n_clusters=2, n_init=30, m=2, max_iter=300, tol=1e-16)
print(iteration)                


# In[135]:


iterations=[]

for i in range(2,12):
    centers, mem ,cost,iteration= fcm(points, n_clusters=i, n_init=30, m=2, max_iter=300, tol=1e-16)
    iterations.append(iteration)
obj=[]

for i in range(2,12):
    centers, mem ,cost,iteration= fcm(points, n_clusters=i, n_init=30, m=2, max_iter=300, tol=1e-16)
    obj.append(cost)



# In[147]:


plt.plot(range(2,12),obj,label='objective function')  
plt.plot(range(2,12),iterations,label='no of iterations')


# In[137]:


centers, mem ,cost,iteration= fcm(points, n_clusters=6, n_init=30, m=2, max_iter=300, tol=1e-16)
colors=['r','b','g','y','c','m','k','black','brown']

f=lambda x: colors[int(x)]

mem=list(map(f,mem))


# In[143]:


plt.scatter(points[:,0],points[:,1],c=mem,s=20)

plt.scatter(centers[:,0],centers[:,1],c='black',s=50,marker='*')

