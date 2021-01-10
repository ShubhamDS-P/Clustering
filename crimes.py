# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:31:07 2021

@author: Shubham
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
crimes = pd.read_csv("D:\\Data Science study\\Documents\\Assignments\\Clustering\\crime_data.csv")
crimes

# Lets create the normalization function
def norm_func(i):
    x = (i-i.mean()) / (i.std())
    return(x)
    
# Now we will normalize the dataframe. We will only consider the numerical part of the data.
crimesdata = crimes.iloc[:,1:]
crimesdata_norm = norm_func(crimesdata)
crimesdata_norm
crimesdata_norm.describe()
    
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

# converting the data into numpy array format
array = np.array(crimesdata_norm)
array

# Now we will use the linkage function to create and measure the distance between the all the records

z = linkage(crimesdata_norm, method='complete', metric='euclidean')
plt.figure(figsize=(15,5));plt.title('Heirarchical clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
        z,
        leaf_rotation=0.,
        leaf_font_size=8.,
)
plt.show

# Now we will use Agglomerative clustering to find out and select different clusters

from sklearn.cluster import AgglomerativeClustering

h_complete= AgglomerativeClustering(n_clusters=4, linkage = 'complete', affinity = 'euclidean').fit(crimesdata_norm)

#
h_complete.labels_

#Converting h_complet.labels into series from arrays

cluster_labels = pd.Series(h_complete.labels_)

#Creating new clumn named clust and assign the cluster_labels to it.
hcrimes = crimes.copy()
hcrimes['clust'] = cluster_labels  

#Now lets shift the position of the columns for our better visual and understandings

hcrimes = crimes.iloc[:,[5,0,1,2,3,4,]]

hcrimes.head()  # First five observations

#Lets find an aggregate mean of each cluster
hcrimes.groupby(hcrimes.clust).mean()
result = hcrimes.groupby(hcrimes.clust).mean()

# Creating a CSV file
hcrimes.to_csv("hcrimes.csv", index = False)

# To find where the file is being saved
import os
os.getcwd()

# To change directory
os.chdir("D:\\Data Science study\\assignment\\Sent\\7")

# we have used the Hierarchical clustering method for above clusters
# Now we will try using the Non hierarchical clustering which is also called 
#  as the K-means clustering to find different clusters and see how much it differes from the Hierarchical clustering

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Lets create the screw plot o elbow curve

k = list(range(2,15))
k
TWSS = [] # variable for storing the Total Within Sum of Squares

# We are creating the function to determine the apropriate k value for our clusters

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(crimesdata_norm)
    WSS = [] # Variable for storing within sum of squared values for clusters
    for j in range (i):
        WSS.append(sum(cdist(crimesdata_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,crimesdata_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

# Plotting scree plot
plt.plot(k,TWSS,'ro-');plt.xlabel("No.of Clusters");plt.ylabel("Total within SS");plt.xticks(k)

# By looking at the above scree plot we can say that 4 clusters will be optimum for our data
model = KMeans(n_clusters = 4)

model.fit(crimesdata_norm)

model.labels_

md = pd.Series(model.labels_) # adding cluster labels to the table 'md'

kcrimes = crimes    # Creating kcrimes for the kmeans clustering datasets

kcrimes['clust'] = md # tranferring vamlues of md to newly creadted clust column in kcrimes

kcrimes.head() # calling top 5 rows

kcrimes = kcrimes.iloc[:,[5,0,1,2,3,4]]  # shifting positions of the columns in the kcrimes

kcrimes

kcrimes.iloc[:,1:7].groupby(kcrimes.clust).mean()  # Taking clustervise mean of all the columns

kcrimes.to_csv("kcrimes.csv", index = False)
# Identifying working directory

import os
os.getcwd()

# Changing working directory
os.chdir("D:\\Data Science study\\assignment\\Sent\\7")

os.getcwd()
