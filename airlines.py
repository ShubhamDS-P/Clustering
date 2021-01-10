# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 12:21:48 2021

@author: Shubham
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Calling xlsx file into the invironment

xls= pd.ExcelFile('D:\Data Science study\Documents\Assignments\Clustering\EastWestAirlines.xlsx')
airlines = pd.read_excel(xls,'data')
airlines
# Lets create normalizaion function

def norm_func(i):
    x = (i-i.mean()) / (i.std())
    return(x)

# Lets normalize the airlines dataframe
airlinesdata = airlines.iloc[:,1:]   # Deleting the first column which we don't required

airlinesdata_norm = norm_func(airlinesdata)  #Normalizing data

airlinesdata_norm.head()  #taking a look at the top five rows

airlinesdata_norm.describe() # Summary of the data

# Now we will use the linkage function to measure the distance between all the records

z = linkage(airlinesdata_norm, method = 'complete', metric = 'euclidean')
plt.figure(figsize = (15,5));plt.title("Heirarchical Clustering Dendrogram");plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(
        z,
        leaf_rotation = 0.,
        leaf_font_size = 8.,
)
plt.show()

# For this data set we can say that the dendrogram doesn't suit it because of its larg size
# But stil for further procession we can select a fix no. by looking at the dendrogram
# And I think the 9 clusters will suffice for that

# Lets use Agglomerative clustering 
h_complete = AgglomerativeClustering(n_clusters = 9,linkage = 'complete', affinity = 'Euclidean').fit(airlinesdata_norm)
h_complete.labels_  # shows cluster numbers

# Converting h_complete.labels from arrays into series
airlines_labels = pd.Series(h_complete.labels_)

# creating new final dataset and adding clust column to it

h_airlines = airlines.copy()
h_airlines['clust'] = airlines_labels

# Shifting the clust column in the dataframe

h_airlines.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]

# Checking clustervise mean of the columns

result = h_airlines.groupby(h_airlines.clust).mean()
result

# Creating csv of the final dataframe
import os # Importing os
os.getcwd() # Getting the working directry
os.chdir("D:\\Data Science study\\assignment\\Sent\\7")

h_airlines.to_csv("h_airlines", index = False)  # saving the file


# we have used the Hierarchical clustering method for above clusters
# Now we will try using the Non hierarchical clustering which is also called 
#  as the K-means clustering to find different clusters and see how much it differes from the Hierarchical clustering


# Lets create the screw plot o elbow curve

k =list(range(2,15))
k
TWSS = [] # variable for storing the Total Within Sum of Squares

# We are creating the function to determine the apropriate k value for our clusters

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(airlinesdata_norm)
    WSS = [] # Variable for storing within sum of squared values for clusters
    for j in range (i):
        WSS.append(sum(cdist(airlinesdata_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,airlinesdata_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

# Plotting scree plot
plt.plot(k,TWSS,'ro-');plt.xlabel('No. of Clusters');plt.ylabel('Total within SS');plt.xticks(k)

# From the graph we an say that the most optimal k value should be 10
# So lets take the k as 10

model = KMeans(n_clusters = 10)

model.fit(airlinesdata_norm)

model.labels_  # cluster labels

md = pd.Series(model.labels_) #converting the cluster labels into series dataframe 'md'

k_airlines = airlines.copy()

k_airlines['clust'] = md  # putting the values of the md into clust column

k_airlines.head(10)  # calling top 10 row

k_airlines.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]  # shifting clust column to first position

k_airlines

k_airlines.iloc[:,1:12].groupby(k_airlines.clust).mean()  # Taking clustervise mean of all the columns

os.getcwd() # Checking current working directry

# Creating the final csv file
k_airlines.to_csv("K_airlines.csv", index = False)
