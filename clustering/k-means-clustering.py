"""
This code uses K-means clustering algorithm to discover insights from an unlabeled randomly generated dataset,
and to perform customer segmentation.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Setting random seed
np.random.seed(0)

"""
Making random clusters of points by using the "make_blobs" class, using the following parameters:

-n_samples: The total number of points equally divided among clusters.
    *(Value: 5000)

-centers: The number of centers to generate, or the fixed center locations.
    *(Value: [[4, 4], [-2, -1], [2, -3],[1,1]])

-cluster_std: The standard deviation of the clusters.
    *(Value: 0.9)

=Output:
-X: Array of shape [n_samples, n_features]. (Feature Matrix)
    *The generated samples.

-y: Array of shape [n_samples]. (Response Vector)
    *The integer labels for cluster membership of each sample.
"""
# Create clusters of points
X, y = make_blobs(n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

# Show a scatter plot
plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.show()

# Setting up K-Means Clustering Model
"""
The KMeans class has many parameters that can be used, but the following three only will be used:
-init: Initialization method of the centroids.
    *Value: "k-means++"
    **k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.

-n_clusters: The number of clusters to form as well as the number of centroids to generate.
    *(Value: 4 (since there are 4 centers in the randomly generated dataset))

-n_init: Number of time the k-means algorithm will be run with different centroid seeds.
    *(Value: 12)
    **The final results will be the best output of n_init consecutive runs in terms of inertia.
"""
k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)

# Training Model
k_means.fit(X)

# Grabbing the labels for each point in the model using k_means.labels_
k_means_labels = k_means.labels_
print(k_means_labels)

# Getting the coordinates of the cluster centers using k_means.cluster_centers_
k_means_cluster_centers = k_means.cluster_centers_
print(k_means_cluster_centers)

# Creating a Visual Plot
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):
    # Create a list of all data points, where the data points that are
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()

# Training a new k-means model using 3 clusters instead of 4
k_means3 = KMeans(init="k-means++", n_clusters=3, n_init=12)
k_means3.fit(X)
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
plt.show()

# Customer Segmentation
cust_df = pd.read_csv("Cust_Segmentation.csv")
print(cust_df.head())

# Pre-processing
# Address in this dataset is a categorical variable.
# The k-means algorithm is not directly applicable to categorical variables because the Euclidean distance function
# is not really meaningful for discrete variables.
df = cust_df.drop('Address', axis=1)
print(df.head())

# Normalizing over the standard deviation
# Normalization is a statistical method that helps mathematical-based algorithms to interpret features with
# different magnitudes and distributions equally.
X = df.values[:, 1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)

# Modeling
clusterNum = 3
k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

# Assigning the labels to each row in the dataframe
df["Clus_km"] = labels
print(df.head(5))

print(df.groupby('Clus_km').mean())

# Plotting the distribution of customers based on their age and income:
area = np.pi * (X[:, 1]) ** 2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()

# Creating a 3D plot:
fig = plt.figure(1, figsize=(8, 6))
plt.clf()

ax = fig.add_subplot(111, projection='3d')
plt.cla()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(float))

plt.show()
