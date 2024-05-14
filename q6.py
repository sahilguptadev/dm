#!/usr/bin/env python
# coding: utf-8
Q6. Use Simple Kmeans, DBScan, Hierachical clustering algorithms for clustering. Compare the performance of clusters by changing the parameters involved in the algorithms.
# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

df1 = pd.read_csv(r'Country-data.csv')
df2 = pd.read_csv(r'wine dataset.csv')

#preprocessing data for df1
df1.info()

df1 = df1.drop(columns='country')
df1.info()

#preprocessing for df2
df2.info()



# In[ ]:


#kmeans for df1
# DETERMINING OPTIMAL K VALUE FOR DF1 using Elbow Method

# Calculate WCSS for different k values
sse = {}
for k in range(1, 11): # Adjust the range as needed
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df1)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

# Plot the WCSS values
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal k")
plt.show()


from sklearn.metrics import silhouette_score

# Calculate Silhouette scores for different k values
silhouette_scores = {}
for k in range(2, 11): # Adjust the range as needed
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df1)
    labels = kmeans.labels_
    silhouette_scores[k] = silhouette_score(df1, labels)

# Plot the Silhouette scores
plt.figure()
plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()))
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal k")
plt.show()

# Applying K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(df1)

# Add cluster labels to the original dataframe
df1['Cluster'] = kmeans.labels_

# Display the dataframe with cluster labels
df1

df1['Cluster'].value_counts()


#Visualizing Clusters

cols = df2.columns
num_features = len(cols) - 1  # Number of features (excluding the 'Cluster' column)
num_rows = (num_features - 1) // 4 + 1  # Number of rows needed
num_plots_last_row = (num_features - 1) % 4  # Number of plots in the last row

# Create a figure and subplots grid
fig, axs = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))

# Flatten the subplots grid for easy indexing
axs = axs.flatten()

# Iterate through combinations of features
subplot_idx = 0
for i in range(num_features):
    for j in range(i + 1, num_features):
        if subplot_idx >= num_rows * 4:
            break
        x_label = cols[i]
        y_label = cols[j]

        # Plot the scatterplot in the current subplot
        sns.scatterplot(x=x_label, y=y_label, data=df2, hue='Cluster', ax=axs[subplot_idx])
        axs[subplot_idx].set_title(f'{x_label} vs {y_label}')

        subplot_idx += 1

# Hide empty subplots in the last row
if num_plots_last_row > 0:
    for i in range(num_plots_last_row, 4):
        axs[-i].axis('off')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:


#kmeans for df2

# Calculate WCSS for different k values
sse = {}
for k in range(1, 11): # Adjust the range as needed
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df2)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

# Plot the WCSS values
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal k")
plt.show()

from sklearn.metrics import silhouette_score

# Calculate Silhouette scores for different k values
silhouette_scores = {}
for k in range(2, 11): # Adjust the range as needed
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df2)
    labels = kmeans.labels_
    silhouette_scores[k] = silhouette_score(df2, labels)

# Plot the Silhouette scores
plt.figure()
plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()))
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal k")
plt.show()

# Applying K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(df2)

# Add cluster labels to the original dataframe
df2['Cluster'] = kmeans.labels_

# Display the dataframe with cluster labels
df2


df2['Cluster'].value_counts()

# Visualizing Clusters
cols = df2.columns
num_features = len(cols) - 1  # Number of features
num_rows = (num_features - 1) // 4 + 1  # Number of rows
num_plots_last_row = (num_features - 1) % 4  # Number of plots 

# Create a figure and subplots grid
fig, axs = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))

# Flatten the subplots grid for easy indexing
axs = axs.flatten()

# Iterate through combinations of features
subplot_idx = 0
for i in range(num_features):
    for j in range(i + 1, num_features):
        if subplot_idx >= num_rows * 4:
            break
        x_label = cols[i]
        y_label = cols[j]

        # Plot the scatterplot in the current subplot
        sns.scatterplot(x=x_label, y=y_label, data=df2, hue='Cluster', ax=axs[subplot_idx])
        axs[subplot_idx].set_title(f'{x_label} vs {y_label}')

        subplot_idx += 1

# Hide empty subplots in the last row
if num_plots_last_row > 0:
    for i in range(num_plots_last_row, 4):
        axs[-i].axis('off')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:


#dbscan for df1
df1 = df1.drop(columns='Cluster')

# Apply DBSCAN clustering
clustering = DBSCAN(eps=15, min_samples=10).fit(df1)

# Add the cluster labels to the original dataframe
df1['Cluster'] = clustering.labels_

# Visualize the distribution of clusters
df1

df1['Cluster'].value_counts()

# Visualizing Clusters

cols = df1.columns
num_features = len(cols) - 1  # Number of features
num_rows = (num_features - 1) // 4 + 1  # Number of rows needed
num_plots_last_row = (num_features - 1) % 4  # Number of plots

fig, axs = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))

axs = axs.flatten()

# Iterate through combinations of features
subplot_idx = 0
for i in range(num_features):
    for j in range(i + 1, num_features):
        if subplot_idx >= num_rows * 4:
            break
        x_label = cols[i]
        y_label = cols[j]

        sns.scatterplot(x=x_label, y=y_label, data=df1, hue='Cluster', ax=axs[subplot_idx])
        axs[subplot_idx].set_title(f'{x_label} vs {y_label}')

        subplot_idx += 1

# Hide empty subplots in the last row
if num_plots_last_row > 0:
    for i in range(num_plots_last_row, 4):
        axs[-i].axis('off')

plt.tight_layout()

plt.show()


# In[ ]:


#dbscan for df2
df2 = df2.drop(columns='Cluster')

# Apply DBSCAN clustering
clustering = DBSCAN(eps=5, min_samples=2).fit(df2)

# Add the cluster labels to the original dataframe
df2['Cluster'] = clustering.labels_

# Visualize the distribution of clusters
df2

df2['Cluster'].value_counts()

cols = df2.columns
num_features = len(cols) - 1  # Number of features (excluding the 'Cluster' column)
num_rows = (num_features - 1) // 4 + 1  # Number of rows needed
num_plots_last_row = (num_features - 1) % 4  # Number of plots in the last row

# Create a figure and subplots grid
fig, axs = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))

# Flatten the subplots grid for easy indexing
axs = axs.flatten()

# Iterate through combinations of features
subplot_idx = 0
for i in range(num_features):
    for j in range(i + 1, num_features):
        if subplot_idx >= num_rows * 4:
            break
        x_label = cols[i]
        y_label = cols[j]

        # Plot the scatterplot in the current subplot
        sns.scatterplot(x=x_label, y=y_label, data=df2, hue='Cluster', ax=axs[subplot_idx])
        axs[subplot_idx].set_title(f'{x_label} vs {y_label}')

        subplot_idx += 1

# Hide empty subplots in the last row
if num_plots_last_row > 0:
    for i in range(num_plots_last_row, 4):
        axs[-i].axis('off')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:


#hierarchial clustering for df1
df1 = df1.drop(columns='Cluster')

# Perform hierarchical clustering on all features
linked = linkage(df1, method='ward', metric='euclidean')

# Create a DataFrame from the linked matrix
df_linked = pd.DataFrame(linked, columns=['c1', 'c2', 'distance', 'size'])
df_linked[['c1', 'c2', 'size']] = df_linked[['c1', 'c2', 'size']].astype('int')

# Visualize the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Features')
plt.ylabel('Distance')
plt.show()

# Optionally, you can cut the dendrogram to create clusters
num_clusters = 2  # Adjust based on your analysis of the dendrogram
clusters = fcluster(linked, num_clusters, criterion='maxclust')
df1['Cluster'] = clusters

df1['Cluster'].value_counts()

# Define the number of plots you want per row
plots_per_row = 4

# Define the number of features (excluding the ‘Cluster’ column)
num_features = len(df1.columns) – 1

# Calculate the number of rows needed
num_rows = (num_features – 1) // plots_per_row + 1

# Calculate the number of plots in the last row
num_plots_last_row = (num_features – 1) % plots_per_row

fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(20, 5 * num_rows))

# Flatten the subplots grid for easy indexing
axs = axs.flatten()

# Iterate through combinations of features
subplot_idx = 0
for I in range(num_features):
    for j in range(I + 1, num_features):
        if subplot_idx >= num_rows * plots_per_row:
            break
        x_label = df1.columns[i]
        y_label = df1.columns[j]

        # Plot the scatterplot in the current subplot
        sns.scatterplot(x=x_label, y=y_label, data=df1, hue=’Cluster’, ax=axs[subplot_idx])
        axs[subplot_idx].set_title(f’{x_label} vs {y_label}’)

        subplot_idx += 1

# Hide empty subplots in the last row
if num_plots_last_row > 0:
    for I in range(num_plots_last_row, plots_per_row):
        axs[-i].axis(‘off’)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:


#hierarchial clustering for df2
df2 = df2.drop(columns='Cluster')

# Perform hierarchical clustering on all features
linked = linkage(df2, method='ward', metric='euclidean')

# Create a DataFrame from the linked matrix
df_linked = pd.DataFrame(linked, columns=['c1', 'c2', 'distance', 'size'])
df_linked[['c1', 'c2', 'size']] = df_linked[['c1', 'c2', 'size']].astype('int')

# Visualize the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Features')
plt.ylabel('Distance')
plt.show()

# Optionally, you can cut the dendrogram to create clusters
num_clusters = 2  # Adjust based on your analysis of the dendrogram
clusters = fcluster(linked, num_clusters, criterion='maxclust')
df2['Cluster'] = clusters

df2['Cluster'].value_counts()

cols = df2.columns

# Define the number of plots you want per row
plots_per_row = 4

# Calculate the number of features (excluding the 'Cluster' column)
num_features = len(cols) - 1

# Calculate the number of rows needed
num_rows = (num_features - 1) // plots_per_row + 1

# Create a figure and subplots grid
fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(20, 5 * num_rows))

# Flatten the subplots grid for easy indexing
axs = axs.flatten()

# Iterate through combinations of features
subplot_idx = 0
for i in range(num_features):
    for j in range(i + 1, num_features):
        if subplot_idx >= num_rows * plots_per_row:
            break
        x_label = cols[i]
        y_label = cols[j]

        # Plot the scatterplot in the current subplot
        sns.scatterplot(x=x_label, y=y_label, data=df2, hue='Cluster', ax=axs[subplot_idx])
        axs[subplot_idx].set_title(f'{x_label} vs {y_label}')

        subplot_idx += 1

# Hide empty subplots in the last row
if num_features % plots_per_row != 0:
    for i in range(num_features % plots_per_row, plots_per_row):
        axs[-i].set_visible(False)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

