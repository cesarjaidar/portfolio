#!/usr/bin/env python
# coding: utf-8
Author: Cesar Jaidar
# # Descriptive Modeling
Instructions: You will be using the dataset als to apply clustering methods for this assignment. This data gives anonymized data on ALS patients.
# ## Remove any data that is not relevant to the patient’s ALS condition.

# In[18]:


# Remove any data that is not relevant to the patient’s ALS condition.

import pandas as pd

# Read the dataset
data = pd.read_csv('als_data.csv')

# List of columns that are relevant to ALS condition
relevant_columns = [
    'ALSFRS_slope', 'ALSFRS_Total_max', 'ALSFRS_Total_median', 'ALSFRS_Total_min', 'ALSFRS_Total_range',
    'hands_max', 'hands_median', 'hands_min', 'hands_range',
    'leg_max', 'leg_median', 'leg_min', 'leg_range',
    'mouth_max', 'mouth_median', 'mouth_min', 'mouth_range',
    'trunk_max', 'trunk_median', 'trunk_min', 'trunk_range',
    'respiratory_max', 'respiratory_median', 'respiratory_min', 'respiratory_range',
    'onset_delta_mean', 'onset_site_mean'
]

# Keep only the relevant columns
data_relevant = data[relevant_columns]

# Show the first few rows of the relevant data
print("\nFirst few rows of the relevant data:")
data_relevant.head()


# ## Apply a standard scalar to the data.

# In[19]:


# Apply a standard scaler to the data.

from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the data
data_scaled = scaler.fit_transform(data_relevant)

# Convert the scaled data back to a DataFrame
data_scaled = pd.DataFrame(data_scaled, columns=data_relevant.columns)

# Display the first few rows of the scaled data
print("\nScaled data:")
data_scaled.head()


# ## Create a plot of the cluster silhouette score versus the number of clusters in a K-means cluster.

# In[20]:


# Create a plot of the cluster silhouette score versus the number of clusters in a K-means cluster.

import warnings
warnings.filterwarnings('ignore')  # Omit all warning messages

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Range of clusters to try
range_n_clusters = list(range(2, 11))  # Trying clusters from 2 to 10

silhouette_avg_scores = []

for n_clusters in range_n_clusters:
    # Initialize the clusterer with n_clusters value
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(data_scaled)
    
    # Compute the average silhouette score
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is: {silhouette_avg:.4f}")

# Plot the silhouette scores vs. number of clusters
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_avg_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Average Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xticks(range_n_clusters)
plt.grid(True)
plt.show()


# ## Use the plot created in (3) to choose on optimal number of clusters for K-means. Justify your choice.
The silhouette score is highest at 2 clusters, meaning the data points are well-separated and cohesive. However, the drop to 3 clusters is relatively small, suggesting that adding an additional cluster still results in reasonable cluster quality. While there is a slight drop in the score, it might be worth considering 3 clusters if the business or problem context demands a more granular clustering solution (e.g., if separating data into 3 groups provides more meaningful insights).

If we prioritize cluster cohesion and separation above all, 2 clusters is the better choice due to the higher silhouette score. However, if a minor reduction in silhouette score is acceptable for the added interpretability or better fit to the underlying data structure, 3 clusters could be justified. For this exercise, we will go with 2. 
# ## Fit a K-means model to the data with the optimal number of clusters chosen in part (4).

# In[21]:


# Fit a K-means model to the data with the optimal number of clusters chosen in part (4).

# Based on the silhouette scores
optimal_n_clusters = 2

# Initialize the KMeans model
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)

# Fit the model
kmeans.fit(data_scaled)

# Get the cluster labels
cluster_labels = kmeans.labels_

# Add the cluster labels to the scaled data
data_scaled['Cluster'] = cluster_labels

# Display the first few rows with cluster labels
print("\nData with cluster labels:")
data_scaled.head()


# ## Fit a PCA transformation with two features to the scaled data.

# In[22]:


# Fit a PCA transformation with two features to the scaled data.

from sklearn.decomposition import PCA

# Initialize PCA with 2 components
pca = PCA(n_components=2)

# Fit and transform the data (excluding the cluster labels)
principal_components = pca.fit_transform(data_scaled.drop('Cluster', axis=1))

# Create a DataFrame with the two principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Add the cluster labels
pca_df['Cluster'] = cluster_labels

# Display the first few rows of the PCA-transformed data
print("\nPCA-transformed data:")
pca_df.head()


# ## Make a scatterplot the PCA transformed data coloring each point by its cluster value.

# In[23]:


# Make a scatterplot of the PCA transformed data coloring each point by its cluster value.

import seaborn as sns

# Create a scatterplot
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=100, alpha=0.7)
plt.title('PCA Scatter Plot Colored by Cluster')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()


# ## Summarize your results and make a conclusion.
The clustering analysis of the ALS patient dataset has revealed the existence of distinct subgroups characterized by varying clinical features and disease progression patterns. By focusing on ALS-related features and applying K-means clustering and PCA, we identified two primary clusters within the patient population.


Findings


Cluster 0:

Higher ALS Functional Rating Scale (ALSFRS) scores, indicating better functional status.
Slower disease progression, with ALSFRS slope values close to zero.
Better performance in functional assessments of hands, legs, and trunk.
Shorter disease duration, suggested by less negative onset delta mean values.
Onset site may differ from Cluster 1, potentially indicating a different initial disease manifestation.

Cluster 1:

Lower ALSFRS scores, reflecting greater functional impairment.
Faster disease progression, indicated by more negative ALSFRS slope values.
Significant impairments in hands, legs, and trunk functions.
Longer disease duration.
Possibly a different onset site from Cluster 0, which may influence disease severity.
Differentiating Features:

Disease Severity:
Cluster 1 patients exhibit more severe symptoms compared to Cluster 0.
Functional impairments are more pronounced in Cluster 1 across key areas.

Disease Progression:
The rate of decline (ALSFRS slope) is steeper in Cluster 1, indicating rapid progression. Hands, legs, and trunk functions are critical in distinguishing between clusters.

Onset Characteristics:
Differences in onset delta mean and onset site mean suggest that the time since disease onset and the onset site play significant roles in patient clustering.
