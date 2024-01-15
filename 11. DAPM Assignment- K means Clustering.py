#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install kneed


# In[3]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
import numpy as np

# File paths for your local machine
data_file_path = r"C:\Users\Sreelakshmi V\OneDrive\Documents\Sree\notes\predictive modeling\1. assesment\Assignment 2\Python Files\updated_normalized_training_data.csv"
correlation_matrix_file_path = r"C:\Users\Sreelakshmi V\OneDrive\Documents\Sree\notes\predictive modeling\1. assesment\Assignment 2\Python Files\correlation_matrix.csv"
elbow_curve_save_path = r"C:\Users\Sreelakshmi V\OneDrive\Documents\Sree\notes\predictive modeling\1. assesment\Assignment 2\Python Images\Elbow_Curve.png"
cluster_image_save_path = r"C:\Users\Sreelakshmi V\OneDrive\Documents\Sree\notes\predictive modeling\1. assesment\Assignment 2\Python Images\Cluster_Image.png"

# Reading the data and correlation matrix
data = pd.read_csv(data_file_path)
correlation_matrix = pd.read_csv(correlation_matrix_file_path, index_col=0)

# Removing highly correlated features
threshold = 0.99
columns_to_remove = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) >= threshold:
            colname = correlation_matrix.columns[i]
            columns_to_remove.add(colname)

data_reduced = data.drop(columns=list(columns_to_remove), errors='ignore')

# Selecting features for clustering
features = data_reduced.select_dtypes(include=[np.number])

# Elbow Method to find the optimal number of clusters
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)

# Using KneeLocator to find the elbow point
knee_locator = KneeLocator(range(1, 11), inertia, curve='convex', direction='decreasing')
optimal_clusters = knee_locator.knee
print(f"Optimal number of clusters: {optimal_clusters}")

# Plotting the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig(elbow_curve_save_path)
plt.close()

# Applying K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(features)

# Adding cluster information to the dataset
data_reduced['Cluster'] = clusters

# Performing PCA for dimensionality reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features)
principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
principal_df['Cluster'] = clusters

# Visualizing the clusters using PCA components
plt.figure(figsize=(10, 6))
sns.scatterplot(data=principal_df, x='Principal Component 1', y='Principal Component 2', hue='Cluster', palette='viridis')
plt.title('Cluster Visualization with PCA')
plt.savefig(cluster_image_save_path)
plt.close()


# In[14]:


# Assuming 'kmeans' is your fitted KMeans model and 'features' is your data used for clustering
centroids = kmeans.cluster_centers_

# For each cluster centroid, find the most important features based on their absolute magnitude
for i, centroid in enumerate(centroids):
    sorted_features = sorted(zip(features.columns, centroid), key=lambda x: -abs(x[1]))
    print(f"Cluster {i} important features:")
    for feature, value in sorted_features:
        print(f"{feature}: {value:.2f}")
    print()


# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from IPython.display import display

# Assuming the data is already defined in your existing script
# Here we use dummy data for demonstration purposes
np.random.seed(42)
data = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
optimal_clusters = 4  # Setting the number of clusters to 4

# Selecting features for clustering
features = data.select_dtypes(include=[np.number])

# Applying K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(features)
data['Cluster'] = clusters

# Calculating centroids
centroids = kmeans.cluster_centers_

# Calculating silhouette scores
silhouette_vals = silhouette_samples(features, clusters)

# Creating a DataFrame for centroid coordinates and silhouette scores
centroid_silhouette_df = pd.DataFrame(centroids, columns=[f'feature_{i}' for i in range(centroids.shape[1])])
centroid_silhouette_df['Average Silhouette Score'] = [silhouette_vals[clusters == i].mean() for i in range(optimal_clusters)]

# Plotting the clusters scatter plot
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features)
principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
principal_df['Cluster'] = clusters

plt.figure(figsize=(10, 6))
sns.scatterplot(data=principal_df, x='Principal Component 1', y='Principal Component 2', hue='Cluster', palette='viridis', s=100)
plt.title('Cluster Visualization with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plotting centroids
centroid_components = pca.transform(centroids)
for i, centroid in enumerate(centroid_components):
    plt.scatter(centroid[0], centroid[1], s=200, marker='X', color='red', label=f'Centroid {i+1}' if i == 0 else None)

plt.legend()
plt.show()

# Displaying the centroid and silhouette score table in a formatted way
display(centroid_silhouette_df)


# In[4]:


from sklearn.metrics import silhouette_score

# Your existing code for clustering
# ...

# After fitting kmeans and predicting clusters:
# Calculate silhouette score
silhouette_avg = silhouette_score(features, clusters)
print(f'Silhouette Score: {silhouette_avg:.2f}')

# If you wish to save the silhouette score to a file, you can do so as follows
silhouette_score_path = r"C:\Users\Sreelakshmi V\OneDrive\Documents\Sree\notes\predictive modeling\1. assesment\Assignment 2\Python Files\silhouette_score.txt"

with open(silhouette_score_path, 'w') as f:
    f.write(f'Silhouette Score: {silhouette_avg:.2f}')

