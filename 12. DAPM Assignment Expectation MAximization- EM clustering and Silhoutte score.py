#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# File paths for your local machine
data_file_path = r"C:\Users\Sreelakshmi V\OneDrive\Documents\Sree\notes\predictive modeling\1. assesment\Assignment 2\Python Files\updated_normalized_training_data.csv"
correlation_matrix_file_path = r"C:\Users\Sreelakshmi V\OneDrive\Documents\Sree\notes\predictive modeling\1. assesment\Assignment 2\Python Files\correlation_matrix.csv"
bic_save_path = r"C:\Users\Sreelakshmi V\OneDrive\Documents\Sree\notes\predictive modeling\1. assesment\Assignment 2\Python Images\BIC_Scores.png"
cluster_image_save_path = r"C:\Users\Sreelakshmi V\OneDrive\Documents\Sree\notes\predictive modeling\1. assesment\Assignment 2\Python Images\GMM_Cluster_Image_with_PCA.png"

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

# Standardizing the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data_reduced)

# Determine the optimal number of clusters using BIC
bic_scores = []
n_components_range = range(1, 11)
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(features_scaled)
    bic_scores.append(gmm.bic(features_scaled))

optimal_n_components = n_components_range[bic_scores.index(min(bic_scores))]

# Plotting BIC scores to visualize the optimal number of clusters
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic_scores, marker='o')
plt.title('BIC Scores per Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC Score')
plt.savefig(bic_save_path)
plt.show()

# Use Gaussian Mixture Model for clustering with the optimal number of clusters based on BIC
gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
gmm.fit(features_scaled)
gmm_clusters = gmm.predict(features_scaled)

# Perform PCA for dimensionality reduction for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['Cluster'] = gmm_clusters

# Plotting the clusters from GMM after dimensionality reduction with PCA
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
plt.title('GMM Clusters Visualized with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.savefig(cluster_image_save_path)
plt.show()

print(f"The optimal number of clusters is: {optimal_n_components}")


# In[2]:


import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# File paths for your local machine
data_file_path = "C:/Users/Sreelakshmi V/OneDrive/Documents/Sree/notes/predictive modeling/1. assesment/Assignment 2/Python Files/updated_normalized_training_data.csv"
correlation_matrix_file_path = "C:/Users/Sreelakshmi V/OneDrive/Documents/Sree/notes/predictive modeling/1. assesment/Assignment 2/Python Files/correlation_matrix.csv"

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

# Standardizing the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data_reduced)

# Determine the optimal number of clusters using BIC
bic_scores = []
n_components_range = range(1, 11)
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(features_scaled)
    bic_scores.append(gmm.bic(features_scaled))

optimal_n_components = n_components_range[bic_scores.index(min(bic_scores))]

# Use Gaussian Mixture Model for clustering with the optimal number of clusters based on BIC
gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
gmm.fit(features_scaled)
gmm_clusters = gmm.predict(features_scaled)

# Add cluster information to the original data
data_reduced['Cluster'] = gmm_clusters

# Calculate the mean of the features for each cluster
cluster_feature_means = data_reduced.groupby('Cluster').mean()

# Print the feature means for each cluster
print(cluster_feature_means)

# Create a heatmap of the features for each cluster
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_feature_means.T, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Heatmap of Feature Means for Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.show()


# In[8]:


from sklearn.metrics import silhouette_score

# Assuming 'features_scaled' contains your scaled features
# and 'gmm_clusters' contains your cluster labels from the GMM

# Calculate the Silhouette Score
silhouette_avg = silhouette_score(features_scaled, gmm_clusters)
print(f'Silhouette Score for GMM clustering: {silhouette_avg:.3f}')


# In[6]:


import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# Calculate centroids in the PCA space
centroids_pca = pca.transform(gmm.means_)

# Compute silhouette scores for each sample
silhouette_vals = silhouette_samples(features_scaled, gmm_clusters)

# Calculate average silhouette score for each cluster
cluster_avg_silhouette = []
for i in range(optimal_n_components):
    cluster_avg_silhouette.append(np.mean(silhouette_vals[gmm_clusters == i]))

# Create a DataFrame for centroids and silhouette scores
centroid_silhouette_df = pd.DataFrame({
    'Cluster': range(optimal_n_components),
    'Centroid_PC1': centroids_pca[:, 0],
    'Centroid_PC2': centroids_pca[:, 1],
    'Avg_Silhouette_Score': cluster_avg_silhouette
})

# Display the DataFrame
print(centroid_silhouette_df)


# In[7]:


import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# [Your existing code for data preparation and GMM goes here]

# Use Gaussian Mixture Model for clustering
gmm.fit(features_scaled)
gmm_clusters = gmm.predict(features_scaled)

# Add cluster labels to the original data
data_reduced['Cluster'] = gmm_clusters

# Calculate mean or median of features in each cluster
cluster_feature_stats = data_reduced.groupby('Cluster').mean()  # or use .median()

# Display cluster feature statistics
print(cluster_feature_stats)

# [Your existing PCA and plotting code goes here]

# Perform PCA for dimensionality reduction for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['Cluster'] = gmm_clusters

# Plotting the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
plt.title('GMM Clusters Visualized with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.savefig(cluster_image_save_path)
plt.show()

