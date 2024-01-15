#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Load the dataset
file_path = 'C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Files\\normalized_training_data.csv'
data = pd.read_csv(file_path)

# Calculate the Euclidean distance matrix
euclidean_dist_matrix = squareform(pdist(data, 'euclidean'))

# Round the matrix to two decimal places
euclidean_dist_matrix = np.around(euclidean_dist_matrix, decimals=2)

# Save the Euclidean distance matrix as a CSV file
output_csv_path = 'C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Files\\euclidean_dist_matrix.csv'
pd.DataFrame(euclidean_dist_matrix).to_csv(output_csv_path, index=False)

# Create and save the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(euclidean_dist_matrix, cmap='viridis')
plt.title('Euclidean Distance Matrix Heatmap')
output_image_path = 'C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Images\\euclidean_heatmap.png'
plt.savefig(output_image_path)
plt.close()

