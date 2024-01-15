#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Reading the dataset
file_path = "C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Files\\updated_normalized_training_data.csv"
df = pd.read_csv(file_path)

# Calculating the correlation matrix
correlation_matrix = df.corr()

# Saving the correlation matrix to a CSV file
correlation_matrix.to_csv("C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Files\\correlation_matrix.csv")

# Creating a heatmap for the correlation matrix
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
plt.title("Correlation Matrix Heatmap")

# Saving the heatmap to a file
plt.savefig("C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Images\\correlation_heatmap.png")

# Display the heatmap
plt.show()

