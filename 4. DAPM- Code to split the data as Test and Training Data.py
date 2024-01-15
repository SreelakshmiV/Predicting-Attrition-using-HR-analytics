#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from the given path
file_path = "C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Files\\processed_data.xlsx"
data = pd.read_excel(file_path)

# Split the dataset into features and target variable using 'Attrition' as the target
X = data.drop('Attrition', axis=1)  # Features
y = data['Attrition']  # Target variable

# Split the data into a training set and a test set (25% for test and 75% for training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Concatenate the features and target variable for training and test sets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Define the directory to save the combined data files
save_dir = "C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Files\\"

# Save the combined training data
train_data_path = save_dir + 'training_data.xlsx'
train_data.to_excel(train_data_path, index=False)

# Save the combined test data
test_data_path = save_dir + 'test_data.xlsx'
test_data.to_excel(test_data_path, index=False)

