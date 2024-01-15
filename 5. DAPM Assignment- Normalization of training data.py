#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data from the Excel file
file_path = r'C:\Users\Sreelakshmi V\OneDrive\Documents\Sree\notes\predictive modeling\1. assesment\Assignment 2\Python Files\training_data.xlsx'
data = pd.read_excel(file_path)

# Initialize the Min-Max Scaler
scaler = MinMaxScaler()

# Fit the scaler to the data and transform it
normalized_data = scaler.fit_transform(data)

# Convert the normalized data back to a DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=data.columns)

# Round the values to at most two decimal places
normalized_df = normalized_df.applymap(lambda x: round(x, 2) if not x.is_integer() else int(x))

# Save the normalized data to a CSV file
output_path = r'C:\Users\Sreelakshmi V\OneDrive\Documents\Sree\notes\predictive modeling\1. assesment\Assignment 2\Python Files\normalized_training_data.csv'
normalized_df.to_csv(output_path, index=False)

print(f'Normalized data saved to {output_path}')

