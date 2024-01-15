#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

# File paths
input_file_path = "C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Files\\normalized_training_data.csv"
output_file_path = "C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Files\\updated_normalized_training_data.csv"

# Read the dataset
df = pd.read_csv(input_file_path)

# Adding new features

# Job Tenure - using 'TotalWorkingYears' as a proxy
df['JobTenure'] = df['TotalWorkingYears']

# Stagnation Period - using 'YearsSinceLastPromotion'
df['StagnationPeriod'] = df['YearsSinceLastPromotion']

# Departmental/Organizational Compa Ratio - using dummy salary data
# Generating random salary data (normalized)
np.random.seed(0)
df['Salary'] = np.random.rand(len(df))

# Assuming departmental and organizational median salaries (normalized)
departmental_median_salary = 0.5  # Hypothetical value
organizational_median_salary = 0.6  # Hypothetical value

# Calculating Compa Ratios
df['DepartmentalCompaRatio'] = df['Salary'] / departmental_median_salary
df['OrganizationalCompaRatio'] = df['Salary'] / organizational_median_salary

# Rounding all decimal values to two places
df = df.round(2)

# Save the updated DataFrame
df.to_csv(output_file_path, index=False)

