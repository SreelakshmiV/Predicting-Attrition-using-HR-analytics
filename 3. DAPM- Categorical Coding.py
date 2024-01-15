#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\Sreelakshmi V\OneDrive\Desktop\ibm data set with feature elimination.csv")

# Select numerical columns for proximity analysis
numerical_df = df.select_dtypes(include=['int64', 'float64'])

# Calculate the pairwise distance matrix
distance_matrix = pairwise_distances(numerical_df, metric='euclidean')

# Convert the distance matrix to a DataFrame for better readability
distance_df = pd.DataFrame(distance_matrix, index=numerical_df.index, columns=numerical_df.index)

# Set the dark background style for the heatmap
plt.style.use('dark_background')

# Plotting the heatmap of the distance matrix with a red color gradient
plt.figure(figsize=(10, 8))
sns.heatmap(distance_df, cmap='hot', square=True)
plt.title('Proximity Analysis Heatmap', color='white')
plt.show()


# In[13]:


pip install XlsxWriter


# In[17]:


import pandas as pd
import xlsxwriter


# Define a mapping for categorical variables to numeric values
categorical_mapping = {
    'BusinessTravel': {'Travel_Rarely': 0, 'Travel_Frequently': 1, 'Non-Travel': 2},
    'Department': {'Sales': 0, 'Research & Development': 1, 'Human Resources': 2},
    'EducationField': {'Life Sciences': 0, 'Medical': 1, 'Other': 2, 'Technical Degree': 3, 'Human Resources': 4},
    'Gender': {'Female': 0, 'Male': 1},
    'JobRole': {'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 2, 
                'Manufacturing Director': 3, 'Healthcare Representative': 4, 'Manager': 5, 
                'Sales Representative': 6, 'Research Director': 7, 'Human Resources': 8},
    'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2},
    'OverTime': {'No': 0, 'Yes': 1}
}

# Replace categorical variables with numeric values
df.replace(categorical_mapping, inplace=True)

# Create a Pandas Excel writer using XlsxWriter as the engine.
with pd.ExcelWriter('processed_data.xlsx', engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Sheet1', index=False)

    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

    # Define a format for headers with border.
    header_format = workbook.add_format({'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter'})

    # Apply the header format.
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)

# Save the Excel file to your computer as 'processed_data.xlsx' with borders and headers.
print("Excel file 'processed_data.xlsx' has been created with borders and headers.")


# In[19]:


import os
print(os.getcwd())


# In[29]:


import pandas as pd
import xlsxwriter

# Define a mapping for categorical variables to numeric values
categorical_mapping = {
    'BusinessTravel': {'Travel_Rarely': 0, 'Travel_Frequently': 1, 'Non-Travel': 2},
    'Department': {'Sales': 0, 'Research & Development': 1, 'Human Resources': 2},
    'EducationField': {'Life Sciences': 0, 'Medical': 1, 'Other': 2, 'Technical Degree': 3, 
                       'Human Resources': 4, 'Marketing': 5},  # Include 'Marketing' here
    'Gender': {'Female': 0, 'Male': 1},
    'JobRole': {'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 2, 
                'Manufacturing Director': 3, 'Healthcare Representative': 4, 'Manager': 5, 
                'Sales Representative': 6, 'Research Director': 7, 'Human Resources': 8},
    'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2},
    'OverTime': {'No': 0, 'Yes': 1},
    'Attrition': {'No': 0, 'Yes': 1}  # Include 'Attrition' here
}

# Replace categorical variables with numeric values
df.replace(categorical_mapping, inplace=True)

# Specify the full path where you want to save the Excel file
output_path = "C:/Users/Sreelakshmi V/OneDrive/Documents/Sree/notes/predictive modeling/1. assesment/Assignment 2/Python Files/processed_data.xlsx"

# Create a Pandas Excel writer using XlsxWriter as the engine.
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Sheet1', index=False)

    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

    # Define a format for headers with border.
    header_format = workbook.add_format({'bold': True, 'border': 1, 'align': 'center', 'valign': 'vcenter'})

    # Apply the header format.
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)

print(f"Excel file '{output_path}' has been created with borders and headers.")

