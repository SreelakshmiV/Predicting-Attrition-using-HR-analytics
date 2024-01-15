#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load your data
data = pd.read_csv(r"C:\Users\Sreelakshmi V\OneDrive\Desktop\ibm data set with feature elimination.csv")

# Set the aesthetic style of the plots
sns.set_style("whitegrid")
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'

# Filter out only categorical variables
categorical_cols = data.select_dtypes(include=['object', 'bool']).columns

# Create a figure for the subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 15), facecolor='black')

# Flatten axes array for easy iterating
axes = axes.flatten()

# Loop through the categorical columns and create pie charts
for ax, col in zip(axes, categorical_cols):
    # Get value counts of the column and plot
    data[col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90, colors=sns.color_palette('bright'))
    ax.set_title(col, color='white')  # Set the title for each pie chart
    ax.set_ylabel('')

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = r"C:\Users\Sreelakshmi V\OneDrive\Documents\Sree\notes\predictive modeling\1. assesment\Assignment 2\Python Images\categorical_charts_dashboard.png"
plt.savefig(output_path)

# Display the plot
plt.show()

