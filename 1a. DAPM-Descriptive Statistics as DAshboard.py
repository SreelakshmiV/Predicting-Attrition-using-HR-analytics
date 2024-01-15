#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# Load the dataset
df = pd.read_csv(r"C:\Users\Sreelakshmi V\OneDrive\Desktop\ibm data set with feature elimination.csv")

# Calculate descriptive statistics
descriptive_stats = df.describe().round(2)
descriptive_stats.loc['mode'] = df.mode().iloc[0]
descriptive_stats.loc['variance'] = df.var(numeric_only=True)

# Apply color gradient to the descriptive statistics table
cm = sns.light_palette("green", as_cmap=True)
styled_descriptive_stats = descriptive_stats.style.background_gradient(cmap=cm)

# Set the dashboard style
plt.style.use('dark_background')

# Create a figure for the histograms
fig, axs = plt.subplots(nrows=len(df.columns), ncols=1, figsize=(10, 5*len(df.columns)))

# Plot histograms
for i, col in enumerate(df.columns):
    sns.histplot(df[col], ax=axs[i], color="skyblue", bins=30, edgecolor='white')
    axs[i].set_title(f'Histogram for {col}', color='white')
    axs[i].grid(color='grey', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.suptitle('Descriptive Statistics Dashboard', color='white', size=30, y=1.02)

# Show the dashboard
plt.show()

# Display the styled descriptive statistics table
display(styled_descriptive_stats)


# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# Load the dataset
df = pd.read_csv(r"C:\Users\Sreelakshmi V\OneDrive\Desktop\ibm data set with feature elimination.csv")

# Calculate descriptive statistics including mean, median, mode, standard deviation, and quartiles
descriptive_stats = df.describe()
descriptive_stats.loc['mode'] = df.mode().iloc[0]
descriptive_stats.loc['variance'] = df.var(numeric_only=True)

# Rounding to two decimal points
descriptive_stats = descriptive_stats.round(2)

# Apply color grading to the descriptive statistics table
cm = sns.light_palette("green", as_cmap=True)
styled_descriptive_stats = descriptive_stats.style.background_gradient(cmap=cm)

# Display the styled descriptive statistics table in Jupyter Notebook
display(HTML(styled_descriptive_stats.to_html()))

# Set the dashboard style with black background
plt.style.use('dark_background')

# Define the number of rows and columns for the subplot grid
num_cols = 3
num_rows = len(df.select_dtypes(include='number').columns) // num_cols + 1

# Create a figure for the histograms
fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 6, num_rows * 4))
axs = axs.flatten()  # Flatten the array of axes for easy iteration

# Plot histograms for each numerical column
for i, col in enumerate(df.select_dtypes(include='number').columns):
    sns.histplot(df[col], ax=axs[i], color="skyblue", bins=30, edgecolor='white')
    axs[i].set_title(f'Histogram for {col}', color='white')
    axs[i].grid(True, linestyle='--', linewidth=0.5)
    axs[i].set_facecolor("#121212")  # Set histogram background color

# Remove extra subplots if not a perfect grid
for ax in axs[len(df.select_dtypes(include='number').columns):]:
    ax.remove()

# Adjust layout and add a main title
plt.tight_layout()
plt.suptitle('Employee Data Dashboard', color='white', size=20, y=1.02)

# Show the dashboard
plt.show()

