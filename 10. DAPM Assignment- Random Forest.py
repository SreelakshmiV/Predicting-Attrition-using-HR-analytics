#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
correlation_matrix_path = 'C:/Users/Sreelakshmi V/OneDrive/Documents/Sree/notes/predictive modeling/1. assesment/Assignment 2/Python Files/correlation_matrix.csv'
training_data_path = 'C:/Users/Sreelakshmi V/OneDrive/Documents/Sree/notes/predictive modeling/1. assesment/Assignment 2/Python Files/updated_normalized_training_data.csv'

correlation_matrix = pd.read_csv(correlation_matrix_path)
training_data = pd.read_csv(training_data_path)

# Set a threshold for high correlation
correlation_threshold = 0.95

# Dropping the first column from the correlation matrix as it's the index
correlation_matrix.set_index('Unnamed: 0', inplace=True)

# Identifying highly correlated features
highly_correlated = set()
for col in correlation_matrix.columns:
    for row in correlation_matrix.index:
        if (abs(correlation_matrix.at[row, col]) > correlation_threshold) and (col != row):
            highly_correlated.add(row)
            highly_correlated.add(col)

# Removing the highly correlated features from the training data
reduced_training_data = training_data.drop(columns=list(highly_correlated))

# Preparing the data for training the Random Forest model
X = reduced_training_data.drop('Attrition', axis=1)  # Features
y = reduced_training_data['Attrition']  # Target variable

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)

# Evaluating the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Plotting feature importances with color coding
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
sorted_idx = np.argsort(feature_importances)
plt.figure(figsize=(12,8))
palette = sns.color_palette("viridis", n_colors=len(feature_importances))
sns.barplot(x=feature_importances[sorted_idx], y=feature_importances.index[sorted_idx], palette=np.array(palette[::-1]))
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features in Random Forest')

# Saving the plot
plot_save_path = 'C:/Users/Sreelakshmi V/OneDrive/Documents/Sree/notes/predictive modeling/1. assesment/Assignment 2/Python Images/random_forest_feature_importance.png'
plt.savefig(plot_save_path)


# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data_path = "C:/Users/Sreelakshmi V/OneDrive/Documents/Sree/notes/predictive modeling/1. assesment/Assignment 2/Python Files/updated_normalized_training_data.csv"
correlation_path = "C:/Users/Sreelakshmi V/OneDrive/Documents/Sree/notes/predictive modeling/1. assesment/Assignment 2/Python Files/correlation_matrix.csv"
data = pd.read_csv(data_path)
correlation_matrix = pd.read_csv(correlation_path, index_col=0)

# Function to remove highly correlated features
def remove_highly_correlated_features(df, corr_matrix, threshold=0.99):
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]
    return df.drop(columns=to_drop), to_drop

# Removing highly correlated features
data_cleaned, _ = remove_highly_correlated_features(data, correlation_matrix)

# Assuming 'Attrition' is the target variable
X = data_cleaned.drop('Attrition', axis=1)
y = data_cleaned['Attrition']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the random forest classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Making predictions for both train and test data
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# Calculating metrics for both train and test data
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
train_roc_auc = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])
test_roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
train_precision, train_recall, _ = precision_recall_curve(y_train, rf.predict_proba(X_train)[:, 1])
test_precision, test_recall, _ = precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1])
train_pr_auc = auc(train_recall, train_precision)
test_pr_auc = auc(test_recall, test_precision)

# Feature importance
feature_importances = rf.feature_importances_
features = X_train.columns
importances = pd.DataFrame({'feature': features, 'importance': feature_importances})
importances_sorted = importances.sort_values('importance', ascending=False)

# Save the directory path
save_dir = "C:/Users/Sreelakshmi V/OneDrive/Documents/Sree/notes/predictive modeling/1. assesment/Assignment 2/Python Images/Random forest/"

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=importances_sorted)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig(save_dir + 'feature_importances.png')

# Results including additional metrics
results = {
    "Train Accuracy": train_accuracy,
    "Test Accuracy": test_accuracy,
    "Test Precision": test_precision,
    "Test Recall": test_recall,
    "Test F1 Score": test_f1,
    "Train ROC AUC": train_roc_auc,
    "Test ROC AUC": test_roc_auc,
    "Train PR AUC": train_pr_auc,
    "Test PR AUC": test_pr_auc,
}

# Print out results or write them to a file as needed
print(results)

# Write the results to a file
results_path = save_dir + 'model_results.txt'
with open(results_path, 'w') as file:
    for key, value in results.items():
        file.write(f'{key}: {value}\n')

# Write the feature importances to a file
importances_path = save_dir + 'feature_importances.csv'
importances_sorted.to_csv(importances_path, index=False)

# Plot ROC curve for both train and test data
plt.figure(figsize=(12, 5))
train_fpr, train_tpr, _ = roc_curve(y_train, rf.predict_proba(X_train)[:, 1])
test_fpr, test_tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
plt.plot(train_fpr, train_tpr, label=f'Train ROC AUC: {train_roc_auc:.2f}')
plt.plot(test_fpr, test_tpr, label=f'Test ROC AUC: {test_roc_auc:.2f}')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig(save_dir + 'roc_curves.png')

# Plot Precision-Recall curve for both train and test data
plt.figure(figsize=(12, 5))
plt.plot(train_recall, train_precision, label=f'Train PR AUC: {train_pr_auc:.2f}')
plt.plot(test_recall, test_precision, label=f'Test PR AUC: {test_pr_auc:.2f}')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.tight_layout()
plt.savefig(save_dir + 'precision_recall_curves.png')


# In[3]:


from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix for the test data
confusion_matrix_result = confusion_matrix(y_test, y_test_pred)

# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, fmt="d", cmap="Blues", linewidths=0.5, annot_kws={"fontsize": 16})
plt.title('Confusion Matrix (Test Data)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig(save_dir + 'confusion_matrix.png')
plt.show()

# Print the confusion matrix
print("Confusion Matrix (Test Data):")
print(confusion_matrix_result)


# In[4]:


from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix for the training data
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)

# Display the confusion matrix for training data
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_train, annot=True, fmt="d", cmap="Blues", linewidths=0.5, annot_kws={"fontsize": 16})
plt.title('Confusion Matrix (Training Data)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig(save_dir + 'confusion_matrix_train.png')
plt.show()

# Print the confusion matrix for training data
print("Confusion Matrix (Training Data):")
print(confusion_matrix_train)

