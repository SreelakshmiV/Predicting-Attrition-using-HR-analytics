#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install xgboost


# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Files\\updated_normalized_training_data.csv'
data = pd.read_csv(file_path)

# Assuming 'Attrition' is the target variable
target = 'Attrition'

# Separating the target variable from the features
X = data.drop(target, axis=1)
y = data[target]

# Removing highly correlating variables
correlation_matrix = X.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.99)]
X_reduced = X.drop(to_drop, axis=1)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Creating the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Training the model
model.fit(X_train, y_train)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred, y_pred_proba):
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    conf_matrix = confusion_matrix(y_true, y_pred)
    precision = round(precision_score(y_true, y_pred), 4)
    recall = round(recall_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred), 4)
    roc_auc = round(roc_auc_score(y_true, y_pred_proba), 4)
    return accuracy, conf_matrix, precision, recall, f1, roc_auc

# Evaluating the model on the training set
y_train_pred = model.predict(X_train)
y_train_pred_proba = model.predict_proba(X_train)[:,1]
train_metrics = calculate_metrics(y_train, y_train_pred, y_train_pred_proba)

# Evaluating the model on the testing set
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)[:,1]
test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)

# Displaying the metrics for training and testing sets
print(f'Training Metrics:\nAccuracy: {train_metrics[0]}, Confusion Matrix:\n{train_metrics[1]}, Precision: {train_metrics[2]}, Recall: {train_metrics[3]}, F1 Score: {train_metrics[4]}, ROC AUC Score: {train_metrics[5]}\n')
print(f'Testing Metrics:\nAccuracy: {test_metrics[0]}, Confusion Matrix:\n{test_metrics[1]}, Precision: {test_metrics[2]}, Recall: {test_metrics[3]}, F1 Score: {test_metrics[4]}, ROC AUC Score: {test_metrics[5]}')

# Plotting the ROC curve for the testing set
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label='XGBoost Test Set (area = %0.4f)' % test_metrics[5])
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Test Set')
plt.legend(loc="lower right")
plt.show()


# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Files\\updated_normalized_training_data.csv'
data = pd.read_csv(file_path)

# Assuming 'Attrition' is the target variable
target = 'Attrition'

# Separating the target variable from the features
X = data.drop(target, axis=1)
y = data[target]

# Removing highly correlating variables
correlation_matrix = X.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.99)]
X_reduced = X.drop(to_drop, axis=1)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Creating the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Training the model
model.fit(X_train, y_train)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred, y_pred_proba):
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    conf_matrix = confusion_matrix(y_true, y_pred)
    precision = round(precision_score(y_true, y_pred), 4)
    recall = round(recall_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred), 4)
    roc_auc = round(roc_auc_score(y_true, y_pred_proba), 4)
    return accuracy, conf_matrix, precision, recall, f1, roc_auc

# Evaluating the model on the training set
y_train_pred = model.predict(X_train)
y_train_pred_proba = model.predict_proba(X_train)[:,1]
train_metrics = calculate_metrics(y_train, y_train_pred, y_train_pred_proba)

# Evaluating the model on the testing set
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)[:,1]
test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)

# Create a DataFrame to display the metrics in a table format
metrics_df = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
                           'Training Set': [train_metrics[0], train_metrics[2], train_metrics[3], train_metrics[4], train_metrics[5]],
                           'Testing Set': [test_metrics[0], test_metrics[2], test_metrics[3], test_metrics[4], test_metrics[5]]})

# Display the metrics DataFrame
print(metrics_df)

# Plotting the ROC curve for the testing set
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label='XGBoost Test Set (area = %0.4f)' % test_metrics[5])
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Test Set')
plt.legend(loc="lower right")
plt.show()


# In[6]:


# Fit the XGBoost model to get feature importance scores
model.fit(X, y)

# Get feature importance scores
feature_importance = model.feature_importances_

# Create a DataFrame to store feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Extract the top 10 important features
top_10_features = feature_importance_df.head(10)

# Display the top 10 features
print("Top 10 Features Impacting Attrition:")
print(top_10_features)

# Plot the top 10 features' importance
plt.figure(figsize=(10, 6))
plt.barh(top_10_features['Feature'], top_10_features['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 10 Features Impacting Attrition')
plt.gca().invert_yaxis()
plt.show()

