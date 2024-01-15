#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt

# Load the data from the provided CSV file
file_path = 'C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Files\\updated_normalized_training_data.csv'
data = pd.read_csv(file_path)

# Remove highly correlated features
corr_matrix = data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
data = data.drop(columns=to_drop)

# Split the data into training and testing sets
X = data.drop('Attrition', axis=1)
y = data['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions and probabilities
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_prob_train = model.predict_proba(X_train)[:, 1]
y_prob_test = model.predict_proba(X_test)[:, 1]

# Precision score and AUC
precision_train = precision_score(y_train, y_pred_train)
precision_test = precision_score(y_test, y_pred_test)
auc_train = roc_auc_score(y_train, y_prob_train)
auc_test = roc_auc_score(y_test, y_prob_test)

# Classification report
report_train = classification_report(y_train, y_pred_train)
report_test = classification_report(y_test, y_pred_test)

# ROC curve
fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)

plt.figure(figsize=(10, 6))
plt.plot(fpr_train, tpr_train, label=f'Train AUC: {auc_train:.2f}')
plt.plot(fpr_test, tpr_test, label=f'Test AUC: {auc_test:.2f}')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

print("Training Data Metrics:")
print("Precision Score:", precision_train)
print("AUC Score:", auc_train)
print("Classification Report:\n", report_train)

print("\nTest Data Metrics:")
print("Precision Score:", precision_test)
print("AUC Score:", auc_test)
print("Classification Report:\n", report_test)


# In[4]:


# Get feature importances from the trained logistic regression model
feature_importances = abs(model.coef_[0])

# Create a DataFrame to store feature names and their importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the top N most important features
top_n = 10  # You can change this value to see the top N most important features
top_features = feature_importance_df.head(top_n)

# Plot the top N most important features
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title(f'Top {top_n} Features Impacting Attrition')
plt.show()

# Print the top N most important features
print(f'Top {top_n} Features Impacting Attrition:')
print(top_features)


# In[9]:


import pandas as pd
from sklearn.metrics import classification_report

# Create classification reports for training and test data
report_train = classification_report(y_train, y_pred_train, output_dict=True)
report_test = classification_report(y_test, y_pred_test, output_dict=True)

# Extract the required metrics from the classification reports
precision_train = report_train['weighted avg']['precision']
precision_test = report_test['weighted avg']['precision']
macro_avg_train = report_train['macro avg']['f1-score']
macro_avg_test = report_test['macro avg']['f1-score']
weighted_avg_train = report_train['weighted avg']['f1-score']
weighted_avg_test = report_test['weighted avg']['f1-score']

# Create a dictionary to store the metrics
metrics_dict = {
    'Metric': ['Precision', 'Macro Avg F1-Score', 'Weighted Avg F1-Score'],
    'Training Data': [precision_train, macro_avg_train, weighted_avg_train],
    'Test Data': [precision_test, macro_avg_test, weighted_avg_test]
}

# Convert the dictionary to a pandas DataFrame
metrics_df = pd.DataFrame(metrics_dict)

# Display the metrics table
print(metrics_df)

