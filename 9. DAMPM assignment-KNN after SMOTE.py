#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install imbalanced-learn


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np

# Paths
data_file_path = "C:/Users/Sreelakshmi V/OneDrive/Documents/Sree/notes/predictive modeling/1. assesment/Assignment 2/Python Files/updated_normalized_training_data.csv"
correlation_matrix_path = "C:/Users/Sreelakshmi V/OneDrive/Documents/Sree/notes/predictive modeling/1. assesment/Assignment 2/Python Files/correlation_matrix.csv"
images_save_path = "C:/Users/Sreelakshmi V/OneDrive/Documents/Sree/notes/predictive modeling/1. assesment/Assignment 2/Python Images/"

# Read the data
data = pd.read_csv(data_file_path)
correlation_matrix = pd.read_csv(correlation_matrix_path, index_col=0)

# Remove features with correlation of 1 or approximately 1
highly_correlated_features = set()
for feature1 in correlation_matrix.columns:
    for feature2 in correlation_matrix.columns:
        if feature1 != feature2 and np.isclose(correlation_matrix.at[feature1, feature2], 1.0, atol=1e-2):
            highly_correlated_features.add(feature1)
            highly_correlated_features.add(feature2)

data = data.drop(columns=list(highly_correlated_features))

# Define features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to generate synthetic samples for the minority class in the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Find the best k using cross-validation on the resampled dataset
cv_scores = []
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_smote, y_train_smote, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = np.argmax(cv_scores) + 1
print(f"Best number of neighbors found: {best_k}")

# Train KNN with the best k on the resampled dataset
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_smote, y_train_smote)
y_pred = knn.predict(X_test)

# Evaluate the KNN model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Plot accuracy vs. k
plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), cv_scores, marker='o')
plt.title('Accuracy vs. Number of Neighbors')
plt.xlabel('Number of Neighbors')
plt.ylabel('Cross-validated accuracy')
plt.tight_layout()

# Save the plot
knn_plot_file = images_save_path + "knn_accuracy_plot_smote.png"
plt.savefig(knn_plot_file)
print(f"The accuracy vs. number of neighbors plot has been saved to {knn_plot_file}")


# In[15]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths to the data, correlation matrix, and directory for saving plots
training_data_path = "C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Files\\updated_normalized_training_data.csv"
correlation_matrix_path = "C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Files\\correlation_matrix.csv"
save_dir = "C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Images\\KNN images"

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Load the data
training_data = pd.read_csv(training_data_path)

# Load the correlation matrix
correlation_matrix = pd.read_csv(correlation_matrix_path, index_col=0)

# Remove highly correlated features
high_corr_var = np.where(correlation_matrix > 0.99)
high_corr_var = [(correlation_matrix.columns[x], correlation_matrix.columns[y]) for x, y in zip(*high_corr_var) if x != y and x < y]
for var_pair in high_corr_var:
    if var_pair[1] in training_data.columns:
        training_data.drop(var_pair[1], axis=1, inplace=True)

# Split the data into features and target
X = training_data.drop('Attrition', axis=1)
y = training_data['Attrition']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to calculate metrics and plot graphs for KNN
def evaluate_and_save_plots(k, X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)

    # Metrics
    auc_train = roc_auc_score(y_train, y_pred_train)
    auc_test = roc_auc_score(y_test, y_pred_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f'Confusion Matrix for K={k}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(save_dir, f'Confusion_Matrix_K{k}.png'))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_test)
    plt.figure()
    plt.plot(fpr, tpr, label=f'K={k} (AUC = {auc_test:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, f'ROC_Curve_K{k}.png'))
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_test)
    plt.figure()
    plt.plot(recall, precision, label=f'K={k}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(save_dir, f'Precision_Recall_Curve_K{k}.png'))
    plt.close()

    return auc_train, auc_test, accuracy_train, accuracy_test

# Evaluate KNN for a range of K values and save results in a table
results = []
for k in range(1, 6):  # You can choose the range of K values
    auc_train, auc_test, accuracy_train, accuracy_test = evaluate_and_save_plots(k, X_train, y_train, X_test, y_test)
    results.append([k, auc_train, auc_test, accuracy_train, accuracy_test])

# Save the results table as a CSV
results_df = pd.DataFrame(results, columns=['K', 'AUC Train', 'AUC Test', 'Accuracy Train', 'Accuracy Test'])
results_df.to_csv(os.path.join(save_dir, 'KNN_Scores.csv'), index=False)


# In[19]:


pip install eli5


# In[21]:


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# Load the training data
training_data_path = "C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Files\\updated_normalized_training_data.csv"
training_data = pd.read_csv(training_data_path)

# Split the data into features and target variable
X = training_data.drop('Attrition', axis=1)
y = training_data['Attrition']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN classifier with the optimal K value
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Compute permutation importances
results = permutation_importance(knn, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# Get the importance scores and their standard deviations
importances = results.importances_mean
std_devs = results.importances_std

# Organize the feature importances into a DataFrame
feature_importances_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances,
    'std_dev': std_devs
}).sort_values(by='importance', ascending=False)

# Define a path to save the feature importances table
feature_importances_path = "C:\\Users\\Sreelakshmi V\\OneDrive\\Documents\\Sree\\notes\\predictive modeling\\1. assesment\\Assignment 2\\Python Files\\feature_importances.csv"

# Save the feature importances to a CSV file
feature_importances_df.to_csv(feature_importances_path, index=False)

# Display the feature importances table
print(feature_importances_df)

