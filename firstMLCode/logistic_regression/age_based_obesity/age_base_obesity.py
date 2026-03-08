# Logistic Regression on Synthetic Obesity Data with Visualizations and Excel Output

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    roc_curve, auc
)
from matplotlib.colors import ListedColormap
import seaborn as sns

# Setting the random seed for reproducibility
np.random.seed(0)

# Number of samples
num_samples = 500

# Generating User IDs
user_ids = np.arange(1, num_samples + 1)

# Generating ages between 18 and 80
ages = np.random.randint(18, 81, size=num_samples)

# Generating obesity status based on age
obesity_prob_by_age = (ages - 18) / (80 - 18)  # Probability increases with age
obesity_status = np.random.binomial(1, obesity_prob_by_age * 0.6)  # Adjusted probability

# Generating weights based on obesity status
weights = []
for age, obese in zip(ages, obesity_status):
    if obese:
        # Obese individuals
        mean_weight = 80 + (age - 18) * 0.5  # Mean weight increases with age
        weight = np.random.normal(mean_weight, 10)
    else:
        # Non-obese individuals
        mean_weight = 60 + (age - 18) * 0.3
        weight = np.random.normal(mean_weight, 8)
    weights.append(weight)

weights = np.array(weights)

# Creating the DataFrame
df = pd.DataFrame({
    'UserID': user_ids,
    'Age': ages,
    'Weight': weights,
    'Obese': obesity_status
})

# Saving the dataset to an Excel file
df.to_excel('obesity_data.xlsx', index=False)

print("Dataset saved to 'obesity_data.xlsx'.")

# --------------------------------------------
# Reading the data and performing Logistic Regression
# --------------------------------------------

# Reading the data from the Excel file
df = pd.read_excel('obesity_data.xlsx')

# Features and target variable
X = df[['UserID', 'Age', 'Weight']]
y = df['Obese']

# Splitting the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Extract UserID from X_train and X_test
X_train_userid = X_train['UserID'].values
X_test_userid = X_test['UserID'].values

# Extract Age and Weight for scaling
X_train_features = X_train[['Age', 'Weight']]
X_test_features = X_test[['Age', 'Weight']]

# Feature Scaling
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train_features)
X_test_scaled = sc.transform(X_test_features)

# Training the Logistic Regression Model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train_scaled, y_train)

# Making Predictions and Evaluating the Model
y_pred = classifier.predict(X_test_scaled)
y_prob = classifier.predict_proba(X_test_scaled)[:, 1]

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
print(f"\nAccuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------
# Saving Outputs to Excel
# --------------------------------------------

# Creating a DataFrame with the test set results
results_df = pd.DataFrame({
    'UserID': X_test_userid,
    'Age': X_test_features['Age'].values,
    'Weight': X_test_features['Weight'].values,
    'Actual Obese': y_test.values,
    'Predicted Obese': y_pred,
    'Predicted Probability': y_prob
})

# Optionally, sort the results by UserID
results_df = results_df.sort_values(by='UserID')

# Save results to an Excel file
with pd.ExcelWriter('obesity_model_outputs.xlsx') as writer:
    # Write the results DataFrame to a sheet
    results_df.to_excel(writer, sheet_name='Test Set Predictions', index=False)

    # Save the confusion matrix as a DataFrame
    cm_df = pd.DataFrame(cm, index=['Actual Not Obese', 'Actual Obese'],
                         columns=['Predicted Not Obese', 'Predicted Obese'])
    cm_df.to_excel(writer, sheet_name='Confusion Matrix')

    # Save the classification report as a DataFrame
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_excel(writer, sheet_name='Classification Report')

print("\nOutputs have been saved to 'obesity_model_outputs.xlsx'.")

# --------------------------------------------
# Visualizations (Optional)
# --------------------------------------------

# 1. Decision Boundary Plot for Training Set
X_set, y_set = X_train_scaled, y_train

# Create meshgrid
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 0.5, stop=X_set[:, 0].max() + 0.5, step=0.01),
    np.arange(start=X_set[:, 1].min() - 0.5, stop=X_set[:, 1].max() + 0.5, step=0.01)
)

plt.figure(figsize=(12, 6))
plt.contourf(
    X1, X2,
    classifier.predict(
        np.array([X1.ravel(), X2.ravel()]).T
    ).reshape(X1.shape),
    alpha=0.3, cmap=ListedColormap(('lightblue', 'lightcoral'))
)
plt.scatter(X_set[y_set == 0, 0], X_set[y_set == 0, 1],
            c='blue', label='Not Obese', edgecolor='k', s=20)
plt.scatter(X_set[y_set == 1, 0], X_set[y_set == 1, 1],
            c='red', label='Obese', edgecolor='k', s=20)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age (Standardized)')
plt.ylabel('Weight (Standardized)')
plt.legend()
plt.show()

# 2. Decision Boundary Plot for Test Set
X_set, y_set = X_test_scaled, y_test

# Create meshgrid
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 0.5, stop=X_set[:, 0].max() + 0.5, step=0.01),
    np.arange(start=X_set[:, 1].min() - 0.5, stop=X_set[:, 1].max() + 0.5, step=0.01)
)

plt.figure(figsize=(12, 6))
plt.contourf(
    X1, X2,
    classifier.predict(
        np.array([X1.ravel(), X2.ravel()]).T
    ).reshape(X1.shape),
    alpha=0.3, cmap=ListedColormap(('lightblue', 'lightcoral'))
)
plt.scatter(X_set[y_set == 0, 0], X_set[y_set == 0, 1],
            c='blue', label='Not Obese', edgecolor='k', s=20)
plt.scatter(X_set[y_set == 1, 0], X_set[y_set == 1, 1],
            c='red', label='Obese', edgecolor='k', s=20)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age (Standardized)')
plt.ylabel('Weight (Standardized)')
plt.legend()
plt.show()

# 3. Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 4. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()