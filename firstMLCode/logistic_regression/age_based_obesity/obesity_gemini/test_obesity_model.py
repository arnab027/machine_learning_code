from bmi_model import RobustBMILogistic
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# 1. Create Synthetic Training Data
np.random.seed(42)
try:
    column_names = ['BMI', 'Is_Obese']
    df = pd.read_csv('D://pythonMachineLearning//firstMLCode//logistic_regression//age_based_obesity//obesity_gemini//obesity_geminihealth_data.csv',names=column_names, header=None)
    # Preprocessing: Remove any potential duplicates or empty rows
    df = df.dropna().drop_duplicates()
    print(df.head())
    print("CSV loaded successfully.")
except FileNotFoundError:
    print("Error: health_data.csv not found. Please ensure the file exists.")
    exit()


# 2. Initialize and Train
predictor = RobustBMILogistic()
X = df['BMI'].values
y = df['Is_Obese'].values

## --3
# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

predictor.train(X_train, y_train)


# --- 4. TEST THE MODEL (Evaluation) ---
# We calculate predictions for the entire test set
y_probs = [predictor.predict(bmi) for bmi in X_test]
y_pred = [1 if p >= 0.5 else 0 for p in y_probs]

accuracy = accuracy_score(y_test, y_pred)

print("\n--- MODEL TEST RESULTS ---")
print(f"Accuracy on Test Set: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))