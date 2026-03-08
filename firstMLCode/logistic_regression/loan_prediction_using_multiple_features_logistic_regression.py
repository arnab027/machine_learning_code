import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import joblib

class LoanPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = ['Age', 'Salary', 'Credit_Score']
        self.all_cols = ['Age', 'Sex', 'Salary', 'Credit_Score']

    def _generate_synthetic_data(self, n):
        """Generates training data with controlled logic to avoid perfect separation."""
        np.random.seed(42)
        data = {
            'Age': np.random.randint(22, 65, n),
            'Sex': np.random.randint(0, 2, n),
            'Salary': np.random.randint(30000, 150000, n),
            'Credit_Score': np.random.randint(300, 850, n)
        }
        # Simplified logic for better stability
        z = (0.02 * data['Credit_Score'] + 0.00005 * data['Salary'] - 25)
        # Adding random noise to prevent 'Perfect Separation' errors
        noise = np.random.normal(0, 1, n)
        prob = 1 / (1 + np.exp(-(z + noise)))
        data['Loan_Granted'] = (prob > 0.5).astype(int)
        return pd.DataFrame(data)

    def train(self, n_samples=1000):
        """Trains the model with stability fixes."""
        df = self._generate_synthetic_data(n_samples)
        
        # 1. Standardizing features is the #1 cure for OverflowErrors
        df_scaled = df.copy()
        df_scaled[self.features] = self.scaler.fit_transform(df[self.features])
        
        X = df_scaled[self.all_cols]
        X = sm.add_constant(X) 
        y = df_scaled['Loan_Granted']
        
        # 2. Using 'fit_regularized' adds stability (L1/L2 penalty)
        # This prevents weights from becoming massive and causing exp() overflow
        logit_model = sm.Logit(y, X)
        
        try:
            # We use 'bfgs' or 'newton' solvers which are more robust
            self.model = logit_model.fit(method='bfgs', maxiter=100, disp=False)
            print("Successfully trained MLE model with BFGS optimization.")
        except Exception as e:
            print(f"Standard MLE failed, falling back to regularized fit: {e}")
            self.model = logit_model.fit_regularized(method='l1', alpha=1.0, disp=False)

    def predict(self, applicant_data, threshold=0.5):
        """Production prediction engine with numerical clipping."""
        if self.model is None:
            raise ValueError("Model is not trained.")

        input_df = pd.DataFrame([applicant_data])
        
        # Consistent Scaling
        input_scaled = input_df.copy()
        input_scaled[self.features] = self.scaler.transform(input_df[self.features])
        
        # Add Constant and align columns
        input_scaled = sm.add_constant(input_scaled, has_constant='add')
        # Ensure 'const' is the first column as expected by statsmodels
        input_final = input_scaled[['const'] + self.all_cols]

        # Use predict and clip output to prevent float issues
        raw_prob = self.model.predict(input_final)[0]
        safe_prob = np.clip(raw_prob, 1e-15, 1 - 1e-15)
        
        return {
            "probability": round(float(safe_prob), 4),
            "decision": "GRANTED" if safe_prob >= threshold else "DENIED"
        }

# --- EXECUTION ---
predictor = LoanPredictor()
predictor.train(n_samples=1000)

# Test with a high-salary, high-score individual
sample = {'Age': 40, 'Sex': 1, 'Salary': 120000, 'Credit_Score': 800}
print(f"Result: {predictor.predict(sample)}")