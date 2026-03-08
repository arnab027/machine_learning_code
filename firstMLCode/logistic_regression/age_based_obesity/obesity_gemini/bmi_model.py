import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

class RobustBMILogistic:
    def __init__(self):
        """
        Initialize RobustBMILogistic model.
        
        Attributes:
            model (statsmodels.Logit): Trained logistic regression model.
            scaler (StandardScaler): Feature scaler for normalization.
        """
        self.model = None
        self.scaler = StandardScaler()

    def _safe_sigmoid(self, z):
        """
        Numerically stable sigmoid to prevent 'Overflow in exp'.
        Handles large positive and negative z separately.
        """
        z = np.clip(z, -500, 500) # Safety cap
        return np.where(z >= 0, 
                        1 / (1 + np.exp(-z)), 
                        np.exp(z) / (1 + np.exp(z)))

    def train(self, bmis, labels):
        """Trains using Maximum Likelihood with Scaling and Regularization."""
        # 1. Scaling: Centers BMI around 0.0 with a std dev of 1.0
        X_raw = np.array(bmis).reshape(-1, 1)
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # 2. Add Intercept (beta_0)
        X_final = sm.add_constant(X_scaled)
        
        # 3. Fit with Regularization (L2) to prevent infinite weights
        # alpha=0.01 provides a 'leash' on weights without biasing the 0.5 threshold
        logit_model = sm.Logit(labels, X_final)
        self.model = logit_model.fit_regularized(method='l1', alpha=0.01, disp=False)
        print("Model trained: BMI features standardized and weights optimized.")

    def predict(self, raw_bmi):
        """Production inference: Scales input before calculating probability."""
        if self.model is None:
            raise ValueError("Model is not trained.")

        # IMPORTANT: Use .transform(), not .fit_transform()
        scaled_bmi = self.scaler.transform([[raw_bmi]])
        
        # Calculate Log-Odds (z) = b0 + b1 * scaled_bmi
        # params[0] is Intercept, params[1] is BMI weight
        z = self.model.params[0] + self.model.params[1] * scaled_bmi[0][0]
        
        prob = self._safe_sigmoid(z)
        return float(prob)

    def get_summary(self):
        return self.model.params