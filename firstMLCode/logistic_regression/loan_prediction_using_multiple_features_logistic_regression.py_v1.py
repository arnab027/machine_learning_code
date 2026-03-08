"""
Credit Risk Prediction using Logistic Regression

This module implements logistic regression for predicting loan defaults/approvals
using the credit_risk_dataset.csv with actual features like:
- Person Age
- Person Income
- Loan Intent
- Loan Grade
- Employment Length
- Credit History Length
- Previous Default Status

Features:
- Real data loading from CSV
- Feature engineering and preprocessing
- Automatic categorical encoding
- Model training with MLE (statsmodels)
- Prediction with probability estimation
- Model evaluation metrics (AUC, accuracy, confusion matrix)
- Model persistence (save/load)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import joblib
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CreditRiskPredictor:
    """
    Logistic Regression model for credit risk (loan default) prediction.
    
    Uses real credit risk data to predict probability of loan default.
    
    Features:
    - Automatic data loading and preprocessing
    - Categorical feature encoding
    - Feature normalization
    - MLE-based model training
    - Comprehensive evaluation metrics
    - Model persistence
    
    Attributes:
        model: Trained statsmodels Logit model
        scaler: StandardScaler for feature normalization
        label_encoders: Dict of LabelEncoders for categorical features
        feature_columns: List of features used in model
        dataset_path: Path to credit_risk_dataset.csv
        training_data_info: Metadata about training
    """
    
    def __init__(
        self,
        dataset_path: str = 'D://pythonMachineLearning//firstMLCode//logistic_regression//credit_risk_dataset.csv',
        random_seed: int = 42
    ):
        """
        Initialize CreditRiskPredictor.
        
        Parameters:
            dataset_path (str): Path to credit_risk_dataset.csv
            random_seed (int): Seed for reproducibility. Default: 42
        """
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.dataset_path = "D://pythonMachineLearning//firstMLCode//logistic_regression//credit_risk_dataset.csv"
        self.random_seed = random_seed
        self.training_data_info = None
        
        # Features to use for prediction
        self.numeric_features = [
            'person_age',
            'person_income',
            'person_emp_length',
            'loan_amnt',
            'loan_int_rate',
            'loan_percent_income',
            'cb_person_cred_hist_length'
        ]
        
        self.categorical_features = [
            'person_home_ownership',
            'loan_intent',
            'loan_grade',
            'cb_person_default_on_file'
        ]
        
        self.all_features = self.numeric_features + self.categorical_features
        self.target_col = 'loan_status'
        
        logger.info(f"CreditRiskPredictor initialized with seed={random_seed}")
    
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load credit risk dataset and preprocess it.
        
        Returns:
            pd.DataFrame: Preprocessed dataset ready for training
        
        Raises:
            FileNotFoundError: If dataset not found
            ValueError: If required columns missing
        """
        logger.info(f"Loading data from {self.dataset_path}")
        
        try:
            df = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        logger.info(f"Original shape: {df.shape}")
        
        # Validate required columns
        required_cols = self.all_features + [self.target_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Handle missing values
        initial_rows = len(df)
        df = df.dropna(subset=self.numeric_features)
        logger.info(f"Dropped {initial_rows - len(df)} rows with missing numeric values")
        
        # Fill categorical missing values with 'Unknown'
        for col in self.categorical_features:
            df[col] = df[col].fillna('Unknown')
        
        # Encode categorical variables
        for col in self.categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Ensure target is binary
        df[self.target_col] = df[self.target_col].astype(int)
        
        logger.info(f"Preprocessed shape: {df.shape}")
        logger.info(f"Target distribution:\n{df[self.target_col].value_counts()}")
        logger.info(f"Default rate: {(df[self.target_col].sum() / len(df) * 100):.2f}%")
        
        return df[[*self.all_features, self.target_col]]
    
    def train(
        self,
        test_size: float = 0.2,
        max_iter: int = 100
    ) -> Dict[str, Any]:
        """
        Train logistic regression model on credit risk data.
        
        Parameters:
            test_size (float): Proportion for test set. Default: 0.2
            max_iter (int): Maximum iterations. Default: 100
        
        Returns:
            dict: Training summary with metrics
        
        Raises:
            RuntimeError: If training fails
        """
        logger.info("Starting model training...")
        
        try:
            # Load and preprocess data
            df = self.load_and_preprocess_data()
            
            # Split into train/test
            np.random.seed(self.random_seed)
            mask = np.random.rand(len(df)) < (1 - test_size)
            df_train = df[mask]
            df_test = df[~mask]
            
            logger.info(f"Train set: {len(df_train)}, Test set: {len(df_test)}")
            
            # Prepare features and target
            X_train = df_train[self.all_features]
            y_train = df_train[self.target_col]
            self.X_test = df_test[self.all_features]
            self.y_test = df_test[self.target_col]
            
            # Normalize numeric features
            X_train_scaled = X_train.copy()
            X_train_scaled[self.numeric_features] = self.scaler.fit_transform(
                X_train[self.numeric_features]
            )
            
            # Add intercept
            X_train_scaled = sm.add_constant(X_train_scaled)
            
            # Train Logit model
            logit_model = sm.Logit(y_train, X_train_scaled)
            
            try:
                self.model = logit_model.fit(
                    method='bfgs',
                    maxiter=max_iter,
                    disp=False
                )
                logger.info("Successfully trained with BFGS optimization")
            except Exception as e:
                logger.warning(f"BFGS failed, trying Newton: {e}")
                self.model = logit_model.fit(
                    method='newton',
                    maxiter=max_iter,
                    disp=False
                )
                logger.info("Successfully trained with Newton method")
            
            # Evaluate on test set
            X_test_scaled = self.X_test.copy()
            X_test_scaled[self.numeric_features] = self.scaler.transform(
                self.X_test[self.numeric_features]
            )
            X_test_scaled = sm.add_constant(X_test_scaled)
            
            y_pred_proba = self.model.predict(X_test_scaled)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Calculate metrics
            accuracy = (y_pred == self.y_test).mean()
            auc = roc_auc_score(self.y_test, y_pred_proba)
            cm = confusion_matrix(self.y_test, y_pred)
            
            training_summary = {
                'n_train_samples': len(df_train),
                'n_test_samples': len(df_test),
                'timestamp': datetime.now().isoformat(),
                'seed': self.random_seed,
                'n_parameters': len(self.model.params),
                'converged': self.model.mle_retvals['converged'] == 1 if hasattr(self.model, 'mle_retvals') else None,
                'log_likelihood': self.model.llf,
                'aic': self.model.aic,
                'bic': self.model.bic,
                'pseudo_r2': self.model.prsquared,
                'test_accuracy': round(accuracy, 4),
                'test_auc': round(auc, 4),
                'confusion_matrix': cm.tolist()
            }
            
            self.training_data_info = training_summary
            logger.info(f"Training summary:\n{training_summary}")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Failed to train model: {e}")
    
    def predict(
    self,
    applicant_data: Dict[str, Any],
    threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Predict loan default risk for an applicant.
        
        Parameters:
            applicant_data (dict): Dictionary with applicant features
                Example:
                {
                    'person_age': 35,
                    'person_income': 85000,
                    'person_emp_length': 10,
                    'loan_amnt': 25000,
                    'loan_int_rate': 12.5,
                    'loan_percent_income': 0.29,
                    'cb_person_cred_hist_length': 5,
                    'person_home_ownership': 'MORTGAGE',
                    'loan_intent': 'PERSONAL',
                    'loan_grade': 'B',
                    'cb_person_default_on_file': 'N'
                }
            threshold (float): Default probability threshold. Default: 0.5
        
        Returns:
            dict: Prediction with default probability and risk status
        
        Raises:
            ValueError: If model not trained or invalid data
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if not (0 < threshold < 1):
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        
        try:
            # Create DataFrame with all required features
            input_df = pd.DataFrame([applicant_data])
            
            # Ensure all required features are present
            missing_features = [f for f in self.all_features if f not in input_df.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Encode categorical variables using trained encoders
            input_prepared = input_df.copy()
            for col in self.categorical_features:
                input_prepared[col] = self.label_encoders[col].transform(
                    input_df[col].astype(str)
                )
            
            # Scale numeric features using fitted scaler
            input_prepared[self.numeric_features] = self.scaler.transform(
                input_df[self.numeric_features]
            )
            
            # Select features in correct order and add intercept
            # IMPORTANT: Must match training order exactly
            X = input_prepared[self.all_features].values  # Convert to numpy array
            X = np.concatenate([[1], X[0]])  # Add constant (1) at the beginning
            X = X.reshape(1, -1)  # Reshape to (1, 12) for prediction
            
            logger.debug(f"Prediction input shape: {X.shape}")
            logger.debug(f"Model expects shape: (n_samples, {len(self.model.params)})")
            
            # Get prediction
            default_prob = float(self.model.predict(X)[0])
            default_prob = np.clip(default_prob, 1e-15, 1 - 1e-15)
            
            # Risk classification
            if default_prob >= threshold:
                risk_status = "HIGH RISK (LIKELY DEFAULT)"
            elif default_prob >= 0.3:
                risk_status = "MODERATE RISK"
            else:
                risk_status = "LOW RISK (LIKELY APPROVED)"
            
            result = {
                'default_probability': round(default_prob, 4),
                'default_percent': round(default_prob * 100, 2),
                'risk_status': risk_status,
                'recommendation': 'DENY' if default_prob >= threshold else 'APPROVE',
                'threshold': threshold,
                'confidence': abs(default_prob - 0.5) * 2
            }
            
            logger.debug(f"Prediction: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Failed to make prediction: {e}")
    
    def batch_predict(
        self,
        applicants_data: list,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Make predictions for multiple applicants.
        
        Parameters:
            applicants_data (list): List of applicant dictionaries
            threshold (float): Default threshold. Default: 0.5
        
        Returns:
            pd.DataFrame: Predictions for all applicants
        """
        logger.info(f"Making batch predictions for {len(applicants_data)} applicants")
        
        results = []
        for i, applicant in enumerate(applicants_data):
            try:
                result = self.predict(applicant, threshold)
                result['applicant_id'] = i
                results.append(result)
            except Exception as e:
                logger.warning(f"Prediction failed for applicant {i}: {e}")
                results.append({
                    'applicant_id': i,
                    'default_probability': None,
                    'risk_status': 'ERROR',
                    'recommendation': 'ERROR'
                })
        
        return pd.DataFrame(results)
    
    def get_model_summary(self) -> str:
        """
        Get detailed model summary.
        
        Returns:
            str: Formatted model summary
        """
        if self.model is None:
            return "Model not trained"
        
        return str(self.model.summary2())
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Parameters:
            filepath (str): Directory to save model files
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)
        
        try:
            joblib.dump(self.model, filepath / 'model.pkl')
            joblib.dump(self.scaler, filepath / 'scaler.pkl')
            joblib.dump(self.label_encoders, filepath / 'encoders.pkl')
            
            metadata = {
                'numeric_features': self.numeric_features,
                'categorical_features': self.categorical_features,
                'training_info': self.training_data_info
            }
            joblib.dump(metadata, filepath / 'metadata.pkl')
            
            logger.info(f"Model saved to {filepath}")
            print(f"✓ Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Save failed: {e}")
            raise IOError(f"Save failed: {e}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk.
        
        Parameters:
            filepath (str): Directory containing model files
        """
        filepath = Path(filepath)
        
        try:
            self.model = joblib.load(filepath / 'model.pkl')
            self.scaler = joblib.load(filepath / 'scaler.pkl')
            self.label_encoders = joblib.load(filepath / 'encoders.pkl')
            metadata = joblib.load(filepath / 'metadata.pkl')
            
            self.numeric_features = metadata['numeric_features']
            self.categorical_features = metadata['categorical_features']
            self.training_data_info = metadata['training_info']
            
            logger.info(f"Model loaded from {filepath}")
            print(f"✓ Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Load failed: {e}")
            raise RuntimeError(f"Load failed: {e}")


# ============================================================================
# DEMONSTRATION & TESTING
# ============================================================================

def main():
    """
    Main execution: Train model and demonstrate predictions.
    """
    print("=" * 80)
    print("CREDIT RISK PREDICTION - LOGISTIC REGRESSION")
    print("=" * 80)
    print()
    
    # Initialize and train
    predictor = CreditRiskPredictor(
        dataset_path='credit_risk_dataset.csv',
        random_seed=42
    )
    
    print("Training model on credit risk dataset...")
    training_result = predictor.train(test_size=0.2)
    print(f"\nTraining completed successfully!")
    print(f"  Train samples: {training_result['n_train_samples']}")
    print(f"  Test samples: {training_result['n_test_samples']}")
    print(f"  Test Accuracy: {training_result['test_accuracy']}")
    print(f"  Test AUC: {training_result['test_auc']}")
    print(f"  Parameters: {training_result['n_parameters']}")
    print()
    
    # Single predictions
    print("-" * 80)
    print("LOAN DEFAULT RISK PREDICTIONS")
    print("-" * 80)
    
    test_applicants = [
        {
            'name': 'Low Risk (Young Professional)',
            'data': {
                'person_age': 28,
                'person_income': 95000,
                'person_emp_length': 5,
                'loan_amnt': 20000,
                'loan_int_rate': 8.5,
                'loan_percent_income': 0.21,
                'cb_person_cred_hist_length': 6,
                'person_home_ownership': 'MORTGAGE',
                'loan_intent': 'PERSONAL',
                'loan_grade': 'A',
                'cb_person_default_on_file': 'N'
            }
        },
        {
            'name': 'High Risk (Poor History)',
            'data': {
                'person_age': 45,
                'person_income': 35000,
                'person_emp_length': 2,
                'loan_amnt': 25000,
                'loan_int_rate': 18.5,
                'loan_percent_income': 0.71,
                'cb_person_cred_hist_length': 2,
                'person_home_ownership': 'RENT',
                'loan_intent': 'PERSONAL',
                'loan_grade': 'D',
                'cb_person_default_on_file': 'Y'
            }
        },
        {
            'name': 'Moderate Risk (Mid-level)',
            'data': {
                'person_age': 35,
                'person_income': 65000,
                'person_emp_length': 8,
                'loan_amnt': 15000,
                'loan_int_rate': 11.2,
                'loan_percent_income': 0.23,
                'cb_person_cred_hist_length': 4,
                'person_home_ownership': 'RENT',
                'loan_intent': 'EDUCATION',
                'loan_grade': 'B',
                'cb_person_default_on_file': 'N'
            }
        }
    ]
    
    for applicant in test_applicants:
        print(f"\n{applicant['name']}:")
        result = predictor.predict(applicant['data'])
        print(f"  Default Probability: {result['default_percent']}%")
        print(f"  Risk Status: {result['risk_status']}")
        print(f"  Recommendation: {result['recommendation']}")
        print(f"  Confidence: {round(result['confidence'], 2)}")
    
    # Model summary
    print("\n" + "-" * 80)
    print("MODEL SUMMARY")
    print("-" * 80)
    print(predictor.get_model_summary())
    
    # Save model
    print("\n" + "-" * 80)
    print("SAVING MODEL")
    print("-" * 80)
    predictor.save_model("./credit_risk_model")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()