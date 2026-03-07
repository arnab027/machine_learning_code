"""
Logistic Regression using Maximum Likelihood Estimation (MLE).

This module implements logistic regression for binary classification using gradient
ascent optimization to maximize the likelihood function. It's designed for solving
binary classification problems such as loan approval prediction based on credit scores.

Improvements:
- Feature normalization (standardization) for better convergence
- Optimized learning rate and increased epochs
- Convergence monitoring to detect training completion

Example:
    >>> from logistirc_reg import LogisticRegressionMLE
    >>> X = np.array([600, 700, 750])
    >>> y = np.array([0, 1, 1])
    >>> model = LogisticRegressionMLE(learning_rate=0.1, epochs=5000)
    >>> model.fit(X, y)
    >>> predictions = model.predict(np.array([650, 725, 780]))
"""

import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_LEARNING_RATE = 0.1  # Increased from 0.001
DEFAULT_EPOCHS = 5000  # Increased from 1000
PRINT_INTERVAL = 500
EPSILON = 1e-10  # Prevents log(0) in likelihood calculation
CONVERGENCE_THRESHOLD = 1e-6  # For early stopping


class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance.
    
    Uses Z-score normalization: (X - mean) / std
    """
    
    def __init__(self) -> None:
        """Initialize the scaler."""
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
    
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Fit the scaler on training data.
        
        Parameters:
            X (np.ndarray): Input features
        
        Returns:
            StandardScaler: Fitted scaler (self)
        """
        self.mean = np.mean(X)
        self.std = np.std(X) #calculates and stores the standard deviation of the input features X.
        if self.std == 0:
            self.std = 1  # Prevent division by zero
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted mean and std.
        
        Parameters:
            X (np.ndarray): Input features
        
        Returns:
            np.ndarray: Normalized features
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before transforming")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Parameters:
            X (np.ndarray): Input features
        
        Returns:
            np.ndarray: Normalized features
        """
        return self.fit(X).transform(X)


class LogisticRegressionMLE:
    """
    Logistic Regression classifier using Maximum Likelihood Estimation (MLE).
    
    This class implements binary logistic regression using gradient ascent to maximize
    the likelihood function. The model learns optimal intercept (b0) and weight (b1)
    parameters for binary classification.
    
    Features:
    - Feature normalization for improved convergence
    - Convergence monitoring with early stopping
    - Numerical stability with clipping
    
    Mathematical Model:
        - Linear combination: z = b0 + b1 * X
        - Prediction: ŷ = sigmoid(z) = 1 / (1 + e^(-z))
        - Likelihood: L = ∏(ŷ^y * (1-ŷ)^(1-y))
    
    Attributes:
        learning_rate (float): Learning rate for gradient ascent (alpha).
        epochs (int): Number of training iterations.
        intercept (float): Learned intercept parameter (b0).
        weight (float): Learned weight parameter (b1).
        likelihood_history (list): Likelihood values during training.
        scaler (StandardScaler): Feature scaler for normalization.
    """
    
    def __init__(
        self,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        epochs: int = DEFAULT_EPOCHS
    ) -> None:
        """
        Initialize the LogisticRegressionMLE model.
        
        Parameters:
            learning_rate (float): Step size for gradient ascent. Default: 0.1.
                Must be positive. Typical range: 0.01 to 0.5 for normalized features.
            epochs (int): Number of training iterations. Default: 5000.
                More epochs allow better convergence but increase training time.
        
        Raises:
            ValueError: If learning_rate <= 0 or epochs <= 0.
        """
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.intercept: Optional[float] = None
        self.weight: Optional[float] = None
        self.likelihood_history: list = []
        self.scaler = StandardScaler()
        self.converged_epoch: Optional[int] = None
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        
        Maps any real-valued input to a probability between 0 and 1.
        Formula: σ(z) = 1 / (1 + e^(-z))
        
        Parameters:
            z (np.ndarray): Input values (can be scalar or array).
        
        Returns:
            np.ndarray: Sigmoid output in range [0, 1].
        """
        # Clip z to prevent overflow in exp calculation
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))
    
    def _calculate_likelihood(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate the likelihood of the predictions.
        
        Likelihood = ∏(ŷ^y * (1-ŷ)^(1-y))
        
        Parameters:
            y_true (np.ndarray): True binary labels (0 or 1).
            y_pred (np.ndarray): Predicted probabilities [0, 1].
        
        Returns:
            float: Likelihood value. Higher is better.
        """
        # Add epsilon to prevent log(0)
        y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
        likelihood = np.prod(y_pred**y_true * (1 - y_pred)**(1 - y_true))
        return float(likelihood)
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'LogisticRegressionMLE':
        """
        Train the logistic regression model using gradient ascent.
        
        Features:
        - Normalizes features using Z-score standardization
        - Uses gradient ascent to maximize likelihood
        - Monitors convergence and stops early if no improvement
        
        Parameters:
            X (np.ndarray): Feature values (1D array of shape (n_samples,)).
            y (np.ndarray): Binary labels (1D array of shape (n_samples,), values 0 or 1).
            verbose (bool): If True, print training progress. Default: True.
        
        Returns:
            LogisticRegressionMLE: Fitted model (self).
        
        Raises:
            ValueError: If X and y have different lengths or if y contains non-binary values.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim != 1 or y.ndim != 1:
            raise ValueError("X and y must be 1D arrays")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length, got {len(X)} and {len(y)}")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only binary values (0 or 1)")
        
        logger.info(f"Starting training with {len(X)} samples for {self.epochs} epochs")
        
        # Normalize features
        X_normalized = self.scaler.fit_transform(X)
        logger.info(
            f"Feature normalization: mean={self.scaler.mean:.2f}, "
            f"std={self.scaler.std:.2f}"
        )
        
        # Initialize parameters
        self.intercept = 0.0
        self.weight = 0.0
        self.likelihood_history = []
        self.converged_epoch = None
        
        if verbose:
            print(f"{'Epoch':<8} | {'b0':<12} | {'b1':<12} | {'Likelihood':<12} | {'Improvement':<12}")
            print("-" * 65)
        
        # Gradient ascent optimization
        for epoch in range(self.epochs):
            # Forward pass: compute predictions
            z = self.intercept + self.weight * X_normalized
            y_pred = self.sigmoid(z)
            
            # Gradient ascent: maximize likelihood
            error = y - y_pred
            grad_intercept = np.sum(error)
            grad_weight = np.sum(error * X_normalized)
            
            # Update parameters
            self.intercept += self.learning_rate * grad_intercept
            self.weight += self.learning_rate * grad_weight
            
            # Calculate and store likelihood
            likelihood = self._calculate_likelihood(y, y_pred)
            self.likelihood_history.append(likelihood)
            
            # Check for convergence
            if len(self.likelihood_history) > 1:
                improvement = self.likelihood_history[-1] - self.likelihood_history[-2]
                if improvement < CONVERGENCE_THRESHOLD and epoch > 100:
                    if self.converged_epoch is None:
                        self.converged_epoch = epoch
                        if verbose:
                            print(f"\nConverged at epoch {epoch} with improvement {improvement:.2e}")
                        break
            else:
                improvement = 0.0
            
            if verbose and epoch % PRINT_INTERVAL == 0:
                print(
                    f"{epoch:<8} | {self.intercept:<12.6f} | "
                    f"{self.weight:<12.6f} | {likelihood:<12.6f} | {improvement:<12.2e}"
                )
        
        logger.info(
            f"Training completed. Final weights - Intercept: {self.intercept:.6f}, "
            f"Weight: {self.weight:.6f}"
        )
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of positive class.
        
        Parameters:
            X (np.ndarray): Feature values (1D array or scalar).
        
        Returns:
            np.ndarray: Predicted probabilities in range [0, 1].
        
        Raises:
            RuntimeError: If model has not been fitted yet.
        """
        if self.intercept is None or self.weight is None:
            raise RuntimeError("Model must be fitted before making predictions")
        
        X = np.asarray(X, dtype=np.float64)
        X_normalized = self.scaler.transform(X)
        z = self.intercept + self.weight * X_normalized
        return self.sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels.
        
        Parameters:
            X (np.ndarray): Feature values (1D array or scalar).
            threshold (float): Classification threshold. Default: 0.5.
                Predictions >= threshold are classified as 1, else 0.
        
        Returns:
            np.ndarray: Predicted binary labels (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def get_params(self) -> dict:
        """
        Get learned model parameters.
        
        Returns:
            dict: Dictionary containing intercept, weight, and scaler statistics.
        """
        return {
            'intercept': self.intercept,
            'weight': self.weight,
            'feature_mean': self.scaler.mean,
            'feature_std': self.scaler.std
        }


def main() -> None:
    """
    Main function demonstrating logistic regression for loan approval prediction.
    
    Uses credit score as feature to predict loan approval (binary classification).
    """
    # Load and prepare data
    data = {
        'credit_score': [600, 700, 750],
        'loan_granted': [0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    X = df['credit_score'].values
    y = df['loan_granted'].values
    
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Feature range: [{X.min()}, {X.max()}]")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Train model with improved hyperparameters
    model = LogisticRegressionMLE(learning_rate=0.1, epochs=5000)
    model.fit(X, y, verbose=True)
    
    # Display results
    params = model.get_params()
    print(f"\n{'='*60}")
    print(f"Final MLE Model Parameters:")
    print(f"  Intercept (b0):      {params['intercept']:.6f}")
    print(f"  Weight (b1):         {params['weight']:.6f}")
    print(f"  Feature Mean:        {params['feature_mean']:.2f}")
    print(f"  Feature Std:         {params['feature_std']:.2f}")
    if model.converged_epoch:
        print(f"  Converged at epoch:  {model.converged_epoch}")
    print(f"{'='*60}\n")
    
    # Make predictions on new data
    test_scores = np.array([620, 700, 780])
    predictions_prob = model.predict_proba(test_scores)
    predictions_class = model.predict(test_scores)
    
    print(f"{'Credit Score':<15} | {'Probability':<15} | {'Prediction':<15}")
    print("-" * 50)
    for score, prob, pred in zip(test_scores, predictions_prob, predictions_class):
        prediction_text = "APPROVED" if pred == 1 else "NOT APPROVED"
        print(f"{score:<15} | {prob:<15.4f} | {prediction_text:<15}")


if __name__ == "__main__":
    main()