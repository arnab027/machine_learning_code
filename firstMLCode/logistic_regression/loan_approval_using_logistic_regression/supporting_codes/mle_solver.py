import numpy as np
import pandas as pd

# Dataset
data = {'credit_score': [600, 700, 750], 'loan_granted': [0, 1, 1]}
df = pd.DataFrame(data)

def sigmoid(z):
    """
    The sigmoid function maps any real-valued number to a value between 0 and 1.
    
    Parameters
    ----------
    z : float
        The input to the sigmoid function.
    
    Returns
    -------
    float
        The output of the sigmoid function.
    """
    return 1 / (1 + np.exp(-z))

def train_mle(X, y, alpha=0.001, epochs=1000):
    # Initialize weights
    b0, b1 = 0.0, 0.0
    
    print(f"{'Epoch':<8} | {'b0':<10} | {'b1':<10} | {'Likelihood':<10}")
    print("-" * 45)
    
    for i in range(epochs):
        # 1. Predict
        z = b0 + b1 * X
        y_hat = sigmoid(z)
        
        # 2. Gradient Ascent Equation (Maximize)
        # Update = weight + (learning_rate * gradient)
        error = y - y_hat
        grad_b0 = np.sum(error)
        grad_b1 = np.sum(error * X)
        
        b0 += alpha * grad_b0
        b1 += alpha * grad_b1
        
        if i % 200 == 0:
            likelihood = np.prod(y_hat**y * (1-y_hat)**(1-y))
            print(f"{i:<8} | {b0:<10.4f} | {b1:<10.4f} | {likelihood:<10.4f}")
            
    return b0, b1

intercept, weight = train_mle(df['credit_score'], df['loan_granted'])
print(f"\nFinal MLE Weights: Intercept={intercept:.4f}, Weight={weight:.4f}")