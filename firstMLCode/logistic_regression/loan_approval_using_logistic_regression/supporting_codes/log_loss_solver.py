import numpy as np
import pandas as pd

# Dataset
data = {'credit_score': [600, 700, 750], 'loan_granted': [0, 1, 1]}
df = pd.DataFrame(data)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_log_loss(X, y, alpha=0.001, epochs=1000):
    b0, b1 = 0.0, 0.0
    n = len(y)
    
    print(f"{'Epoch':<8} | {'b0':<10} | {'b1':<10} | {'Log Loss':<10}")
    print("-" * 45)
    
    for i in range(epochs):
        # 1. Predict
        z = b0 + b1 * X
        y_hat = sigmoid(z)
        
        # 2. Log Loss Equation (Minimize)
        # Loss = -1/n * sum(y*log(y_hat) + (1-y)*log(1-y_hat))
        loss = -np.mean(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))
        
        # 3. Gradient Descent Equation
        # Update = weight - (learning_rate * gradient)
        error = y_hat - y
        grad_b0 = (1/n) * np.sum(error)
        grad_b1 = (1/n) * np.sum(error * X)
        
        b0 -= alpha * grad_b0
        b1 -= alpha * grad_b1
        
        if i % 200 == 0:
            print(f"{i:<8} | {b0:<10.4f} | {b1:<10.4f} | {loss:<10.4f}")
            
    return b0, b1

intercept, weight = train_log_loss(df['credit_score'], df['loan_granted'])
print(f"\nFinal Log Loss Weights: Intercept={intercept:.4f}, Weight={weight:.4f}")