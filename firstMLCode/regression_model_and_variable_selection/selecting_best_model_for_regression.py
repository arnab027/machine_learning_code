import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error,root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
# import matplotlib
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
# matplotlib.use('Agg')
# %matplotlib inline
# Load dataset
data = load_diabetes()

##see all records 
""" Removing highly correlated independent variables using Pearson correlation is a common technique in 
feature selection to address multicollinearity.  """

""" Step-by-step example: Selecting predictors for regression in Python
Here’s a practical, end-to-end workflow using a built-in dataset (Diabetes) from scikit-learn. 
We’ll explore domain understanding, correlations, filter methods, embedded methods (Lasso), 
and wrapper methods (RFE), then compare results with cross-validation.
"""
independent_df = pd.DataFrame(data.data, columns=data.feature_names)
dependent_Df =pd.DataFrame(data.target, columns=['disease_progression'])
current_dir = Path.cwd()
print(f"current directory: {current_dir}")

joined_df = pd.concat([independent_df, dependent_Df], axis=1)
print(f"all records: \n {joined_df}")
print(joined_df.corr().head())
plt.figure(figsize=(10,10))
sns.heatmap(joined_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
# plt.show(block=True) ## show correlation matrix (block=False)
plt.savefig(current_dir/"regression_model_and_variable_selection/correlation_matrix.png") ## save correlation matrix as png file

"""
before dropping independent variables that are highly correlated  , drop the dependent variable first
and then do train and test split.

then using the training dataset, calculate correlation matrix and drop independent 
variables that are highly correlated.

Drop the same columns from test dataset as well.
After dropping the columns, then do model training and testing.
"""

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='disease_progression')

print(X.head())
print(f"dependent variable \n : {y}")
print("Shape:", X.shape) ## Shape: (442, 10)

"""
       age       sex       bmi        bp        s1        s2        s3        s4        s5        s6
0  0.038076  0.050680  0.061696  0.021872 -0.044223 -0.034821 -0.043401 -0.002592  0.019907 -0.017646
1 -0.001882 -0.044642 -0.051474 -0.026328 -0.008449 -0.019163  0.074412 -0.039493 -0.068332 -0.092204
2  0.085299  0.050680  0.044451 -0.005670 -0.045599 -0.034194 -0.032356 -0.002592  0.002861 -0.025930
3 -0.089063 -0.044642 -0.011595 -0.036656  0.012191  0.024991 -0.036038  0.034309  0.022688 -0.009362
4  0.005383 -0.044642 -0.036385  0.021872  0.003935  0.015596  0.008142 -0.002592 -0.031988 -0.046641
"""
"""selecting best model by comparing R2 scores on training and testing datasets."""
def compare_models_with_r2(X, y):
    # 1. Generate synthetic data (some noise, some complexity)
    # X, y = make_regression(n_samples=1000, n_features=20, noise=15.0, random_state=42)
    
    # 2. Split into Training and Testing sets
    # CRITICAL STEP: Never evaluate R2 on the training set alone! 
    # High training R2 usually means overfitting.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Define the models to test
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print(f"{'Model Name':<20} | {'Train R2':<10} | {'Test R2 (The Real Score)':<25}")
    print("-" * 65)
    
    best_model_name = None
    best_test_r2 = -float('inf')
    
    # 4. Train and Evaluate
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        # Calculate R2
        r2_train = r2_score(y_train, train_preds)
        r2_test = r2_score(y_test, test_preds)
        
        results[name] = r2_test
        
        print(f"{name:<20} | {r2_train:.4f}     | {r2_test:.4f}")
        
        if r2_test > best_test_r2:
            best_test_r2 = r2_test
            best_model_name = name
            
    print("-" * 65)
    print(f"\nThe best model for this problem is: {best_model_name}")
    print(f"It explains {best_test_r2*100:.2f}% of the variance in the unseen test data.")

compare_models_with_r2(X, y)