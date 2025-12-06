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

"""
Dataset: 10 numeric predictors (e.g., bmi, bp, s1 - s6) with a continuous target.

Goal: Predict disease progression; select the most informative predictors.
"""

# Basic stats
desc = X.describe()
print(f"Describe:{desc}")

# Correlation with target
corr_with_target = pd.Series(np.corrcoef(np.c_[X.values, y.values], rowvar=False)[-1, :-1], index=X.columns)
print(f" Correlation with target: {pd.Series(np.corrcoef(np.c_[X.values, y.values], rowvar=False)[-1, :-1], index=X.columns)}")

print("Correlation with target (top 5):")
print(corr_with_target.sort_values(ascending=False).head(5))

# Optional: full correlation matrix
corr_matrix = X.join(y).corr()
corr_matrix.loc[X.columns, ['disease_progression']].sort_values('disease_progression', ascending=False)


"""
Domain hint: bmi and bp often influence disease progression.

Correlation use: High absolute correlation suggests signal, but watch multicollinearity among s1–s6.
 """
####Filter methods: SelectKBest (F-test and mutual information)

from sklearn.feature_selection import SelectKBest

# F-test for linear relationship strength
skb_f = SelectKBest(score_func=f_regression, k=5).fit(X, y)
f_scores = pd.Series(skb_f.scores_, index=X.columns).sort_values(ascending=False)
print("Top 5 by F-test:")
print(f_scores.head(5))

# Mutual Information for non-linear association
skb_mi = SelectKBest(score_func=mutual_info_regression, k=5).fit(X, y)
mi_scores = pd.Series(skb_mi.scores_, index=X.columns).sort_values(ascending=False)
print("Top 5 by Mutual Information:")
print(mi_scores.head(5))


""" Filter takeaway: Fast screening. F-test assumes linearity; mutual information captures non-linear signals."""

"""Embedded method: LassoCV for sparse selection """

# Standardize then Lasso with cross-validated alpha
lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LassoCV(cv=5, random_state=42, max_iter=5000))
]).fit(X, y)

lasso_coefs = pd.Series(lasso.named_steps['model'].coef_, index=X.columns)
selected_by_lasso = lasso_coefs[lasso_coefs != 0].sort_values(key=abs, ascending=False)

print("Lasso selected features and coefficients:")
print(selected_by_lasso)
print("Alpha chosen:", lasso.named_steps['model'].alpha_)


"""Embedded takeaway: Lasso performs feature selection via coefficient shrinkage. It’s great when 𝑝
 is moderate and some features are redundant. """
 
 
""" Wrapper method: Recursive Feature Elimination (RFE)"""
# RFE with linear regression, selecting top 5
rfe = RFE(estimator=LinearRegression(), n_features_to_select=5)
rfe.fit(X, y)

rfe_selected = X.columns[rfe.support_]
rfe_ranking = pd.Series(rfe.ranking_, index=X.columns).sort_values()

print("RFE selected features:", list(rfe_selected))
print("RFE ranking (1 = selected):")
print(rfe_ranking)


"""Wrapper takeaway: RFE evaluates subsets by model performance, but is computationally heavier."""

def cv_score(features, name):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X[features], y, cv=cv, scoring='r2')
    print(f"{name} | R2 mean: {scores.mean():.3f}, std: {scores.std():.3f}")
    return scores.mean()

# Baseline: all features
cv_score(X.columns, "All features")

# Top 5 by F-test
top5_f = f_scores.head(5).index
cv_score(top5_f, "Top5 F-test")

# Top 5 by Mutual Information
top5_mi = mi_scores.head(5).index
cv_score(top5_mi, "Top5 Mutual Information")

# Lasso-selected (may be <=10 features)
lasso_feats = list(selected_by_lasso.index)
cv_score(lasso_feats, "Lasso-selected")

# RFE-selected
cv_score(list(rfe_selected), "RFE-selected")

"""Interpretation: Prefer the subset that yields higher cross-validated 𝑅2  and lower variance. If differences are small, choose the more interpretable subset. """

"""Final model training and evaluation on a holdout set"""
# Choose the best performing subset (example: lasso_feats)
X_train, X_test, y_train, y_test = train_test_split(X[lasso_feats], y, test_size=0.2, random_state=42)

final_model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
]).fit(X_train, y_train)

y_pred = final_model.predict(X_test)
print("Holdout R2:", r2_score(y_test, y_pred))
print("Holdout RMSE:", root_mean_squared_error(y_test, y_pred))
print("Holdout RMSE:", mean_squared_error(y_test, y_pred))


"""Practical guidance
Start with domain knowledge and correlation to shortlist candidates.

Use filter methods for quick screening; cross-check with mutual information if non-linearity is suspected.

Apply Lasso to remove weak or redundant predictors; tune via cross-validation.

Use RFE when you want a specific number of features based on model performance.

Validate choices with cross-validation and a holdout set; prioritize stable performance and interpretability.

Want this on your data?
Label: Share your feature names, sample size, and target definition.

Label: Tell me if interpretability or raw accuracy matters more.

Label: I can adapt the code to your exact dataset and produce a clean, reproducible notebook."""