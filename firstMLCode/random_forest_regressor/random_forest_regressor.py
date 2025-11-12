import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
# import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("Position_Salaries.csv")

print(df.head())
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

print(X)
print(y)
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)
regressor.predict([[6.5]])

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

