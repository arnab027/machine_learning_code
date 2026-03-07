# This script creates a 3D visualization of the **Log Loss landscape**. In machine learning, we call this a "Loss Surface."

# Seeing this "bowl" shape helps explain why Gradient Descent works so well: no matter where you start (where your initial random weights are), gravity (the gradient) will eventually pull you down to the single lowest point—the **Maximum Likelihood Estimate**.

### `visualize_loss.py`

# ```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Dataset (Same as before)
x_data = np.array([600, 700, 750])
y_data = np.array([0, 1, 1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calculate_log_loss(b0, b1):
    z = b0 + b1 * x_data
    y_hat = sigmoid(z)
    # Average Log Loss across the 3 points
    loss = -np.mean(y_data * np.log(y_hat + 1e-9) + (1 - y_data) * np.log(1 - y_hat + 1e-9))
    return loss

# 2. Create a grid of weight values to test
# We'll look at a range around likely values
b0_range = np.linspace(-30, 10, 50) 
b1_range = np.linspace(-0.02, 0.06, 50)
B0, B1 = np.meshgrid(b0_range, b1_range)

# 3. Calculate Loss for every point on the grid
# vectorize the loss function for the grid
Z_loss = np.array([calculate_log_loss(b0, b1) for b0, b1 in zip(np.ravel(B0), np.ravel(B1))])
Z_loss = Z_loss.reshape(B0.shape)

# 4. Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(B0, B1, Z_loss, cmap='viridis', alpha=0.8, edgecolor='none')

# Add labels
ax.set_title('Log Loss Surface for Loan Prediction', fontsize=15)
ax.set_xlabel('Intercept (beta_0)', fontsize=12)
ax.set_ylabel('Weight (beta_1)', fontsize=12)
ax.set_zlabel('Log Loss (Error)', fontsize=12)

# Add a color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# ```

# ### How to interpret this graph:

# * **The "Bowl" (Convexity):** Notice that there is one clear "bottom" to this surface. This is why we prefer Log Loss over Squared Loss for logistic regression. Squared Loss would look like a mountain range with many small pits (local minima) where the computer could get stuck.
# * **The Flat Areas:** On the far edges, the surface becomes very flat. This is where the "Vanishing Gradient" problem lives. If your weights start too far out in the flats, the slope is so gentle that the computer barely knows which way to move.
# * **The Optimization Path:** When you run the `log_loss_solver.py` script we made earlier, you are essentially dropping a ball onto the side of this bowl and letting it roll to the bottom.

# ---

# ### Comparison of the two scripts

# If you run both the MLE and Log Loss scripts, you will notice:

# 1. **Result:** They converge on almost identical weights for $\beta_0$ and $\beta_1$.
# 2. **Stability:** The Log Loss version is usually more stable with larger datasets because it averages the error ($1/n$), preventing the gradient from exploding.

# Would you like me to show you how to add a "history" tracker to the solver so you can plot the actual path the ball takes down the bowl?