import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit

# ---------- Exercice 1 ----------

n = 100
x = np.linspace(0,1,n)
y1,y2 = [],[]

X1 = np.ones((n,3))
X2 = np.ones((n,4))
Y1 = np.ones((n,1))
Y2 = np.ones((n,1))
model1 = np.ones((n,1))
model2 = np.ones((n,1))

eps = np.random.normal(0,2,n)
for i in range(n):
    X1[i][1] = x[i] 
    X1[i][2] = x[i]**2 

    y1i = 10 + 2*x[i] + 8*(x[i]**2) + eps[i]
    y1.append(y1i)
    Y1[i][0] = y1i

for i in range(n):
    X2[i][1] = x[i] 
    X2[i][2] = x[i]**2 
    X2[i][3] = np.sin(np.pi*x[i]) 

    y2i = 10 + 2*x[i] + 8*(x[i]**2) + 10*np.sin(np.pi*x[i]) + eps[i]
    y2.append(y2i)
    Y2[i][0] = y2i

beta1 = np.dot(np.dot(np.linalg.inv(np.dot(X1.T,X1)),X1.T),Y1)
beta2 = np.dot(np.dot(np.linalg.inv(np.dot(X2.T,X2)),X2.T),Y2)

for i in range(n):
    model1[i] = beta1[0] + beta1[1]*x[i] + beta1[2]*(x[i]**2) + eps[i]
    model1[i] = beta2[0] + beta2[1]*x[i] + beta2[2]*(x[i]**2) + beta2[3]*np.sin(np.pi*x[i]) + eps[i]

print('beta1 : ', beta1)
print('beta2 : ', beta2)

plt.figure(1)
plt.plot(x, y1,label = 'y1')
plt.plot(x, model1, label = 'model1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting curves')
plt.legend()

plt.figure(2)
plt.plot(x, y2,label = 'y2')
plt.plot(x, model2, label = 'model2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting curves')
plt.legend()

plt.show()

# ---------- Exercice 2 ----------
# Define the model function
def model(theta, p, e):
    return p / (1 - e * np.cos(np.radians(theta)))

# Define the observed data
angles = np.array([43, 45, 52, 93, 108, 126])
observed_values = np.array([4.7126, 4.5542, 4.0419, 2.2187, 1.8910, 1.7599])

# Convert angles to radians
theta = np.radians(angles)

# Fit the model to the data
params, _ = curve_fit(model, theta, observed_values)

# Extract the estimated parameters
p_estimated, e_estimated = params

print("Estimated parameters:")
print("p =", p_estimated)
print("e =", e_estimated)

# ---------- Exercice 3 ----------
# Set the random seed for reproducibility
np.random.seed(0)

# Step 1: Generate the data
n = 5000
x = np.linspace(0, 1, n)
y = 10 + 2 * x

# Step 2: Introduce outliers
outlier_indices = np.random.choice(n, size=10, replace=False)
y[outlier_indices] = np.random.normal(0, 1, size=10)

# Plot the data
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data with Outliers')
plt.show()

# Step 3: Function to compare models based on inliers
def compare_models(model1, model2, x, y, threshold):
    inliers1 = np.abs(model1[0] * x + model1[1] - y) < threshold
    inliers2 = np.abs(model2[0] * x + model2[1] - y) < threshold
    return np.sum(inliers1) > np.sum(inliers2)

# Step 4: Function to compute a model from random samples
def compute_model(x, y):
    indices = np.random.choice(n, size=2, replace=False)
    x_samples = x[indices]
    y_samples = y[indices]
    model = np.polyfit(x_samples, y_samples, 1)
    return model

# Step 5: RANSAC algorithm
threshold = 1
best_model = None

for _ in range(10):
    model = compute_model(x, y)
    if best_model is None or compare_models(model, best_model, x, y, threshold):
        best_model = model

# Plot the best model
best_y = np.polyval(best_model, x)

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b.', label='Data')
plt.plot(x, best_y, 'r-', label='Best Model')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Best Model from RANSAC')
plt.legend()
plt.show()

# Step 6: Compare with naive approach
naive_model = np.polyfit(x, y, 1)
naive_y = np.polyval(naive_model, x)

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b.', label='Data')
plt.plot(x, naive_y, 'g-', label='Naive Model')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Naive Model')
plt.legend()
plt.show()
