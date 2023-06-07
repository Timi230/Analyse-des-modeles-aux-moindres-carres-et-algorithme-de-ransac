import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit

# ---------- Exercice 1 ----------

# n = 100
# x = np.linspace(0,1,n)
# y1,y2 = [],[]

# X1 = np.ones((n,3))
# X2 = np.ones((n,4))
# Y1 = np.ones((n,1))
# Y2 = np.ones((n,1))
# model1 = np.ones((n,1))
# model2 = np.ones((n,1))

# eps = np.random.normal(0,2,n)
# for i in range(n):
#     X1[i][1] = x[i] 
#     X1[i][2] = x[i]**2 

#     y1i = 10 + 2*x[i] + 8*(x[i]**2) + eps[i]
#     y1.append(y1i)
#     Y1[i][0] = y1i

# for i in range(n):
#     X2[i][1] = x[i] 
#     X2[i][2] = x[i]**2 
#     X2[i][3] = np.sin(np.pi*x[i]) 

#     y2i = 10 + 2*x[i] + 8*(x[i]**2) + 10*np.sin(np.pi*x[i]) + eps[i]
#     y2.append(y2i)
#     Y2[i][0] = y2i

# beta1 = np.dot(np.dot(np.linalg.inv(np.dot(X1.T,X1)),X1.T),Y1)
# beta2 = np.dot(np.dot(np.linalg.inv(np.dot(X2.T,X2)),X2.T),Y2)

# for i in range(n):
#     model1[i] = beta1[0] + beta1[1]*x[i] + beta1[2]*(x[i]**2) + eps[i]
#     model1[i] = beta2[0] + beta2[1]*x[i] + beta2[2]*(x[i]**2) + beta2[3]*np.sin(np.pi*x[i]) + eps[i]

# print('beta1 : ', beta1)
# print('beta2 : ', beta2)

# plt.figure(1)
# plt.plot(x, y1,label = 'y1')
# plt.plot(x, model1, label = 'model1')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Fitting curves')
# plt.legend()

# plt.figure(2)
# plt.plot(x, y2,label = 'y2')
# plt.plot(x, model2, label = 'model2')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Fitting curves')
# plt.legend()

# plt.show()

# Génération des données

n = 6 # nombre d'échantillons observés
# n = 1000 # meilleur beta
x = np.linspace(0,1,n)
y = []
y_sin = []

# Génération d'epsilon et du vecteur y
for i in range(n):
    eps = np.random.normal(0, np.sqrt(2)) # bruit (loi normale, variance 2)

    yi = 10 + 2*x[i] + 8*(x[i]**2) + eps
    yi_sin = 10 + 2*x[i] + 8*(x[i]**2) + 10*np.sin(np.pi*x[i]) + eps

    y.append(yi)
    y_sin.append(yi_sin)

# Génération des matrices X et Y
X = np.ones((n,3))
Y = np.ones((n,1))

X_sin = np.ones((n,4))
Y_sin = np.ones((n,1))

for i in range(n):
    X[i,1] = x[i]
    X[i,2] = x[i]**2 

    Y[i] = y[i]

    X_sin[i,1] = x[i]
    X_sin[i,2] = x[i]**2 
    X_sin[i,3] = np.sin(np.pi*x[i])

    Y_sin[i] = y_sin[i]

# Génération des paramètres

beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)
beta_sin = np.dot(np.dot(np.linalg.inv(np.dot(X_sin.T,X_sin)),X_sin.T),Y_sin)

print("beta     : ", beta)
print("beta_sin : ", beta_sin)

# Vérification
# Tracé des résultats
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.') # Données observées
plt.plot(x, beta[0] + beta[1]*x + beta[2]*(x**2), 'r.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Données observées pour une distribution aléatoire du bruit + modèle')
plt.show()

plt.figure(figsize = (10,8))
plt.plot(x, y_sin, 'b.') # Données observées
plt.plot(x, beta_sin[0] + beta_sin[1]*x + beta_sin[2]*(x**2) + 10*np.sin(np.pi*x), 'r.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Données observées pour une distribution aléatoire du bruit + modèle (sin)')
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
print("p gpt =", p_estimated)
print("e gpt =", e_estimated)


# Tableau
theta = [43, 45, 52, 93, 108, 126]
f_theta = [4.7126, 4.5542, 4.0419, 2.2187, 1.8910, 1.7599]

n_theta = len(theta)

X_theta = np.ones((n_theta,2))
Y_theta = np.ones((n_theta,1))

for i in range(n_theta):
    X_theta[i,1] = -np.cos(theta[i])

    Y_theta[i] = 1/f_theta[i]


beta_theta = np.dot(np.dot(np.linalg.inv(np.dot(X_theta.T,X_theta)),X_theta.T),Y_theta)


p = 1/beta_theta[0]
e = p*beta_theta[1]

print("X_theta : ",X_theta)
print("Y_theta : ",Y_theta)
print("ftheta",f_theta)
print("test", p/(1 - e*np.cos(theta)))

print("beta_theta : ", beta_theta)
print("p : ", p)
print("e : ", e)

# Vérification
# Tracé des résultats
plt.figure(figsize = (10,8))
plt.plot(theta, f_theta, 'b.') # Données observées
plt.plot(theta, p / (1 - e*np.cos(theta)), 'r.') # Modèle 
plt.plot(theta, p_estimated / (1 - e_estimated*np.cos(theta)), 'k.') # Modèle 
plt.xlabel('theta')
plt.ylabel('f(theta)')
plt.title('Données observées + modèle')
plt.show()

# ---------- Exercice 3 ----------
# # Set the random seed for reproducibility
# np.random.seed(0)

# # Step 1: Generate the data
# n = 5000
# x = np.linspace(0, 1, n)
# y = 10 + 2 * x

# # Step 2: Introduce outliers
# outlier_indices = np.random.choice(n, size=10, replace=False)
# y[outlier_indices] = np.random.normal(0, 1, size=10)

# # Plot the data
# plt.figure(figsize=(10, 8))
# plt.plot(x, y, 'b.')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Data with Outliers')
# plt.show()

# # Step 3: Function to compare models based on inliers
# def compare_models(model1, model2, x, y, threshold):
#     inliers1 = np.abs(model1[0] * x + model1[1] - y) < threshold
#     inliers2 = np.abs(model2[0] * x + model2[1] - y) < threshold
#     return np.sum(inliers1) > np.sum(inliers2)

# # Step 4: Function to compute a model from random samples
# def compute_model(x, y):
#     indices = np.random.choice(n, size=2, replace=False)
#     x_samples = x[indices]
#     y_samples = y[indices]
#     model = np.polyfit(x_samples, y_samples, 1)
#     return model

# # Step 5: RANSAC algorithm
# threshold = 1
# best_model = None

# for _ in range(10):
#     model = compute_model(x, y)
#     if best_model is None or compare_models(model, best_model, x, y, threshold):
#         best_model = model

# # Plot the best model
# best_y = np.polyval(best_model, x)

# plt.figure(figsize=(10, 8))
# plt.plot(x, y, 'b.', label='Data')
# plt.plot(x, best_y, 'r-', label='Best Model')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Best Model from RANSAC')
# plt.legend()
# plt.show()

# # Step 6: Compare with naive approach
# naive_model = np.polyfit(x, y, 1)
# naive_y = np.polyval(naive_model, x)

# plt.figure(figsize=(10, 8))
# plt.plot(x, y, 'b.', label='Data')
# plt.plot(x, naive_y, 'g-', label='Naive Model')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Naive Model')
# plt.legend()
# plt.show()

n_out = 100 # nombre d'échantillons observés
# n = 1000 # meilleur beta
x = np.linspace(0,1,n_out)
y = []

# Génération d'epsilon et du vecteur y
for i in range(n_out):
    yi = 10 + 2*x[i] 

    y.append(yi)

# Vérification
# Tracé des résultats
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Données observées pour une distribution aléatoire du bruit')
plt.show()

y_out = np.copy(y)

n_outliers = 10
# n_outliers = 200
# n_outliers = 500

for o in range(n_outliers):
    i_random = random.randint(0,n_out)
    y_out[i_random] = np.random.normal(11, 1) # en gaussian

# Vérification
# Tracé des résultats
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.')
plt.plot(x, y_out, 'k.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Données observées pour une distribution aléatoire du bruit')
plt.show()


# Intialisation des vecteurs x et y, de la variance et de la covariance
x_moy = 0
y_moy = 0
y2_moy = 0 # modèle non-linéaire
y4_moy = 0 # outliers

sigma2_x = 0
Cxy = 0
Cxy2 = 0 # modèle non-linéaire
Cxy4 = 0 # outliers

# Calcul de la variance et de la covariance
for i in range(1,n):
    x_moy += x[i]
    y_moy += y[i]
    y2_moy += y2[i]
    y4_moy += y4[i]

x_moy = (1/n)*x_moy
y_moy = (1/n)*y_moy
y2_moy = (1/n)*y2_moy
y4_moy = (1/n)*y4_moy

for i in range(1,n):
    sigma2_x += (x[i] - x_moy)**2
    Cxy += np.dot((x[i] - x_moy),(y[i] - y_moy))
    Cxy2 += np.dot((x[i] - x_moy),(y2[i] - y2_moy))
    Cxy4 += np.dot((x[i] - x_moy),(y4[i] - y4_moy))
sigma2_x = (1/n)*sigma2_x
Cxy = (1/n)*Cxy
Cxy2 = (1/n)*Cxy2
Cxy4 = (1/n)*Cxy4

# Calcul de beta1 et beta2
beta2 = Cxy / sigma2_x
beta1 = y_moy - beta2*x_moy


def inliers(m1,m2):
    
    
    
    pass
