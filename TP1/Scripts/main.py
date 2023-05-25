import numpy as np
import matplotlib.pyplot as plt
import random

# ---------- Exercice 1 ----------

# Génération des données
n = 10000
# n = 1000     # diminution du nombre d'échantillons observés
# n = 100000   # augmentation du nombre d'échantillons observés
x = np.linspace(0,1,n)
y = []
y2 = []

# Génération d'epsilon et du vecteur y
for i in range(n):
    eps = random.uniform(-1,1)
    yi = 10 + 2*x[i] + eps
    y2 = 10 + 2*(x[i]**2) + eps # modèle non-linéaire
    y.append(yi)

# Tracé
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simulation des données')
plt.show()

# ---------- Exercice 2 ----------

# Intialisation des vecteurs x et y, de la variance et de la covariance
x_moy, y_moy, y2_moy = 0, 0, 0
sigma2_x, Cxy, Cxy2 = 0, 0, 0

# Calcul de la variance et de la covariance
for i in range(1,n):
    x_moy += x[i]
    y_moy += y[i]
    y2_moy += y2[i]
x_moy = (1/n)*x_moy
y_moy = (1/n)*y_moy
y2_moy = (1/n)*y2_moy

for i in range(1,n):
    sigma2_x += (x[i] - x_moy)**2
    Cxy += np.dot((x[i] - x_moy),(y[i] - y_moy))
    Cxy2 += np.dot((x[i] - x_moy),(y2[i] - y2_moy))
sigma2_x = (1/n)*sigma2_x
Cxy = (1/n)*Cxy
C2xy = (1/n)*C2xy

# Calcul de beta1 et beta2
beta2 = Cxy / sigma2_x
beta2_2 = Cxy / sigma2_x
beta1 = y_moy - beta2*x_moy
beta1_2 = y_moy - beta2*x_moy

# Affichage de la variance, de la covariance, de beta1 et de beta2
print('sigma carré de x : ', sigma2_x)
print('Cxy : ', Cxy)
print('Beta1 (linéaire): ', beta1)
print('Beta2 (linéaire): ', beta2)
print('Beta1 (non-linéaire): ', beta1_2)
print('Beta2 (non-linéaire): ', beta2_2)

# Tracés
plt.plot(x, y, 'b.') # tracé de la simulation des données
plt.plot(x, beta1 + beta2*x, 'r.') # tracé du modèle
plt.xlabel('x')
plt.ylabel('beta1 + beta2*x')
plt.title('Tracé du modèle sur la simulation des données')
plt.show()

# Valeurs prédites pour x = 1/2, x = 0 et x = 1
print('Valeur prédite pour x = 1/2 : ',beta1 + beta2*(1/2))
print('Valeur prédite pour x = 0   : ',beta1 + beta2*0)
print('Valeur prédite pour x = 1   : ',beta1 + beta2*1)

# Test pour un modèle non-linéaire
print('Valeur prédite pour x = 0   : ',beta1_2 + beta2_2*0)

# ---------- Exercice 3 ----------

beta2_3_estimates = []
beta1_3_estimates = []

# for _ in range(n):
#     y = []
#     for i in range(n):
#         eps = np.random.normal(0, 1)
#         yi = 10 + 2 * x[i] + eps
#         y.append(yi)
    
#     # estimate beta2
#     beta2_3 = np.cov(x, y, ddof=0)[0, 1] / np.var(x, ddof=0)
#     beta2_3_estimates.append(beta2_3)

y_3 = []

# Génération d'epsilon et du vecteur y
for i in range(n):
    eps_3 = np.random.normal(0, 1)
    y_3i = 10 + 2*x[i] + eps
    y_3.append(y_3i)

# plot the results
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b.')
plt.plot(x, beta1 + beta2*x, 'r.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Observed data')
plt.show()

# plot histogram
plt.figure(figsize=(10, 8))
plt.hist(beta2_3_estimates, bins=30, edgecolor='black')
plt.xlabel('Estimated β2')
plt.ylabel('Frequency')
plt.title('Histogram of Estimated β2')
plt.show()

"""
 l'utilisation d'un histogramme pour évaluer β2 permet d'obtenir des informations 
 sur la distribution, le biais, la précision et la détection des valeurs aberrantes 
 dans les estimations. Cela facilite l'analyse et l'évaluation des performances de 
 l'estimation de β2.
"""

# ---------- Exercice 4 ----------
num_outliers = 50  # number of outliers to introduce
outlier_indices = random.sample(range(n), num_outliers)
outlier_magnitude = 20  # magnitude of the outliers

for i in outlier_indices:
    y[i] += outlier_magnitude * np.random.choice([-1, 1])

# plot the results
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b.')
plt.plot(x, beta1 + beta2*x, 'r.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Observed data with outliers')
plt.show()

""""
l'ajout d'outliers dans les données est une technique utile pour évaluer la robustesse des méthodes, 
étudier l'influence des valeurs extrêmes, tester la résilience des modèles et fournir des visualisations 
illustratives des effets des outliers. Cela permet d'obtenir une meilleure compréhension des données et de 
s'assurer de la fiabilité des analyses et des estimations.
"""