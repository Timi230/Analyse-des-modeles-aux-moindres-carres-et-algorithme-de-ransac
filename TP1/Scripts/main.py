import numpy as np
# from scipy import optimize
import matplotlib.pyplot as plt
import random

# ---------- Exercice 1 ----------

# Génération des données
# n = 10000
n = 1000     # diminution du nombre d'échantillons observés
# n = 100000   # augmentation du nombre d'échantillons observés
x = np.linspace(0,1,n)
y = []
y2 = []
y3 = []

# Génération d'epsilon et du vecteur y
for i in range(n):
    eps = random.uniform(-1,1) # bruit (aléatoire)
    eps3 = np.random.normal(0,1) # bruit (loi normale)

    yi = 10 + 2*x[i] + eps # modèle linéaire
    y2i = 10 + 2*(x[i]**2) + eps # modèle non-linéaire
    y3i = 10 + 2*x[i] + eps3

    y.append(yi)
    y2.append(y2i)
    y3.append(y3i)

# Vérification
# Tracé des résultats
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Données observées pour une distribution aléatoire du bruit')
plt.show()

# ---------- Exercice 2 ----------
y4 = np.copy(y) # exo4

n_outliers = 20
# n_outliers = 200
# n_outliers = 500

for o in range(n_outliers):
    i_random = random.randint(0,n)
    y4[i_random] = random.uniform(9,13)


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

beta2_2 = Cxy2 / sigma2_x
beta1_2 = y2_moy - beta2_2*x_moy

beta2_4 = Cxy4 / sigma2_x
beta1_4 = y4_moy - beta2_4*x_moy

# Vérification
# Tracé des résultats
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.') # Données observées
plt.plot(x, beta1 + beta2*x, 'r.') # Modèle LS
plt.xlabel('x')
plt.ylabel('y')
plt.title('Données observées pour une distribution aléatoire du bruit + modèle LS')
plt.show()

# Valeurs prédites pour x = 1/2, x = 0 et x = 1
print('Valeur prédite pour x = 1/2 : ',beta1 + beta2*(1/2))
print('Valeur prédite pour x = 0   : ',beta1 + beta2*0)
print('Valeur prédite pour x = 1   : ',beta1 + beta2*1)

# Affichage de la variance, de la covariance, de beta1 et de beta2
print('Variance de x        : ', sigma2_x)
print('Covariance de x et y : ', Cxy)
print('Beta1 (linéaire)     : ', beta1)
print('Beta2 (linéaire)     : ', beta2)
print('Beta1 (non-linéaire) : ', beta1_2)
print('Beta2 (non-linéaire) : ', beta2_2)

# Test pour un modèle non-linéaire
print('Valeur prédite pour x = 0   : ',beta1_2 + beta2_2*0)

# ---------- Exercice 3 ----------

# Vérification
# Tracé des résultats
plt.figure(figsize = (10,8))
plt.plot(x, y3, 'b.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Données observées pour une distribution du bruit selon une loi normale')
plt.show()

n_simu = 1000
Liste_beta2_hist = []

for i in range(n_simu):
    y_hist = []

    for i in range(n):
        eps_hist = np.random.normal(0, 1) # bruit (loi normale)

        yi_hist = 10 + 2*x[i] + eps_hist
        y_hist.append(yi_hist)

    y_moy_hist = 0

    sigma2_x_hist = 0
    Cxy_hist = 0

    for i in range(1,n):
        y_moy_hist += y_hist[i]

    y_moy_hist = (1/n)*y_moy_hist

    for i in range(1,n):
        Cxy_hist += np.dot((x[i] - x_moy),(y_hist[i] - y_moy_hist))

    Cxy_hist = (1/n)*Cxy_hist

    beta2_hist = Cxy_hist / sigma2_x

    Liste_beta2_hist.append(beta2_hist)


# Histogramme
plt.hist(Liste_beta2_hist, range = (1,3), bins = 100, color = 'yellow',
         edgecolor = 'red')
plt.xlabel('beta2')
plt.ylabel('Nombre d\'occurences')
plt.title('Histogramme de beta2')

# beta2_3_estimates = []
# beta1_3_estimates = []

# for _ in range(n):
#     y = []
#     for i in range(n):
#         eps = np.random.normal(0, 1)
#         yi = 10 + 2 * x[i] + eps
#         y.append(yi)
    
#     # estimate beta2
#     beta2_3 = np.cov(x, y, ddof=0)[0, 1] / np.var(x, ddof=0)
#     beta2_3_estimates.append(beta2_3)

# # plot histogram
# plt.figure(figsize=(10, 8))
# plt.hist(beta2_3_estimates, bins=30, edgecolor='black')
# plt.xlabel('Estimated β2')
# plt.ylabel('Frequency')
# plt.title('Histogram of Estimated β2')
# plt.show()

"""
 l'utilisation d'un histogramme pour évaluer β2 permet d'obtenir des informations 
 sur la distribution, le biais, la précision et la détection des valeurs aberrantes 
 dans les estimations. Cela facilite l'analyse et l'évaluation des performances de 
 l'estimation de β2.
"""

# ---------- Exercice 4 ----------
# num_outliers = 50  # number of outliers to introduce
# outlier_indices = random.sample(range(n), num_outliers)
# outlier_magnitude = 20  # magnitude of the outliers

# for i in outlier_indices:
#     y[i] += outlier_magnitude * np.random.choice([-1, 1])

# # plot the results
# plt.figure(figsize=(10, 8))
# plt.plot(x, y, 'b.')
# plt.plot(x, beta1 + beta2*x, 'r.')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Observed data with outliers')
# plt.show()


# Vérification
# Tracé des résultats
plt.figure(figsize = (10,8))
plt.plot(x, y4, 'b.') # Données observées
plt.plot(x, beta1_4 + beta2_4*x, 'r.') # Modèle LS
plt.xlabel('x')
plt.ylabel('y')
plt.title('Données observées pour une distribution aléatoire du bruit + modèle LS avec outliers')
plt.show()


""""
l'ajout d'outliers dans les données est une technique utile pour évaluer la robustesse des méthodes, 
étudier l'influence des valeurs extrêmes, tester la résilience des modèles et fournir des visualisations 
illustratives des effets des outliers. Cela permet d'obtenir une meilleure compréhension des données et de 
s'assurer de la fiabilité des analyses et des estimations.
"""