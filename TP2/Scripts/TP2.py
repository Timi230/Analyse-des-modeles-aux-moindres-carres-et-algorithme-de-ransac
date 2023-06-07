import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
from library_TP2 import *

# ---------- Exercice 1 ----------

# Génération des données

n = 100 # nombre d'échantillons observés
x = np.linspace(0,1,n)


y1 = y_exo1(n, x, esp_aleatoire_gaussian(n, 0, np.sqrt(2)))
y_sin = y_exo1_sin(n, x, esp_aleatoire_gaussian(n, 0, np.sqrt(2)))



X = mat_X_exo1(x, n)
Y = mat_Y_exo1(y1, n)
X_sin = mat_X_sin(x, n)
Y_sin = mat_Y_exo1(y_sin, n)


beta = calcul_beta(X, Y)
beta_sin = calcul_beta(X_sin, Y_sin)

print("beta     : ", beta)
print("beta_sin : ", beta_sin)

model(x, y1, beta[0] + beta[1]*x + beta[2]*(x**2), 'Données observées pour une distribution aléatoire du bruit + modèle', 'b.', 'r.', 'x', 'y')
model(x, y1, beta_sin[0] + beta_sin[1]*x + beta_sin[2]*(x**2) + 10*np.sin(np.pi*x), 'Données observées pour une distribution aléatoire du bruit + modèle', 'b.', 'r.', 'x', 'y')

# ---------- Exercice 2 ----------

# Tableau
theta = [43, 45, 52, 93, 108, 126]
f_theta = [4.7126, 4.5542, 4.0419, 2.2187, 1.8910, 1.7599]

n_theta = len(theta)

X_theta = np.ones((n_theta,2))
Y_theta = np.ones((n_theta,1))

for i in range(n_theta):
    X_theta[i,1] = -np.cos(theta[i])
    Y_theta[i] = 1/f_theta[i]


beta_theta = calcul_beta(X_theta,  Y_theta)


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

model(theta, f_theta, p / (1 - e*np.cos(theta)), 'Données observées + modèle', 'b.', 'r.', 'theta', 'f(theta)')

# ---------- Exercice 3 ----------

n = 100

y3 = y_exo3(n, x)

traces(x, y3, 'Données observées pour une distribution aléatoire du bruit', 'b.', 'x', 'y')

y_out = np.copy(y3)

n_outliers = 10
y_out = outliers(y_out, n, n_outliers, 11, 0.5) # en gaussian

traces(x, y_out, 'Données observées pour une distribution aléatoire du bruit', 'b.', 'x', 'y')


m1 = calcul_model(x, n)
m2 = calcul_model(x, n)


m_TP1 = [10.037398645699664, 1.9031078680985294]
m_TP2 = inliers(m1, m2, 1, n, x, y3)

print("La meilleure stratégie est : ", compare(m_TP1, m_TP2, 1, n, x, y3))