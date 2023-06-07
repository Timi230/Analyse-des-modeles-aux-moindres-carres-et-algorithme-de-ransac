import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import random
from library_TP3 import *

# ---------- Exercice 1 ----------

# Génération des données

n = 100 # nombre d'échantillons observés

x = np.linspace(0,1,n)

eps = esp_aleatoire_gaussian(n, 0, np.sqrt(2))

y1 = TP1_y_exo1_3(n, x, eps)



m_TP3 = calcul_model(x, y1, n)

Liste_n_m = inliers(m_TP3, 1,n,x,y1)

n_in = len(Liste_n_m)

x_in = []
y_in = []

for i in range(n_in):
    x_in.append(Liste_n_m[i][0])
    y_in.append(Liste_n_m[i][1])

m_TP2 = [10.000000000000002, 2.000000000000001]
m_TP3_re = reestimated(Liste_n_m)

plt.figure(figsize = (10,8))
plt.plot(x, y1, 'b.')
plt.plot(x, m_TP2[0] + m_TP2[1]*x, 'r.')
plt.plot(x_in, y_in, 'k.')
plt.plot(x, m_TP3[0] + m_TP3[1]*x, 'g.')
plt.plot(x, m_TP3_re[0] + m_TP3_re[1]*x, 'm.')
plt.show()

print("La meilleure stratégie est : ", compare(m_TP2, m_TP3, 1, n, x, y1))

# ---------- Exercice 2 ----------

# Tableau
theta = [43, 45, 52, 93, 108, 126]
f_theta = [4.7126, 4.5542, 4.0419, 2.2187, 1.8910, 1.7599]

param = [0,0]

solution = optimize.least_squares(fun, param, args = (theta, f_theta))

beta = solution.x

print("beta : ", beta)

# ---------- Exercice 3 ----------

A = np.array([[1,2],[3,4],[5,6]])
U,S,V = np.linalg.svd(A)
print("U : ", U)
print("S : ", S)
print("V : ", V)

sigma1 = S[0]
sigma2 = S[1]

S = np.array([[sigma1,0],[0,sigma2],[0,0]])

A_recompose = U@S@V
print("A recomposé : ", A_recompose)
