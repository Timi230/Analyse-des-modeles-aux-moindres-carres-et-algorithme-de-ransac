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

y1 = y_exo1_3(n, x, eps)

def inliers(m, Treshold):
    beta1_m = m[0]
    beta2_m = m[1]

    Liste_n_m = []

    for i in range(n):
        if abs(beta1_m + beta2_m*x[i] - y1[i]) < Treshold:
            Liste_n_m.append([x[i],y1[i]])

    return Liste_n_m

def calcul_model(n):

    beta2_m = beta2(variance(x,n),covariance(x,y1,n))
    beta1_m = beta1(x, y1, beta2_m)

    m = [beta1_m,beta2_m]

    return m

def reestimated(Liste_n_m):

    n_in = len(Liste_n_m)

    x_in = []
    y_in = []

    for i in range(n_in):
        x_in.append(Liste_n_m[i][0])
        y_in.append(Liste_n_m[i][1])

    beta2_m = beta2(variance(x_in,n_in),covariance(x_in,y_in,n_in))
    beta1_m = beta1(x_in, y_in, beta2_m)

    m = [beta1_m,beta2_m]

    return m

def compare(m1,m2, Treshold):

    beta1_m1 = m1[0]
    beta2_m1 = m1[1]
    beta1_m2 = m2[0]
    beta2_m2 = m2[1]

    n_m1 = 0
    n_m2 = 0

    for i in range(n):
        if abs(beta1_m1 + beta2_m1*x[i] - y1[i]) < Treshold:
            n_m1 += 1

        if abs(beta1_m2 + beta2_m2*x[i] - y1[i]) < Treshold:
            n_m2 += 1

    if n_m1 >= n_m2:
        m2 = calcul_model(n)
        return "TP2"
            
    else:
        m1 = calcul_model(n)
        return "TP3"

m_TP3 = calcul_model(n)

Liste_n_m = inliers(m_TP3, 1)

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

print("La meilleure stratégie est : ", compare(m_TP2, m_TP3, 1))

# ---------- Exercice 2 ----------

# Tableau
theta = [43, 45, 52, 93, 108, 126]
f_theta = [4.7126, 4.5542, 4.0419, 2.2187, 1.8910, 1.7599]

param = [0,0]
def fun(parameters, theta, f_theta):
    return parameters[0] / (1-parameters[1]*np.cos(theta))-f_theta

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
