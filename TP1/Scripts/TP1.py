import numpy as np
# from scipy import optimize
import matplotlib.pyplot as plt
import random
from library_TP1 import *

# ---------- Exercice 1 ----------

# Génération des données

n = 1000     # nombre d'échantillons observés

x = np.linspace(0,1,n)

y1 = y_exo1_3(n, x, eps_aleatoire(n))

traces(x, y1, 'Données observées pour une distribution aléatoire du bruit', 'b.', 'x', 'y')

# ---------- Exercice 2 ----------

beta2_l = beta2(variance(x,n), covariance(x,y1, n))
beta1_l = beta1(x,y1,beta2_l)

y2 = y_exo2(n, x, eps_aleatoire(n))

beta2_nl = beta2(variance(x,n), covariance(x,y2, n))
beta1_nl = beta1(x,y2,beta2_nl)

#model(x,y1, beta1_l  + beta2_l * x, 'Données observées pour une distribution aléatoire du bruit + modèle linéaire', 'b.', 'r.', 'x', 'y')
#model(x,y2, beta1_nl + beta2_nl*(x**2), 'Données observées pour une distribution aléatoire du bruit + modèle non-linéaire', 'b.', 'r.', 'x', 'y')

# Valeurs prédites pour x = 1/2, x = 0 et x = 1
print('Valeur prédite pour x = 1/2 : ',beta1_l + beta2_l*(1/2))
print('Valeur prédite pour x = 0   : ',beta1_l + beta2_l*0)
print('Valeur prédite pour x = 1   : ',beta1_l + beta2_l*1)

# Affichage de la variance, de la covariance, de beta1 et de beta2
print('Variance de x        : ', variance(x, n))
print('Covariance de x et y : ', covariance(x, y2, n))
print('Beta1 (linéaire)     : ', beta1_l)
print('Beta2 (linéaire)     : ', beta2_l)
print('Beta1 (non-linéaire) : ', beta1_nl)
print('Beta2 (non-linéaire) : ', beta2_nl)

# Test pour un modèle non-linéaire
print('Valeur prédite pour x = 0   : ',beta1_nl + beta2_nl*0)

# ---------- Exercice 3 ----------

y3= y_exo1_3(n, x, esp_aleatoire_gaussian(n, 0, 1))

traces(x, y3, 'Données observées pour une distribution du bruit selon une loi normale', 'b.', 'x', 'y')

n_simu = 1000
Liste_beta2_hist = []

for i in range(n_simu):
    y3 = y_exo1_3(n, x, esp_aleatoire_gaussian(n, 0, 1))
    var = variance(x, n)
    cov = covariance(x, y3, n)
    beta2_hist = beta2(var, cov)
    Liste_beta2_hist.append(beta2_hist)

# Histogramme
plt.hist(Liste_beta2_hist, range = (1,3), bins = 100, color = 'yellow',
         edgecolor = 'red')
plt.xlabel('beta2')
plt.ylabel('Nombre d\'occurences')
plt.title('Histogramme de beta2')
plt.show()

"""
 l'utilisation d'un histogramme pour évaluer β2 permet d'obtenir des informations 
 sur la distribution, le biais, la précision et la détection des valeurs aberrantes 
 dans les estimations. Cela facilite l'analyse et l'évaluation des performances de 
 l'estimation de β2.
"""

# ---------- Exercice 4 ----------

y4 = np.copy(y1) # exo4

n_outliers = 20

y4 = outliers(y4, n, n_outliers, 9, 13)

beta2_4 = beta2(variance(x,n), covariance(x,y4, n))
beta1_4 = beta1(x,y4,beta2_4)

model(x, y4, beta1_4 + beta2_4*x, 'Données observées pour une distribution aléatoire du bruit + modèle linéaire avec outliers', 'b.', 'r.', 'x', 'y')

print("m_TP1 : ", beta1_4,beta2_4)

fichier = open('../../TP3/Scripts/doc_comparaison.txt')
fichier.write("beta2_4 : ", beta2_4, "\n")
fichier.write("beta1_4 : ", beta1_4, "\n")

fichier.close()

""""
l'ajout d'outliers dans les données est une technique utile pour évaluer la robustesse des méthodes, 
étudier l'influence des valeurs extrêmes, tester la résilience des modèles et fournir des visualisations 
illustratives des effets des outliers. Cela permet d'obtenir une meilleure compréhension des données et de 
s'assurer de la fiabilité des analyses et des estimations.
"""