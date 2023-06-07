import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
from library_TP2 import *

# ---------- Exercice 1 ----------

# Génération des données

n = 100 # nombre d'échantillons observés

x = np.linspace(0,1,n)

def y_exo1(n, x, eps):
    y = []
    for i in range(n):
        yi = 10 + 2*x[i] + 8*(x[i]**2) + eps[i] # modèle linéaire
        y.append(yi)
    return y

def y_exo1_sin(n, x, eps):
    y = []
    for i in range(n):
        yi = 10 + 2*x[i] + 8*(x[i]**2) + 10*np.sin(np.pi*x[i]) + eps[i] # modèle linéaire
        y.append(yi)
    return y

y1 = y_exo1(n, x, esp_aleatoire_gaussian(n, 0, np.sqrt(2)))
y_sin = y_exo1_sin(n, x, esp_aleatoire_gaussian(n, 0, np.sqrt(2)))

def mat_X_exo1(x):
    X = np.ones((n,3))
    for i in range(n):
        X[i,1] = x[i]
        X[i,2] = x[i]**2 
    return X

def mat_Y_exo1(y):
    Y = np.ones((n,1))
    for i in range(n):
        Y[i] = y[i]
    return Y

def mat_X_sin(x):
    X_sin = np.ones((n,4))
    for i in range(n):
        X_sin[i,1] = x[i]
        X_sin[i,2] = x[i]**2 
        X_sin[i,3] = np.sin(np.pi*x[i])
    return X_sin

X = mat_X_exo1(x)
Y = mat_Y_exo1(y1)
X_sin = mat_X_sin(x)
Y_sin = mat_Y_exo1(y_sin)

def calcul_beta(X, Y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)

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
plt.xlabel('theta')
plt.ylabel('f(theta)')
plt.title('Données observées + modèle')
plt.show()

# ---------- Exercice 3 ----------

n = 100

def y_exo3(n, x, eps):
    y = []
    for i in range(n):
        yi = 10 + 2*x[i] 
        y.append(yi)
    return y

y3 = y_exo3(n, x, 0)

traces(x, y3, 'Données observées pour une distribution aléatoire du bruit', 'b.', 'x', 'y')

y_out = np.copy(y3)

n_outliers = 10

for o in range(n_outliers):
    i_random = random.randint(0,n-1)
    y_out[i_random] = np.random.normal(11, 0.5) # en gaussian

traces(x, y_out, 'Données observées pour une distribution aléatoire du bruit', 'b.', 'x', 'y')

def calcul_model(n):
    eps = eps_aleatoire(n)

    y_m = y_exo3(n, x, eps)

    beta2_m = beta2(variance(x,n),covariance(x,y_m,n))
    beta1_m = beta1(x, y_m, beta2_m)

    m = [beta1_m,beta2_m]

    return m

m1 = calcul_model(n)
m2 = calcul_model(n)

def inliers(m1,m2, Treshold):

    for _ in range(10):

        beta1_m1 = m1[0]
        beta2_m1 = m1[1]
        beta1_m2 = m2[0]
        beta2_m2 = m2[1]

        n_m1 = 0
        n_m2 = 0

        for i in range(n):
            if abs(beta1_m1*x[i] + beta2_m1 - y3[i]) < Treshold:
                n_m1 += 1
            
            if abs(beta1_m2*x[i] + beta2_m2 - y3[i]) < Treshold:
                n_m2 += 1

        if n_m1 >= n_m2:
            m2 = calcul_model(n)
            best_model = m1
            
        else:
            m1 = calcul_model(n)
            best_model = m2

    return best_model

def compare(m1,m2, Treshold):

    beta1_m1 = m1[0]
    beta2_m1 = m1[1]
    beta1_m2 = m2[0]
    beta2_m2 = m2[1]

    n_m1 = 0
    n_m2 = 0

    for i in range(n):
        if abs(beta1_m1*x[i] + beta2_m1 - y3[i]) < Treshold:
            n_m1 += 1

        if abs(beta1_m2*x[i] + beta2_m2 - y3[i]) < Treshold:
            n_m2 += 1

    if n_m1 >= n_m2:
        m2 = calcul_model(n)
        return "TP1"
            
    else:
        m1 = calcul_model(n)
        return "TP2"

m_TP1 = [10.037398645699664, 1.9031078680985294]
m_TP2 = inliers(m1, m2, 1)

print("La meilleure stratégie est : ", compare(m_TP1, m_TP2, 1))