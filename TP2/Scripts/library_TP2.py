import numpy as np
# from scipy import optimize
import matplotlib.pyplot as plt
import random

def eps_aleatoire(n):
    liste_eps = []
    for i in range(n):
        eps = random.uniform(-1,1)
        liste_eps.append(eps)
    return liste_eps

def esp_aleatoire_gaussian(n, mu, sigma):
    liste_eps = []
    for i in range(n):
        liste_eps.append(np.random.normal(mu,sigma))
    
    return liste_eps

def y_exo1_3(n, x, eps):
    y = []
    for i in range(n):
        yi = 10 + 2*x[i] + eps[i] # modèle linéaire
        y.append(yi)
    return y

def y_exo2(n, x, eps):
    y = []
    for i in range(n):
        yi = 10 + 2*(x[i]**2) + eps[i] # modèle non-linéaire
        y.append(yi)
    return y

def traces(x, y, title, color, label_x, label_y):
    plt.figure(figsize = (10,8))
    plt.plot(x, y, color)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    return plt.show()

def model(x, y, y_model, title, color, color_model, label_x, label_y):
    plt.figure(figsize = (10,8))
    plt.plot(x, y, color)
    plt.plot(x,y_model, color_model)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    return plt.show()


def variance(x,n):
    sigma2_x = 0
    for i in range(1,n):
        sigma2_x += (x[i] - np.mean(x))**2
    return (1/n)*sigma2_x

def covariance(x, y, n):
    Cxy = 0
    for i in range(1,n):
        Cxy += np.dot((x[i] - np.mean(x)),(y[i] - np.mean(y)))
    Cxy = (1/n)*Cxy
    return Cxy
   
def beta2(var, cov):
    return cov / var

def beta1(x, y, beta2):
    return np.mean(y) - beta2*np.mean(x)