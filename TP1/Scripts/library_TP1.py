import numpy as np
import matplotlib.pyplot as plt
import random


def eps_aleatoire(n):
    """ Cette fonction permet de générer le bruit de façon aléatoire de -1 à 1

    Args:
        n (int): nombre d échantillons observés 

    Returns:
        list: valeur des epsilon
    """    

    liste_eps = []
    for i in range(n):
        eps = random.uniform(-1,1)
        liste_eps.append(eps)
    return liste_eps

def esp_aleatoire_gaussian(n, mu, sigma):
    """ Cette fonction permet de générer le bruit de façon aléatoire avec un 
        espilon définit aléatoire gaussienne centrée avec une variance unitaire

    Args:
        n (int): nombre d échantillons observés 
        mu (int): moyenne pour centrer la distribution
        sigma (int): la variance

    Returns:
        list: valeur des epsilon
    """    

    liste_eps = []
    for i in range(n):
        liste_eps.append(np.random.normal(mu,sigma))
    
    return liste_eps

def y_exo1_3(n, x, eps):
    """Permet de faire le calule de y dans l'exercice 1 et 3 avec : 
            y = 10 + 2*x + eps

    Args:
        n (int): nombre d échantillons observés
        x (NDArray[floating[Any]]): l espace d'étude
        eps (list): liste des valeurs du bruits 

    Returns:
        list: valeur de y
    """   

    y = []
    for i in range(n):
        yi = 10 + 2*x[i] + eps[i] # modèle linéaire
        y.append(yi)
    return y

def y_exo2(n, x, eps):
    """Permet de faire le calule de y dans l'exercice 1 et 3 avec : 
            y = 10 + 2*x^2 + eps

    Args:
        n (int): nombre d échantillons observés
        x (NDArray[floating[Any]]): l espace d'étude
        eps (list): liste des valeurs du bruits 

    Returns:
        list: valeur de y
    """  
    y = []
    for i in range(n):
        yi = 10 + 2*(x[i]**2) + eps[i] # modèle non-linéaire
        y.append(yi)
    return y

def traces(x, y, title, color, label_x, label_y):
    """Permet de représenter les donées générés sous forme d'un nuage de points

    Args:
        x (NDArray[floating[Any]]): valeurs de x (axes des ordonnées)
        y (list): valeurs de y (axes des abscices)
        title (string): titre de la figure
        color (string): couleurs des points des données 
        label_x (string): label de l axe des ordonées
        label_y (string): label de l axe des absices

    Returns:
        Any: 
    """    

    plt.figure(figsize = (10,8))
    plt.plot(x, y, color)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    return plt.show()

def model(x, y, y_model, title, color, color_model, label_x, label_y):
    """Permet de représenter les donées générés sous forme d'un nuage de points et 
        de superposer une représentation de notre modèle

    Args:
        x (NDArray[floating[Any]]): valeurs de x (axes des ordonnées)
        y (list): valeurs de y (axes des abscices)
        y_model (list): valeurs de y de notre modèle
        title (_type_): titre de la figure
        color (_type_): couleur des points des données générer
        color_model (_type_): couleur des points du modèle
        label_x (_type_): label de l axe des ordonées
        label_y (_type_): label de l axe des absices

    Returns:
        Any: 
    """    

    plt.figure(figsize = (10,8))
    plt.plot(x, y, color)
    plt.plot(x,y_model, color_model)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    return plt.show()


def variance(x,n):
    """Calcul de la variance d'une variable

    Args:
        x (list): varibale de calcul 
        n (int): nombre d'échantillons observés 

    Returns:
        float: valeur de la variance de notre variable x
    """    

    sigma2_x = 0
    for i in range(1,n):
        sigma2_x += (x[i] - np.mean(x))**2
    return (1/n)*sigma2_x

def covariance(x, y, n):
    """Calcul de la covariance entre deux variables

    Args:
        x (list): première variable de calcul
        y (list): deuxième variable de calcul
        n (int): nombre d'échantillons observés 

    Returns:
        float: valeur de la covariance entre les deux variables
    """    

    Cxy = 0
    for i in range(1,n):
        Cxy += np.dot((x[i] - np.mean(x)),(y[i] - np.mean(y)))
    Cxy = (1/n)*Cxy
    return Cxy
   
def beta2(var, cov):
    """Calcul du coéficient beta2 à partir le variance et de la covariance 

    Args:
        var (float): variance
        cov (float): covariance

    Returns:
        float: beta2
    """    
    return cov / var

def beta1(x, y, beta2):
    """Calcul du coéficient beta1 à partir de x, y et de beta2

    Args:
        x (list): première variable de calcul
        y (list): deuxième variable de calcul
        beta2 (float): valeur de beta2

    Returns:
        float: beta1
    """
    return np.mean(y) - beta2*np.mean(x)

def outliers(y, n, n_outliers, min, max):
    """Permet de géngérer des outiliers, des valeurs qui sont très éloignées de la masse 
        de points générés

    Args:
        y (list): valeur de y
        n (int): nombre d échantillons observés 
        n_outliers (int): nombre de outliers générés
        min (int): valeur minimal de du choix aléatoire de la valeur du outliers
        max (int): valeur maximal de du choix aléatoire de la valeur du outliers

    Returns:
        list: y avec les outliers
    """    
    for o in range(n_outliers):
        i_random = random.randint(0,n)
        y[i_random] = random.uniform(min,max)

    return y