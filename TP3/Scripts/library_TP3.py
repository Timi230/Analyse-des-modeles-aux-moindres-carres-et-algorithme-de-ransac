import numpy as np
import matplotlib.pyplot as plt
import random


def TP1_y_exo1_3(n, x, eps):
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


def y_exo1(n, x, eps):
    """Permet de faire le calule de y dans l'exercice 1 et 3 avec : 
            y = 10 + 2*x + 8*x^2 + eps

    Args:
        n (int): nombre d échantillons observés
        x (NDArray[floating[Any]]): l espace d'étude
        eps (list): liste des valeurs du bruits 

    Returns:
        list: valeur de y
    """   
    y = []
    for i in range(n):
        yi = 10 + 2*x[i] + 8*(x[i]**2) + eps[i] # modèle linéaire
        y.append(yi)
    return y

def y_exo1_sin(n, x, eps):
    """Permet de faire le calule de y dans l'exercice 1 et 3 avec : 
            y = 10 + 2*x + 8*x^2 + 10*sin(pi*x) + eps

    Args:
        n (int): nombre d échantillons observés
        x (NDArray[floating[Any]]): l espace d'étude
        eps (list): liste des valeurs du bruits 

    Returns:
        list: valeur de y
    """
    y = []
    for i in range(n):
        yi = 10 + 2*x[i] + 8*(x[i]**2) + 10*np.sin(np.pi*x[i]) + eps[i] # modèle linéaire
        y.append(yi)
    return y

def y_exo3(n, x):
    """Permet de faire le calule de y dans l'exercice 1 et 3 avec : 
            y = 10 + 2*x 

    Args:
        n (int): nombre d échantillons observés
        x (NDArray[floating[Any]]): l espace d'étude
        eps (list): liste des valeurs du bruits 

    Returns:
        list: valeur de y
    """
    y = []
    for i in range(n):
        yi = 10 + 2*x[i] 
        y.append(yi)
    return y

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


def outliers(y, n, n_outliers, mu, sigma):
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
        i_random = random.randint(0,n-1)
        y[i_random] = np.random.normal(mu,sigma)

    return y


def mat_X_exo1(x, n):
    """Calcul la matrice X de l'ex 1

    Args:
        x (list): valeur de x
        n (int): nombre d échantillons observés 

    Returns:
        NDArray[float64]: matrice X 
    """
    X = np.ones((n,3))
    for i in range(n):
        X[i,1] = x[i]
        X[i,2] = x[i]**2 
    return X

def mat_Y_exo1(y, n):
    """Calcul la matrice Y de l'ex 1

    Args:
        y (list): valeur de y
        n (int): nombre d échantillons observés 

    Returns:
        NDArray[float64]: matrice colone Y
    """
    Y = np.ones((n,1))
    for i in range(n):
        Y[i] = y[i]
    return Y

def mat_X_sin(x, n):
    """Calcul la matrice X de l'ex 1 pour le modèle non-linéaire

    Args:
        x (list): valeur de x
        n (int): nombre d échantillons observés 

    Returns:
        NDArray[float64]: matrice X pour le modèle non-linéaire
    """
    X_sin = np.ones((n,4))
    for i in range(n):
        X_sin[i,1] = x[i]
        X_sin[i,2] = x[i]**2 
        X_sin[i,3] = np.sin(np.pi*x[i])
    return X_sin

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


def calcul_beta(X, Y):
    """Génération de la matrice beta en fonction de X et Y

    Args:
        X (NDArray[float64]): Matrice X
        Y (NDArray[float64]): Matrice Y

    Returns:
        NDArray[float64]: Matrice de beta
    """
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)

def calcul_model(x, y, n):
    """ Estimation des beta_1 et beta_2

    Args:
        x (list): valeur de x
        y (list): valeur de y
        n (int): nombre d échantillons observés 

    Returns:
        list: liste contenant beta_1 et beta_2
    """
    

    beta2_m = beta2(variance(x,n),covariance(x,y,n))
    beta1_m = beta1(x, y, beta2_m)

    m = [beta1_m,beta2_m]

    return m

def inliers(m, Treshold, n, x,y):
    beta1_m = m[0]
    beta2_m = m[1]

    Liste_n_m = []

    for i in range(n):
        if abs(beta1_m + beta2_m*x[i] - y[i]) < Treshold:
            Liste_n_m.append([x[i],y[i]])

    return Liste_n_m


def compare(m1,m2, Treshold, n,x, y):
    """Cette fonction compare 10 fois les deux modèles et nous indique lequel est le meilleur 

    Args:
        m1 (list): premier modèle
        m2 (list): premier modèle
        Treshold (int): _description_
        n (int): nombre d échantillons observés 
        x (list) : valeurs de x
        y (list) : valeurs de y

    Returns:
        string: renvoie le nom du meilleur modèle
    """

    beta1_m1 = m1[0]
    beta2_m1 = m1[1]
    beta1_m2 = m2[0]
    beta2_m2 = m2[1]

    n_m1 = 0
    n_m2 = 0

    for i in range(n):
        if abs(beta1_m1 + beta2_m1*x[i] - y[i]) < Treshold:
            n_m1 += 1

        if abs(beta1_m2 + beta2_m2*x[i] - y[i]) < Treshold:
            n_m2 += 1

    if n_m1 >= n_m2:
        m2 = calcul_model(x,y, n)
        return "TP2"
            
    else:
        m1 = calcul_model(x,y, n)
        return "TP3"

def reestimated(Liste_n_m):
    """Pemrmet de réestime le modèle en identifiant et en gardant que les inliners

    Args:
        Liste_n_m (list): liste de tout les inliners

    Returns:
        list: nouveau modèle
    """

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

def fun(parameters, theta, f_theta):
    """fonction de notre modèle

    Args:
        parameters (list): p, e 
        theta (list): liste des différents angles
        f_theta (list): liste des valeurs observées

    Returns:
        list: la différence pour chaque theta et f_theta
    """    
    return parameters[0] / (1-parameters[1]*np.cos(theta))-f_theta
