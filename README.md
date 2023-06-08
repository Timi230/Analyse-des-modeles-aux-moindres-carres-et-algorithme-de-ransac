# Analyse des modèles aux moindres carrés et algorithme de ransac



*Intitulé du projet* : 

Analyse des modèles aux moindres carrés et algorithme de ransac

*Type de projet* : 

expérimental, simulation

*Objectifs du projet*  : 

L'objectif de cette étude est de simuler des modèles linéaires et non-linéaires, sans ou avec bruit selon une loi uniforme ou normale. Sera étudié aussi l'effet des outliers puis la réestimation du modèle en ne prenant en compte que les inliers à l'aide de l'agorithme de ransac. Enfin, deux toolbox seront abordées : scipy.optimize.least\_square et numpy.linalg.svd.

*Mots-clefs du projet* :

    - Méthode des moindres carrés
    - Échantillons
    - Outliers / Inliers
    - Algorithme de ransac
    
*Auteurs* :

    - Emma Fillipone
    - Guillaume Faivre
    - Titouan Millet
    - Camille Pieussergues
    
 *Rapport du projet* :
 

#

**Motivation**

Les ingénieurs font couramment appel à des modèles mathématiques afin de décrire une observation ou des jeux de valeurs. L'utilisation de ces modèles permettent de donner des explications, faire des prédictions, des simulations. Dans tous les cas de figure, l'objectif est de comprendre un phénomène ou de le simplifier pour l'étudier. Cependant, il faut que le modèle soit le plus proche de la réalité. Afin de pouvoir l'ajuster au mieux, nous pouvons utiliser des méthodes d'optimisation différentiable afin de trouver les meilleurs coefficients pour notre modèle. 

C'est dans ce cadre que ce projet intervient. À travers ce rapport, nous étudierons différents jeux de données et nous découvrirons toutes les notions d'optimisation de modèle. Ce projet est découpé selon différents travaux qui contiennent eux-même différents exercices. 

Plus précisément, nous allons simuler des modèles linéaires et non-linéaires, sans ou avec bruit selon une loi uniforme ou normale. Sera étudié aussi l'effet des outliers puis la réestimation du modèle en ne prenant en compte que les inliers à l'aide de l'agorithme de ransac. Enfin, deux toolbox seront abordées : scipy.optimize.least\_square et numpy.linalg.svd.

#
**Arborécence de travail**

Le projet se base sur les différents TP que nous avons pu réaliser durant le semestre. Afin de rester rigoureux, nous avons choisit de segmenter notre code selon les TP. Voici notre arborescence : 

    - TP1
        
        - Rapport
              -DOC_library_TP1.html
        - Doc 
              - Sujet_TP1.pdf
        - Scripts 
              - library_TP1.py
              - TP1.py
    - TP2
    - TP3
    
Nous avons donc trois dossiers, un pour chaque TP. Dans chaque dossier, nous retrouvons le dossier Rapport qui contient la documentation sous un format html. Dans le dossier Scripts nous retrouvons la bibliothèque qui contient toutes les déinitions de fonctions utilisées dans le script principale : [TP°].py


