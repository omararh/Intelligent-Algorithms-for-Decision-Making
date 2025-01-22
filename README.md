### Requis

`python version` : 3.10.0

### Fichiers

`oneMaxSteadyState.py` : permet d'effectuer différentes expérimentations, où l'on peut choisir d'étudier soit les effets
de la mutation, soit l'impact de la taille de la population, soit l'influence du croisement (crossover), soit les
mécanismes de sélection sur l'évolution des solutions. La fonction principale main() initialise une configuration et un
objet d'expérimentation, puis appelle la méthode correspondant à l'expérience souhaitée, avec actuellement la mutation
activée et les autres types d'expériences en commentaires.

`OneMaxEstimateDistrib.py` : implémente une expérimentation sur un algorithme à estimation de distribution, où
différentes configurations sont testées en faisant varier le nombre d'individus "k-best" (2, 4, 8, 10 et 14) qui sont
sélectionnés comme parents à chaque génération.

`OneMaxCompactAlgo.py` : présente une expérimentation sur l'algorithme génétique compact (cGA) où l'on teste deux
configurations différentes du taux d'apprentissage α : une première avec α = 1 sur 7000 générations, et une seconde avec
α = 2 sur 3500 générations pour comparer leur impact sur la convergence.

`OneMaxAptativeRoulette.py`: présente une expérimentation sur la roulette adaptative qui compare l'efficacité de
différents opérateurs de mutation (1-flip, 3-flip, 5-flip, bit-flip) dans un algorithme génétique, où la probabilité de
sélection de chaque opérateur est ajustée dynamiquement selon leur performance. Plus spécifiquement, deux configurations
sont testées et visualisées : une première qui compare les taux d'utilisation des différents opérateurs au fil des
générations, et une seconde qui compare les performances entre une roulette adaptative et une roulette fixe utilisant
uniquement le bit-flip, ici on peut aussi expérimenter le masque ansi que le probléme leading ones (voir le main ou il
faut
suivre les commentaires pour lancer ces expérimentations).

`OneMaxUcb.py` : présente une expérimentation sur l'algorithme UCB (Upper Confidence Bound) appliqué à la sélection
adaptative des opérateurs de mutation dans un algorithme génétique, avec plusieurs tests : une première comparaison
entre différents opérateurs de mutation (bit-flip, 1-flip, 3-flips, 5-flips) où l'on observe leur taux d'utilisation,
puis un test avec un opérateur inutile (identité) pour valider l'efficacité de la méthode, et enfin des tests sur des
variantes du problème OneMax (avec masque) et sur le problème LeadingOnes pour évaluer la robustesse de l'approche (
toujours suivre
ce qui a comme commentaires dans le main pour lancer ces expérimentations).