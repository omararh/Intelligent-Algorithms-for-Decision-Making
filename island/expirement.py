from islandsManager import IslandsManager
from settings import Config


def run_experiment():
    """
    Exécute une expérience complète.
    """
    manager = IslandsManager()  # Utilise un nom différent pour l'instance
    # Initialisation
    for island in manager.islands:
        for ind in island.population:
            ind.current_island = island.id
            ind.origin = island.id

    # Historiques
    population_history = {island.name: [] for island in manager.islands}
    best_fitness_history = []

    # Boucle principale
    for generation in range(Config.MAX_ITERATIONS):
        # Arrêt si solution optimale trouvée
        if manager.best_fitness == Config.VECTOR_SIZE:
            break

        manager.run_one_generation()

        # Sauvegarde des statistiques
        best_fitness_history.append(manager.best_fitness)
        for island in manager.islands:
            population_history[island.name].append(len(island.population))

        # Complétion de l'historique si solution trouvée
        if manager.best_fitness == Config.VECTOR_SIZE:
            remaining_gens = range(generation + 1, Config.MAX_ITERATIONS)
            for _ in remaining_gens:
                best_fitness_history.append(manager.best_fitness)
                for island in manager.islands:
                    population_history[island.name].append(len(island.population))
            break

    return best_fitness_history, population_history, IslandsManager
