import numpy as np
from collections import defaultdict
from settings import Config
from island_model import Island
from mutations import MutationType


class IslandsManager:
    def __init__(self):
        # Création des îles
        self.islands = [
            Island(0, MutationType.ONE_FLIP, "Ile 1flip"),
            Island(1, MutationType.THREE_FLIPS, "Ile 3flip"),
            Island(2, MutationType.FIVE_FLIPS, "Ile 5flip"),
            Island(3, MutationType.BIT_FLIP, "Ile BitFlip")
        ]

        # Matrices
        self.migration_matrix = np.full((Config.NUM_ISLANDS, Config.NUM_ISLANDS),
                                        1.0 / Config.NUM_ISLANDS)
        self.reward_matrix = np.zeros((Config.NUM_ISLANDS, Config.NUM_ISLANDS))

        # Suivi des performances
        self.best_fitness = 0
        self.best_individual = None

    def handle_migrations(self) -> None:
        """Gère les migrations entre les îles."""
        for source_island in self.islands:
            probabilities = self.migration_matrix[source_island.id]
            for ind in source_island.population[:]:
                if not ind.migrated:
                    destination = np.random.choice(Config.NUM_ISLANDS, p=probabilities)

                    if destination != source_island.id:
                        ind.origin = source_island.id
                        ind.current_island = destination
                        ind.migrated = True

                        source_island.population.remove(ind)
                        self.islands[destination].population.append(ind)

        # Réinitialisation des flags de migration
        for island in self.islands:
            for ind in island.population:
                ind.migrated = False

    def update_migration_matrix(self) -> None:
        """Met à jour la matrice de migration."""
        migrants_by_origin_dest = defaultdict(list)

        # Collecte des migrants par origine/destination
        for island in self.islands:
            for ind in island.population:
                if ind.origin is not None:
                    migrants_by_origin_dest[(ind.origin, island.id)].append(ind)

        # Calcul des récompenses
        self.reward_matrix.fill(0.0)
        for i_source in range(Config.NUM_ISLANDS):
            dest_improvements = {}
            for i_dest in range(Config.NUM_ISLANDS):
                migrants = migrants_by_origin_dest.get((i_source, i_dest), [])
                mean_upgrade = (sum(ind.upgrade for ind in migrants) / len(migrants)) if migrants else 0.0
                dest_improvements[i_dest] = mean_upgrade

            # Récompense des meilleures destinations
            best_value = max(dest_improvements.values())
            if best_value > 0:
                best_destinations = [d for d, v in dest_improvements.items()
                                     if v == best_value]
                reward = 1.0 / len(best_destinations)
                for dest in best_destinations:
                    self.reward_matrix[i_source][dest] = reward

        # Mise à jour de la matrice de migration
        for i in range(Config.NUM_ISLANDS):
            for j in range(Config.NUM_ISLANDS):
                self.migration_matrix[i][j] = (1 - Config.BETA) * (
                        Config.ALPHA * self.migration_matrix[i][j] +
                        (1 - Config.ALPHA) * self.reward_matrix[i][j]
                ) + Config.BETA * Config.NOISE

            # Normalisation
            row_sum = self.migration_matrix[i].sum()
            if row_sum > 0:
                self.migration_matrix[i] /= row_sum
            else:
                self.migration_matrix[i].fill(1.0 / Config.NUM_ISLANDS)

    def update_best_solution(self) -> None:
        """Met à jour la meilleure solution trouvée."""
        current_best = max(
            (island.get_best_element() for island in self.islands),
            key=lambda x: x.get_fitness() if x else 0
        )
        if current_best and current_best.get_fitness() > self.best_fitness:
            self.best_fitness = current_best.get_fitness()
            self.best_individual = current_best.clone()

    def run_one_generation(self) -> None:
        """Exécute une génération complète."""
        self.handle_migrations()

        for island in self.islands:
            island.local_search()

        self.update_best_solution()
        self.update_migration_matrix()
