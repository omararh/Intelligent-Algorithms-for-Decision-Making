import networkx as nx  # Pour la représentation du graphe
from dataclasses import dataclass
import numpy as np
from oneMaxSteadyState import GeneticAlgorithm, GeneticConfig, MutationType, OperatorConfig
import random
import matplotlib.pyplot as plt


@dataclass
class IslandModelConfig:
    genetic_config: GeneticConfig
    migration_rate: float = 0.2  # Taux de migration de base


class IslandModel:
    def __init__(self, config: IslandModelConfig):
        self.config = config

        # Création du graphe de topologie
        self.topology = nx.Graph()

        # Création des îles (nœuds du graphe)
        self.islands = {}
        for op in MutationType:
            self.islands[op] = GeneticAlgorithm(config.genetic_config)
            self.topology.add_node(op)

        # Création des connexions entre îles (arêtes)
        # Ici on crée un graphe complet, mais on peut changer la topologie
        self._setup_complete_topology()

        # Matrice de migration (M)
        self.migration_matrix = np.zeros((len(MutationType), len(MutationType)))
        self._initialize_migration_matrix()

        # Initialisation des populations
        self.populations = {op: [] for op in MutationType}
        self._initialize_populations()

    def _setup_complete_topology(self):
        """Crée une topologie complète entre les îles"""
        for op1 in MutationType:
            for op2 in MutationType:
                if op1 != op2:
                    self.topology.add_edge(op1, op2)

    def _setup_ring_topology(self):
        """Alternative: Crée une topologie en anneau"""
        operators = list(MutationType)
        for i in range(len(operators)):
            self.topology.add_edge(operators[i], operators[(i + 1) % len(operators)])

    def _initialize_migration_matrix(self):
        """Initialise la matrice de migration selon la topologie"""
        n = len(MutationType)
        base_rate = self.config.migration_rate / (n - 1)

        for i, op1 in enumerate(MutationType):
            for j, op2 in enumerate(MutationType):
                if op1 != op2 and self.topology.has_edge(op1, op2):
                    self.migration_matrix[i][j] = base_rate
                elif op1 == op2:
                    self.migration_matrix[i][j] = 1 - self.config.migration_rate

    def _initialize_populations(self):
        """Initialise les populations de chaque île"""
        size_per_island = self.config.genetic_config.population_size // len(MutationType)
        for op in MutationType:
            population = self.islands[op].toolbox.populationCreator(n=size_per_island)
            # Évaluation initiale
            fitnesses = map(self.islands[op].toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            self.populations[op] = population

    def evolve_populations(self):
        """Fait évoluer chaque île et gère les migrations"""
        for op in MutationType:
            if self.populations[op]:  # Si la population n'est pas vide
                # Évolution avec l'opérateur de l'île
                operator_config = OperatorConfig(
                    name=op.value,
                    color="blue",
                    mutation_type=op
                )
                # Évolution
                offspring = self.islands[op]._select_and_modify_offspring(
                    self.populations[op], operator_config)
                self.populations[op] = self.islands[op]._insert_best_fitness(
                    self.populations[op], offspring)

        # Migration selon la matrice M
        self._handle_migrations()

    def _handle_migrations(self):
        """Gère les migrations entre les îles selon la matrice M"""
        # Initialisation du dictionnaire pour toutes les îles
        migrations = {op: [] for op in MutationType}

        # Traitement des migrations
        for i, op1 in enumerate(MutationType):
            for ind in self.populations[op1][:]:  # Copie de la liste
                for j, op2 in enumerate(MutationType):
                    if self.topology.has_edge(op1, op2):  # Vérifie si la migration est possible
                        if random.random() < self.migration_matrix[i][j]:
                            migrations[op2].append(ind)
                            self.populations[op1].remove(ind)
                            break

        # Application des migrations
        for op, migrants in migrations.items():
            self.populations[op].extend(migrants)

    def get_best_solution(self):
        """Retourne la meilleure solution parmi toutes les îles"""
        best_fitness = float('-inf')
        best_individual = None

        for population in self.populations.values():
            if population:  # Si la population n'est pas vide
                current_best = max(population, key=lambda x: x.fitness.values[0])
                if current_best.fitness.values[0] > best_fitness:
                    best_fitness = current_best.fitness.values[0]
                    best_individual = current_best

        return best_individual

    def run(self, max_generations: int = None):
        """Exécute l'algorithme pour un nombre donné de générations"""
        if max_generations is None:
            max_generations = self.config.genetic_config.max_generations

        history = {op: [] for op in MutationType}

        for gen in range(max_generations):
            self.evolve_populations()

            # Collecte des statistiques
            for op in MutationType:
                if self.populations[op]:
                    avg_fitness = np.mean([ind.fitness.values[0] for ind in self.populations[op]])
                    pop_size = len(self.populations[op])
                else:
                    avg_fitness, pop_size = 0.0, 0
                history[op].append((avg_fitness, pop_size))

            best_sol = self.get_best_solution()
            if best_sol and best_sol.fitness.values[0] == self.config.genetic_config.one_max_length:
                break

        return history

    def plot_results(self, history):
        """Visualisation des résultats"""
        plt.figure(figsize=(12, 5))

        # Plot de la fitness moyenne
        plt.subplot(1, 2, 1)
        for op, data in history.items():
            fitness_values = [d[0] for d in data]
            plt.plot(fitness_values, label=op.value)
        plt.xlabel('Génération')
        plt.ylabel('Fitness moyenne')
        plt.title('Évolution de la fitness par île')
        plt.legend()

        # Plot de la taille des populations
        plt.subplot(1, 2, 2)
        for op, data in history.items():
            pop_sizes = [d[1] for d in data]
            plt.plot(pop_sizes, label=op.value)
        plt.xlabel('Génération')
        plt.ylabel('Taille population')
        plt.title('Évolution des tailles de population')
        plt.legend()

        plt.tight_layout()
        plt.show()


def main():
    genetic_config = GeneticConfig(
        one_max_length=300,
        population_size=100,
        max_generations=7000,
        p_crossover=1.0,
        p_mutation=1.0,
        tournament_size=3
    )

    island_config = IslandModelConfig(
        genetic_config=genetic_config,
        migration_rate=0.2
    )

    model = IslandModel(island_config)
    history = model.run()
    model.plot_results(history)


if __name__ == "__main__":
    main()
