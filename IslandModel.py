import networkx as nx  # Pour la représentation du graphe
from dataclasses import dataclass
import numpy as np
from oneMaxSteadyState import GeneticAlgorithm, GeneticConfig, MutationType, OperatorConfig
import matplotlib.pyplot as plt


@dataclass
class IslandModelConfig:
    genetic_config: GeneticConfig
    migration_rate: float = 0.2  # Taux de migration de base
    alpha: float = 0.8  # Paramètre d'inertie
    beta: float = 0.1  # Paramètre de bruit


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
        self._setup_complete_topology()

        # Matrice de migration (M)
        self.migration_matrix = np.zeros((len(MutationType), len(MutationType)))
        self._initialize_migration_matrix()

        # Matrice de données D pour stocker les améliorations
        self.data_matrix = np.zeros((len(MutationType), len(MutationType)))

        # Initialisation des populations
        self.populations = {op: [] for op in MutationType}
        self._initialize_populations()

    def _setup_complete_topology(self):
        """Crée une topologie complète entre les îles"""
        for op1 in MutationType:
            for op2 in MutationType:
                if op1 != op2:
                    self.topology.add_edge(op1, op2)

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

    def _update_migration_matrix(self):
        """Met à jour la matrice de migration en fonction des améliorations observées."""

        # Mise à jour de la matrice de données D
        for i, op1 in enumerate(MutationType):
            for j, op2 in enumerate(MutationType):
                if op1 != op2:
                    improvements = []
                    for ind in self.populations[op1]:
                        if ind.fitness.values:
                            original_fitness = ind.fitness.values[0]
                            # Simuler le traitement sur l'île j
                            new_fitness = self.islands[op2].toolbox.evaluate(ind)
                            improvements.append(new_fitness[0] - original_fitness)
                    if improvements:
                        self.data_matrix[i][j] = np.mean(improvements)

        # Mise à jour de la matrice de migration M
        for i, op1 in enumerate(MutationType):
            # Calcul du vecteur de récompense R
            R = np.zeros(len(MutationType))
            best_islands = np.argwhere(self.data_matrix[i] == np.max(self.data_matrix[i])).flatten()
            for k in best_islands:
                R[k] = 1.0 / len(best_islands)

            # Mise à jour de la ligne i de la matrice de migration M
            for j, op2 in enumerate(MutationType):
                # ui,t+1 = (1 - α)ui,t + αrt
                self.migration_matrix[i][j] = (1 - self.config.beta) * (
                        self.config.alpha * self.migration_matrix[i][j] + (1 - self.config.alpha) * R[j]
                ) + self.config.beta * np.random.random()

        # Normalisation de la matrice de migration pour que chaque ligne somme à 1
        self.migration_matrix = self.migration_matrix / self.migration_matrix.sum(axis=1, keepdims=True)

    def _analyze_improvements(self):
        """Évalue les améliorations obtenues par les individus après migration."""
        for i, op1 in enumerate(MutationType):
            for j, op2 in enumerate(MutationType):
                if op1 != op2:
                    improvements = []
                    for ind in self.populations[op1]:
                        if ind.fitness.values:
                            original_fitness = ind.fitness.values[0]
                            # Simuler le traitement sur l'île j
                            new_fitness = self.islands[op2].toolbox.evaluate(ind)
                            improvements.append(new_fitness[0] - original_fitness)
                    if improvements:
                        self.data_matrix[i][j] = np.mean(improvements)

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
                    color=None,
                    mutation_type=op
                )
                # Évolution
                offspring = self.islands[op]._select_and_modify_offspring(
                    self.populations[op], operator_config)
                self.populations[op] = self.islands[op]._insert_best_fitness(
                    self.populations[op], offspring)

        # Analyse des améliorations après évolution
        self._analyze_improvements()

        # Mise à jour de la matrice de migration
        self._update_migration_matrix()

        # Gestion des migrations
        self._handle_migrations()

    def _handle_migrations(self):
        """Gère les migrations entre les îles selon la matrice M"""
        # Initialisation du dictionnaire pour toutes les îles
        migrations = {op: [] for op in MutationType}

        # Traitement des migrations
        for i, op1 in enumerate(MutationType):
            for ind in self.populations[op1][:]:  # Copie de la liste
                # Sélection de l'île de destination en fonction des probabilités de migration
                destination_island = np.random.choice(
                    list(MutationType),
                    p=self.migration_matrix[i]
                )

                # Migration de l'individu vers l'île de destination
                if destination_island != op1:  # On ne migre pas vers la même île
                    migrations[destination_island].append(ind)
                    self.populations[op1].remove(ind)

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
        fig, ax = plt.subplots(figsize=(15, 8))

        for i, op in enumerate(MutationType):
            data = history[op]
            fitness_values = [d[0] for d in data]
            ax.plot(fitness_values, label=f'Île {op.value}')

        ax.set_xlabel('Génération')
        ax.set_ylabel('Fitness moyenne')
        ax.set_title('Évolution de la fitness')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_migration_matrix(self):
        """Visualise la matrice de migration."""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.migration_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Probabilité de migration')
        plt.xticks(range(len(MutationType)), [op.value for op in MutationType], rotation=45)
        plt.yticks(range(len(MutationType)), [op.value for op in MutationType])
        plt.title('Matrice de migration dynamique')
        plt.xlabel('Île de destination')
        plt.ylabel('Île de départ')
        plt.tight_layout()
        plt.show()


def main():
    genetic_config = GeneticConfig(
        one_max_length=400,
        population_size=100,
        max_generations=7000,
        p_crossover=1.0,
        p_mutation=1.0,
        tournament_size=3
    )

    island_config = IslandModelConfig(
        genetic_config=genetic_config,
        migration_rate=0.2,
        alpha=0.8,
        beta=0.1
    )

    model = IslandModel(island_config)
    history = model.run()
    model.plot_results(history)
    model.plot_migration_matrix()  # Visualisation de la matrice de migration


if __name__ == "__main__":
    main()
