from dataclasses import dataclass
from typing import List, Tuple
from deap import base, creator, tools
import random
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class CompactGAConfig:
    """Configuration de l'algorithme génétique compact"""
    individual_length: int = 300  # Longueur des individus
    population_size: int = 2  # Taille de la population (toujours 2 pour cGA)
    max_generations: int = 3800  # Nombre maximum de générations
    nb_runs: int = 3  # Nombre d'exécutions
    learning_rate: int = 2  # Taux d'apprentissage (alpha)
    min_probability: float = 0.01  # Probabilité minimum pour éviter la convergence à 0


class CompactGeneticAlgorithm:
    """Implémentation de l'algorithme génétique compact"""

    def __init__(self, config: CompactGAConfig):
        self.config = config
        self.toolbox = self._setup_toolbox()
        random.seed()

    def _setup_toolbox(self) -> base.Toolbox:
        """Configuration initiale de la boîte à outils DEAP"""
        toolbox = base.Toolbox()

        # Configuration de base
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Enregistrement des opérations
        toolbox.register("zeroOrOne", random.randint, 0, 1)
        toolbox.register("individualCreator",
                         tools.initRepeat,
                         creator.Individual,
                         toolbox.zeroOrOne,
                         self.config.individual_length)
        toolbox.register("populationCreator",
                         tools.initRepeat,
                         list,
                         toolbox.individualCreator)
        toolbox.register("evaluate", self._evaluate_individual)

        return toolbox

    def _evaluate_individual(self, individual: List[int]) -> Tuple[int]:
        """Calcule la fitness d'un individu (OneMax)"""
        return sum(individual),

    def _initialize_probability_vector(self) -> List[float]:
        """Initialise le vecteur de probabilité à 0.5"""
        return [0.5] * self.config.individual_length

    def _generate_population(self, probability_vector: List[float]) -> List:
        """Génère une population selon le vecteur de probabilité"""
        population = self.toolbox.populationCreator(n=self.config.population_size)

        for individual in population:
            for pos in range(self.config.individual_length):
                individual[pos] = 1 if random.random() < probability_vector[pos] else 0

        return population

    def _update_probability_vector(self, population: List, probability_vector: List[float]) -> List[float]:
        """Met à jour le vecteur de probabilité basé sur le winner et le loser"""
        winner = tools.selBest(population, 1)[0]
        loser = tools.selWorst(population, 1)[0]

        for pos in range(self.config.individual_length):
            if winner[pos] != loser[pos]:
                update = (1 / self.config.individual_length) * self.config.learning_rate
                if winner[pos] == 1:
                    probability_vector[pos] = min(1, probability_vector[pos] + update)
                else:
                    probability_vector[pos] = max(self.config.min_probability,
                                                  probability_vector[pos] - update)

        return probability_vector

    def _run_single_iteration(self) -> Tuple[List[float], List[float]]:
        """Exécute une itération complète de l'algorithme"""
        max_fitness_values = []
        mean_fitness_values = []
        probability_vector = self._initialize_probability_vector()

        for _ in range(self.config.max_generations):
            # Génération et évaluation de la population
            population = self._generate_population(probability_vector)
            for individual in population:
                individual.fitness.values = self.toolbox.evaluate(individual)

            # Mise à jour du vecteur de probabilité
            probability_vector = self._update_probability_vector(population, probability_vector)

            # Collecte des statistiques
            fitness_values = [ind.fitness.values[0] for ind in population]
            max_fitness_values.append(max(fitness_values))
            mean_fitness_values.append(sum(fitness_values) / len(population))

        return max_fitness_values, mean_fitness_values

    def run(self) -> Tuple[List[float], List[float]]:
        """Exécute plusieurs runs de l'algorithme et retourne les moyennes"""
        all_max_fitness = []
        all_mean_fitness = []

        for _ in range(self.config.nb_runs):
            max_values, mean_values = self._run_single_iteration()
            all_max_fitness.append(max_values)
            all_mean_fitness.append(mean_values)

        # Calcul des moyennes sur tous les runs
        avg_max = [sum(run[gen] for run in all_max_fitness) / self.config.nb_runs
                   for gen in range(self.config.max_generations)]
        avg_mean = [sum(run[gen] for run in all_mean_fitness) / self.config.nb_runs
                    for gen in range(self.config.max_generations)]

        return avg_max, avg_mean


class ResultVisualizer:
    """Classe pour la visualisation des résultats"""

    @staticmethod
    def plot_fitness_evolution(max_values: List[float], mean_values: List[float],
                               nb_runs: int, title: str = None):
        """Affiche l'évolution de la fitness"""
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))

        plt.plot(max_values, color='blue', label='Fitness max')
        plt.plot(mean_values, color='orange', label='Fitness moyenne')

        plt.legend()
        plt.xlabel('Génération')
        plt.ylabel('Fitness')
        if title:
            plt.title(title)
        else:
            plt.title(f'Évolution de la fitness sur {nb_runs} runs')
        plt.show()


def main():
    # Configuration
    config = CompactGAConfig()

    # Initialisation et exécution
    algorithm = CompactGeneticAlgorithm(config)
    max_fitness, mean_fitness = algorithm.run()

    # Visualisation
    visualizer = ResultVisualizer()
    visualizer.plot_fitness_evolution(
        max_fitness,
        mean_fitness,
        config.nb_runs,
        f'Fitness moyenne et max en fonction des générations sur {config.nb_runs} runs'
    )


if __name__ == '__main__':
    main()
