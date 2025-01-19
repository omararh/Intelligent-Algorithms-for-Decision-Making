from dataclasses import dataclass
from typing import List, Tuple
from deap import base, creator, tools
import random
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class AlgorithmConfig:
    """Configuration de l'algorithme"""
    individual_length: int = 300
    population_size: int = 30
    max_generations: int = 60
    nb_runs: int = 30
    min_probability: float = 0.01  # Probabilité minimum pour éviter la convergence vers 0


@dataclass
class ExperimentConfig:
    """Configuration pour une expérience"""
    nb_selected: int  # k best a selectionner
    name: str
    color: str
    values: List[float] = None

    def __post_init__(self):
        if self.values is None:
            self.values = []


class DistributionEstimator:
    """Classe principale pour l'algorithme d'estimation de distribution"""

    def __init__(self, config: AlgorithmConfig):
        self.config = config
        self.toolbox = self._setup_toolbox()
        random.seed()

    def _setup_toolbox(self) -> base.Toolbox:
        """Initialise la boîte à outils DEAP"""
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

    @staticmethod
    def _evaluate_individual(individual: List[int]) -> Tuple[int]:
        """Évalue un individu avec la fonction OneMax"""
        return sum(individual),

    def generate_initial_distribution(self) -> List[float]:
        """Génère le vecteur de probabilité initial"""
        return [random.random() for _ in range(self.config.individual_length)]

    def generate_population_from_distribution(self, population: List, distribution: List[float]):
        """Génère une population à partir d'une distribution de probabilité"""
        for individual in population:
            for pos in range(self.config.individual_length):
                individual[pos] = 1 if random.random() < distribution[pos] else 0

    def update_distribution(self, population: List, distribution: List[float], k: int):
        """Met à jour la distribution basée sur les k meilleurs individus"""
        best_individuals = tools.selBest(population, k)

        for pos in range(self.config.individual_length):
            # Calcule la moyenne des bits à cette position
            bit_sum = sum(ind[pos] for ind in best_individuals)
            distribution[pos] = bit_sum / k

            # Évite la convergence vers 0
            if distribution[pos] < self.config.min_probability:
                distribution[pos] = self.config.min_probability

    def run_experiment(self, experiment: ExperimentConfig) -> Tuple[List[float], List[float]]:
        """Exécute une expérience complète"""
        max_fitness_history = []
        mean_fitness_history = []

        for _ in range(self.config.nb_runs):
            max_values, mean_values = self._run_single_iteration(experiment.nb_selected)
            max_fitness_history.append(max_values)
            mean_fitness_history.append(mean_values)

        # Calcul des moyennes sur tous les runs
        mean_max = self._calculate_mean_over_runs(max_fitness_history)
        mean_mean = self._calculate_mean_over_runs(mean_fitness_history)

        return mean_max, mean_mean

    def _run_single_iteration(self, nb_selected: int) -> Tuple[List[float], List[float]]:
        """Exécute une itération de l'algorithme"""
        max_fitness_values = []
        mean_fitness_values = []
        population = self.toolbox.populationCreator(n=self.config.population_size)
        distribution = self.generate_initial_distribution()

        for _ in range(self.config.max_generations):
            self.generate_population_from_distribution(population, distribution)

            # Évaluation de la population
            fitnesses = map(self.toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Mise à jour de la distribution
            self.update_distribution(population, distribution, nb_selected)

            # Collecte des statistiques
            fitness_values = [ind.fitness.values[0] for ind in population]
            max_fitness_values.append(max(fitness_values))
            mean_fitness_values.append(sum(fitness_values) / len(population))

        return max_fitness_values, mean_fitness_values

    def _calculate_mean_over_runs(self, history: List[List[float]]) -> List[float]:
        """Calcule la moyenne sur tous les runs pour chaque génération"""
        return [sum(run[gen] for run in history) / len(history)
                for gen in range(self.config.max_generations)]


class Visualizer:
    """Classe pour la visualisation des résultats"""

    @staticmethod
    def plot_fitness_evolution(max_values: List[float], mean_values: List[float], nb_runs: int):
        """Affiche l'évolution de la fitness maximale et moyenne"""
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))

        plt.plot(max_values, color='blue', label='Fitness max')
        plt.plot(mean_values, color='orange', label='Fitness moyenne')

        plt.legend()
        plt.xlabel('Génération')
        plt.ylabel('Fitness')
        plt.title(f'Évolution de la fitness sur {nb_runs} runs')
        plt.show()

    @staticmethod
    def plot_comparison(experiments: List[ExperimentConfig]):
        """Affiche la comparaison entre différentes configurations"""
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))

        for exp in experiments:
            plt.plot(exp.values, color=exp.color, label=exp.name)

        plt.legend()
        plt.xlabel('Génération')
        plt.ylabel('Fitness')
        plt.title('Comparaison des configurations')
        plt.show()


def main():
    # Configuration de base
    config = AlgorithmConfig()
    algorithm = DistributionEstimator(config)
    visualizer = Visualizer()

    # Configuration des expériences
    experiments = [
        ExperimentConfig(2, "2 individus", "orange"),
        ExperimentConfig(4, "4 individus", "blue"),
        ExperimentConfig(8, "8 individus", "red"),
        ExperimentConfig(10, "10 individus", "green"),
        ExperimentConfig(14, "14 individus", "black")
    ]

    # Exécution des expériences
    max_fitness = []
    mean_fitness = []

    for exp in experiments:
        max_values, mean_values = algorithm.run_experiment(exp)
        exp.values = mean_values
        if len(max_fitness) == 0:  # Garde seulement les valeurs du premier run
            max_fitness = max_values
            mean_fitness = mean_values

    # Visualisation des résultats
    visualizer.plot_fitness_evolution(max_fitness, mean_fitness, config.nb_runs)
    visualizer.plot_comparison(experiments)


if __name__ == "__main__":
    main()
