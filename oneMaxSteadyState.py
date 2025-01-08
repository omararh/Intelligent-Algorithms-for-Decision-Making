from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from enum import Enum


class MutationType(Enum):
    BIT_FLIP = "bit_flip"
    ONE_FLIP = "one_flip"
    THREE_FLIPS = "three_flips"
    FIVE_FLIPS = "five_flips"


class CrossoverType(Enum):
    ONE_POINT = "one_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"


@dataclass
class GeneticConfig:
    """Configuration pour l'algorithme génétique"""
    one_max_length: int = 300
    population_size: int = 20
    p_crossover: float = 1.0
    p_mutation: float = 1.0
    max_generations: int = 1700
    nb_runs: int = 30
    tournament_size: int = 3


@dataclass
class OperatorConfig:
    """Configuration pour un opérateur génétique"""
    name: str
    color: str
    population_size: Optional[int] = None
    mutation_type: Optional[MutationType] = None
    crossover_type: Optional[CrossoverType] = None
    selection_function: Optional[Callable] = None
    values: List[float] = None

    def __post_init__(self):
        if self.values is None:
            self.values = []


class GeneticAlgorithm:
    def __init__(self, config: GeneticConfig):
        self.config = config
        self.toolbox = self._setup_toolbox()
        random.seed()

    def _setup_toolbox(self) -> base.Toolbox:
        """Initialisation de la boîte à outils DEAP"""
        toolbox = base.Toolbox()

        # Création des types de base
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Enregistrement des opérations de base
        toolbox.register("zeroOrOne", random.randint, 0, 1)
        toolbox.register("individualCreator",
                         tools.initRepeat,
                         creator.Individual,
                         toolbox.zeroOrOne,
                         self.config.one_max_length)
        toolbox.register("populationCreator",
                         tools.initRepeat,
                         list,
                         toolbox.individualCreator)

        # Enregistrement des opérateurs génétiques
        toolbox.register("evaluate", self._one_max_fitness)
        toolbox.register("select",
                         tools.selTournament,
                         tournsize=self.config.tournament_size)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate",
                         tools.mutFlipBit,
                         indpb=1.0 / self.config.one_max_length)
        toolbox.register("worst", tools.selWorst, fit_attr='fitness')

        return toolbox

    @staticmethod
    def _one_max_fitness(individual: List[int]) -> Tuple[float]:
        """Calcul de la fitness pour le problème OneMax"""
        return sum(individual),

    ### Opérateurs de mutations ###
    def _flip_bit(self, bit: int) -> int:
        """Inverse la valeur d'un bit"""
        return 1 - bit

    def _mutation_one_flip(self, individual: List[int]) -> None:
        """Mutation par inversion d'un seul bit"""
        pos = random.randint(0, self.config.one_max_length - 1)
        individual[pos] = self._flip_bit(individual[pos])

    def _mutation_n_flips(self, individual: List[int], n: int) -> None:
        """Mutation par inversion de n bits"""
        positions = random.sample(range(self.config.one_max_length), n)
        for pos in positions:
            individual[pos] = self._flip_bit(individual[pos])

    ### Opérateurs de croissement ###

    # mate in one point
    def cxOnePoint(self, ind1, ind2):
        size = min(len(ind1), len(ind2))
        cxpoint = random.randint(1, size - 1)
        ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
        return ind1, ind2

    # mate in two points
    def cxTwoPoint(self, ind1, ind2):
        size = min(len(ind1), len(ind2))
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
        return ind1, ind2

    def cxUniform(self, ind1, ind2, indpb=0.5):
        size = min(len(ind1), len(ind2))
        for i in range(size):
            if random.random() < indpb:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2

    def _insert_best_fitness(self, population: List, offspring: List) -> List:
        """Insertion des meilleurs individus dans la population"""
        worst = self.toolbox.worst(population, 1)
        for ind in offspring:
            if ind.fitness.values[0] > worst[0].fitness.values[0]:
                population.remove(worst[0])
                population.append(ind)
                worst = self.toolbox.worst(population, 1)
        return population

    def run_steady_state(self, operators: List[OperatorConfig]) -> None:
        """Exécution de l'algorithme génétique en mode steady-state"""
        for operator in operators:
            max_fitness_history = []
            mean_fitness_history = []

            for _ in range(self.config.nb_runs):
                population_size = operator.population_size or self.config.population_size
                population = self.toolbox.populationCreator(n=population_size)

                # Évaluation initiale
                fitnesses = map(self.toolbox.evaluate, population)
                for ind, fit in zip(population, fitnesses):
                    ind.fitness.values = fit

                # Évolution
                max_fitness_values, mean_fitness_values = self._evolve_population(
                    population, operator)

                max_fitness_history.append(max_fitness_values)
                mean_fitness_history.append(mean_fitness_values)

            # Calcul des moyennes
            operator.values = self._calculate_mean_values(
                max_fitness_history, mean_fitness_history)

    def _evolve_population(self, population: List, operator: OperatorConfig) -> Tuple[List, List]:
        """Évolution de la population sur plusieurs générations"""
        max_fitness_values = []
        mean_fitness_values = []

        for _ in range(self.config.max_generations):
            offspring = self._select_and_modify_offspring(population, operator)
            population = self._insert_best_fitness(population, offspring)

            # Collecte des statistiques
            fitness_values = [ind.fitness.values[0] for ind in population]
            max_fitness_values.append(max(fitness_values))
            mean_fitness_values.append(sum(fitness_values) / len(population))

        return max_fitness_values, mean_fitness_values

    def _get_mutation_function(self, mutation_type: MutationType) -> Callable:
        """Retourne la fonction de mutation appropriée selon le type"""
        mutation_functions = {
            MutationType.BIT_FLIP: self._mutation_bit_flip,
            MutationType.ONE_FLIP: self._mutation_one_flip,
            MutationType.THREE_FLIPS: lambda ind: self._mutation_n_flips(ind, 3),
            MutationType.FIVE_FLIPS: lambda ind: self._mutation_n_flips(ind, 5)
        }
        return mutation_functions.get(mutation_type, self._mutation_bit_flip)

    def _get_crossover_function(self, crossover_type: CrossoverType) -> Callable:
        """Retourne la fonction de croisement appropriée selon le type"""
        crossover_functions = {
            CrossoverType.ONE_POINT: self.cxOnePoint,
            CrossoverType.TWO_POINT: self.cxTwoPoint,
            CrossoverType.UNIFORM: self.cxUniform
        }
        return crossover_functions.get(crossover_type, self.cxUniform)

    def _mutation_bit_flip(self, individual: List[int]) -> None:
        """Mutation bit-flip standard"""
        for i in range(len(individual)):
            if random.random() < 1.0 / self.config.one_max_length:
                individual[i] = self._flip_bit(individual[i])

    def _select_and_modify_offspring(self, population: List, operator: OperatorConfig) -> List:
        """Sélection et modification des descendants"""
        selection_function = operator.selection_function or self.toolbox.select
        offspring = selection_function(population, 2)
        offspring = list(map(self.toolbox.clone, offspring))

        # Application des opérateurs génétiques
        crossover_function = self._get_crossover_function(operator.crossover_type)
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.config.p_crossover:
                crossover_function(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Sélection de la fonction de mutation appropriée
        mutation_function = self._get_mutation_function(operator.mutation_type)
        for mutant in offspring:
            if random.random() < self.config.p_mutation:
                mutation_function(mutant)
                del mutant.fitness.values

        # Réévaluation des individus modifiés
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        return offspring

    def _calculate_mean_values(self, max_history: List[List], mean_history: List[List]) -> List[float]:
        """Calcul des valeurs moyennes sur plusieurs runs"""
        mean_values = []
        for gen in range(self.config.max_generations):
            gen_max_sum = sum(run[gen] for run in max_history)
            mean_values.append(gen_max_sum / self.config.nb_runs)
        return mean_values

    def plot_results(self, operators: List[OperatorConfig], experiment_title: str = "") -> None:
        """Affichage des résultats"""
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))

        for operator in operators:
            plt.plot(operator.values,
                     color=operator.color,
                     label=operator.name)

        plt.legend()
        plt.xlabel('Génération')
        plt.ylabel('Fitness')
        title = f'{experiment_title}\nFitness moyenne sur {self.config.nb_runs} runs' if experiment_title else f'Fitness moyenne sur {self.config.nb_runs} runs'
        plt.title(title)
        plt.show()


class ExperimentConfig:
    """Classe pour gérer les différentes configurations d'expériences"""

    @staticmethod
    def get_population_config() -> List[OperatorConfig]:
        """Configuration pour les tests de taille de population"""
        sizes = [20, 30, 40, 50]
        colors = ['blue', 'red', 'green', 'black']

        return [
            OperatorConfig(
                name=f"{size} individus",
                color=color,
                population_size=size
            )
            for size, color in zip(sizes, colors)
        ]

    @staticmethod
    def get_mutation_config() -> List[OperatorConfig]:
        """Configuration pour les tests de mutation"""
        mutation_configs = [
            (MutationType.BIT_FLIP, "Bit Flip", "blue"),
            (MutationType.ONE_FLIP, "Un Flip", "red"),
            (MutationType.THREE_FLIPS, "Trois Flips", "green"),
            (MutationType.FIVE_FLIPS, "Cinq Flips", "black")
        ]

        return [
            OperatorConfig(
                name=name,
                color=color,
                mutation_type=mutation_type
            )
            for mutation_type, name, color in mutation_configs
        ]

    @staticmethod
    def get_crossover_config() -> List[OperatorConfig]:
        """Configuration pour les tests de croisement"""
        crossover_configs = [
            (CrossoverType.ONE_POINT, "Un Point", "blue"),
            (CrossoverType.TWO_POINT, "Deux Points", "red"),
            (CrossoverType.UNIFORM, "Uniforme", "green")
        ]

        return [
            OperatorConfig(
                name=name,
                color=color,
                crossover_type=crossover_type
            )
            for crossover_type, name, color in crossover_configs
        ]


class GeneticExperiment:
    """Classe pour gérer les expériences avec l'algorithme génétique"""

    def __init__(self, config: GeneticConfig):
        self.ga = GeneticAlgorithm(config)
        self.experiment_config = ExperimentConfig()

    def run_population_experiment(self):
        """Exécute l'expérience sur les tailles de population"""
        operators = self.experiment_config.get_population_config()
        self._run_experiment(operators, "Comparaison des tailles de population")

    def run_mutation_experiment(self):
        """Exécute l'expérience sur les opérateurs de mutation"""
        operators = self.experiment_config.get_mutation_config()
        self._run_experiment(operators, "Comparaison des opérateurs de mutation")

    def run_crossover_experiment(self):
        """Exécute l'expérience sur les opérateurs de croisement"""
        operators = self.experiment_config.get_crossover_config()
        self._run_experiment(operators, "Comparaison des opérateurs de croisement")

    def _run_experiment(self, operators: List[OperatorConfig], title: str):
        """Exécute une expérience avec les opérateurs donnés"""
        self.ga.run_steady_state(operators)
        self.ga.plot_results(operators, title)


def main():
    """Fonction principale"""
    config = GeneticConfig()
    experiment = GeneticExperiment(config)

    # Choisir l'expérience à exécuter
    # experiment.run_mutation_experiment()
    # Autre experiences : croissement, population ou selection ?
    # experiment.run_population_experiment()
    experiment.run_crossover_experiment()


if __name__ == '__main__':
    main()
