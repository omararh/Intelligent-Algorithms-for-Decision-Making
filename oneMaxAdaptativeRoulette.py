from dataclasses import dataclass
from typing import List, Callable, Tuple
import random
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools


@dataclass
class GeneticAlgorithmConfig:
    """Configuration pour l'algorithme génétique"""
    one_max_length: int = 300
    population_size: int = 20
    p_mutation: float = 1.0
    max_generations: int = 3000
    nb_runs: int = 30
    alpha: float = 0.1
    p_min: float = 0.125


class MutationOperator:
    """Classe gérant les opérateurs de mutation"""

    @staticmethod
    def flip(b: int) -> int:
        """Inverse un bit"""
        return 0 if b == 1 else 1

    @staticmethod
    def bit_flip(individual: List[int]) -> None:
        """Mutation bit-flip classique"""
        for i in range(len(individual)):
            if random.random() < 1 / len(individual):
                individual[i] = MutationOperator.flip(individual[i])

    @staticmethod
    def one_flip(individual: List[int]) -> None:
        """Mutation d'un seul bit"""
        pos = random.randint(0, len(individual) - 1)
        individual[pos] = MutationOperator.flip(individual[pos])

    @staticmethod
    def n_flips(individual: List[int], n: int) -> None:
        """Mutation de n bits"""
        positions = random.sample(range(len(individual)), n)
        for pos in positions:
            individual[pos] = MutationOperator.flip(individual[pos])

    @staticmethod
    def trois_flips(individual: List[int]) -> None:
        """Mutation de 3 bits"""
        MutationOperator.n_flips(individual, 3)

    @staticmethod
    def cinq_flips(individual: List[int]) -> None:
        """Mutation de 5 bits"""
        MutationOperator.n_flips(individual, 5)


class AdaptiveRoulette:
    """Classe gérant la roulette adaptative"""

    def __init__(self, config: GeneticAlgorithmConfig):
        self.config = config
        self.operators = [
            MutationOperator.bit_flip,
            MutationOperator.one_flip,
            MutationOperator.trois_flips,
            MutationOperator.cinq_flips
        ]
        self.operator_names = ['bit-flip', '1-flip', '3-flip', '5-flip']
        self.prob_dist = [1 / len(self.operators)] * len(self.operators)

    def choose_operator(self, isAdaptative=True) -> Callable:
        """Sélectionne un opérateur selon la distribution de probabilité"""
        if not isAdaptative:
            return MutationOperator.bit_flip
        return random.choices(self.operators, weights=self.prob_dist)[0]

    def update_probabilities(self, utilities: List[List[float]]) -> None:
        """Met à jour les probabilités des opérateurs"""
        sum_utilities = sum(u[-1] for u in utilities)
        if sum_utilities != 0:
            for i, _ in enumerate(self.operators):
                self.prob_dist[i] = (self.config.p_min +
                                     (1 - len(self.operators) * self.config.p_min) *
                                     utilities[i][-1] / sum_utilities)

    def calculate_utility(self, operator: Callable, utilities: List[List[float]],
                          gains: List[List[float]]) -> None:
        """Calcule l'utilité d'un opérateur"""
        op_index = self.operators.index(operator)
        ui = ((1 - self.config.alpha) * utilities[op_index][-1] +
              self.config.alpha * gains[op_index][-1])
        utilities[op_index].append(ui)


class GeneticAlgorithm:
    """Classe principale de l'algorithme génétique"""

    def __init__(self, config: GeneticAlgorithmConfig, isMasked=False):
        self.config = config
        self.isMasked = isMasked
        self.toolbox = self._setup_toolbox()
        self.roulette = AdaptiveRoulette(config)
        self.masque = [random.choice([0, 1]) for _ in range(GeneticAlgorithmConfig.one_max_length)]

    def _setup_toolbox(self) -> base.Toolbox:
        """Configure la boîte à outils DEAP"""
        toolbox = base.Toolbox()

        # Création des types
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Enregistrement des opérateurs
        toolbox.register("zero", random.randint, 0, 0)
        toolbox.register("individualCreator", tools.initRepeat,
                         creator.Individual, toolbox.zero,
                         self.config.one_max_length)
        toolbox.register("populationCreator", tools.initRepeat,
                         list, toolbox.individualCreator)

        # Opérateurs génétiques
        toolbox.register("evaluate", self._evaluate_masked if self.isMasked else self._evaluate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("worst", tools.selWorst, fit_attr='fitness')

        return toolbox

    @staticmethod
    def _evaluate(individual: List[int]) -> Tuple[float]:
        """Fonction d'évaluation"""
        return sum(individual),

    def _evaluate_masked(self, individual: List[int]) -> Tuple[float]:
        sum = 0
        for i in range(0, GeneticAlgorithmConfig.one_max_length):
            sum += individual[i] * self.masque[i]
        return sum,

    def _insertion_best_fitness(self, population: List, offspring: List) -> List:
        """Insertion des meilleurs individus dans la population"""
        worst = self.toolbox.worst(population, 1)
        for ind in offspring:
            if ind.fitness.values[0] > worst[0].fitness.values[0]:
                population.remove(worst[0])
                population.append(ind)
                worst = self.toolbox.worst(population, 1)
        return population

    def _single_run(self, isAdaptative=True) -> Tuple[List[float], List[float], List[List[float]]]:
        """Exécute une seule instance de l'algorithme"""
        # Initialisation des accumulateurs
        gains = [[0.] for _ in range(len(self.roulette.operators))]
        utilities = [[0.] for _ in range(len(self.roulette.operators))]
        max_fitness_values = []
        mean_fitness_values = []
        proba_distrib_values = [[self.roulette.prob_dist[i]] for i in range(len(self.roulette.operators))]

        # Création de la population initiale
        population = self.toolbox.populationCreator(n=self.config.population_size)

        # Évaluation initiale
        fitnesses = list(map(self.toolbox.evaluate, population))
        for individual, fitness_value in zip(population, fitnesses):
            individual.fitness.values = fitness_value

        # Calcul de la fitness moyenne initiale
        fitness_values = [ind.fitness.values[0] for ind in population]
        mean_fitness = sum(fitness_values) / len(population)

        # Boucle principale d'évolution
        for gen in range(self.config.max_generations):
            old_fitness = mean_fitness

            # Sélection et clonage
            offspring = list(map(self.toolbox.clone, self.toolbox.select(population, 2)))

            # Application de l'opérateur de mutation choisi
            operator = self.roulette.choose_operator(isAdaptative)
            for individual in offspring:
                if random.random() < self.config.p_mutation:
                    operator(individual)
                    del individual.fitness.values

            # Évaluation des nouveaux individus
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Insertion dans la population
            population = self._insertion_best_fitness(population, offspring)

            # Mise à jour des statistiques
            fitness_values = [ind.fitness.values[0] for ind in population]
            max_fitness = max(fitness_values)
            mean_fitness = sum(fitness_values) / len(population)
            if isAdaptative:
                # Calcul du gain et mise à jour des utilités
                gain = max(0, mean_fitness - old_fitness)
                for op in self.roulette.operators:
                    op_index = self.roulette.operators.index(op)
                    gains[op_index].append(0. if op != operator else gain)

                self.roulette.calculate_utility(operator, utilities, gains)
                self.roulette.update_probabilities(utilities)

            # Enregistrement des valeurs
            for i in range(len(self.roulette.operators)):
                proba_distrib_values[i].append(self.roulette.prob_dist[i])
            max_fitness_values.append(max_fitness)
            mean_fitness_values.append(mean_fitness)

        return max_fitness_values, mean_fitness_values, proba_distrib_values

    def _calculate_statistics(self, maxFitness_history: List[List[float]],
                              meanFitness_history: List[List[float]],
                              proba_distrib_history: List[List[List[float]]]) -> Tuple[
        List[float], List[float], List[List[float]]]:
        """Calcule les statistiques sur plusieurs runs"""
        # Calcul des moyennes de fitness
        mean_max_fitness = []
        mean_avg_fitness = []
        for gen in range(self.config.max_generations):
            mean_max_fitness.append(sum(run[gen] for run in maxFitness_history) / self.config.nb_runs)
            mean_avg_fitness.append(sum(run[gen] for run in meanFitness_history) / self.config.nb_runs)

        # Calcul des moyennes de probabilités
        mean_proba_op = [[] for _ in range(len(self.roulette.operators))]
        for op_idx in range(len(self.roulette.operators)):
            for gen in range(self.config.max_generations):
                mean_proba = sum(run[op_idx][gen] for run in proba_distrib_history) / self.config.nb_runs
                mean_proba_op[op_idx].append(mean_proba)

        return mean_max_fitness, mean_avg_fitness, mean_proba_op

    def run(self, isAdaptative=True) -> Tuple[List[float], List[float], List[List[float]]]:
        """Exécute l'algorithme génétique avec la roulette adaptative"""
        maxFitness_history = []
        meanFitness_history = []
        proba_distrib_history = []

        for _ in range(self.config.nb_runs):
            max_fitness, mean_fitness, proba_distrib = self._single_run(isAdaptative)
            maxFitness_history.append(max_fitness)
            meanFitness_history.append(mean_fitness)
            proba_distrib_history.append(proba_distrib)

        return self._calculate_statistics(maxFitness_history,
                                          meanFitness_history,
                                          proba_distrib_history)


class Visualiser:
    def mutation_op_distribution(self, mean_max: List[float], mean_avg: List[float],
                                 mean_proba_op: List[List[float]], algorithm) -> None:
        # Graphique des distributions
        plt.figure(figsize=(10, 6))
        colors = ['red', 'black', 'pink', 'purple']
        for i, (proba, name) in enumerate(zip(mean_proba_op,
                                              algorithm.roulette.operator_names)):
            plt.plot(proba, color=colors[i], label=name)
        plt.legend()
        plt.xlabel('Generation')
        plt.ylabel('Distribution')
        plt.title(f'Distribution des opérateurs')
        plt.show()

    def compare_roulettes(self, comparateur: List[dict]) -> None:
        """Compare les roulettes fixe et adaptative"""
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        for comp in comparateur:
            plt.plot(comp["mean_avg"], color=comp["color"],
                     label=f'Fitness moyenne ({comp["nom"]})')
        plt.legend()
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(f'Comparaison des fitness entre roulette fixe et adaptative')
        plt.show()


def main():
    """Point d'entrée principal"""
    config = GeneticAlgorithmConfig()
    # Pour appliquer le masque GeneticAlgorithm(config, isMasked=True)
    algorithm = GeneticAlgorithm(config)
    visualiser = Visualiser()

    # Configuration des différentes versions à comparer
    COMPARATEUR = [
        {
            "nom": "Roulette fixe",
            "isAdaptative": False,
            "color": "red",
            "values": []
        },
        {
            "nom": "Roulette adaptative",
            "isAdaptative": True,
            "color": "purple",
            "values": []
        }
    ]

    # Exécution des différentes versions
    for comp in COMPARATEUR:  # Afficher la distribution des opérateurs pour chaque version

        mean_max, mean_avg, mean_proba_op = algorithm.run(isAdaptative=comp["isAdaptative"])
        comp["mean_max"] = mean_max
        comp["mean_avg"] = mean_avg
        comp["mean_proba_op"] = mean_proba_op
        if comp["isAdaptative"]:
            visualiser.mutation_op_distribution(mean_max, mean_avg, mean_proba_op, algorithm)

    # Comparaison des deux roulettes sur le même graphique
    visualiser.compare_roulettes(COMPARATEUR)


if __name__ == '__main__':
    main()
