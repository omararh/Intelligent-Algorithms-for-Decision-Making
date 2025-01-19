from deap import base, creator, tools
import random
import matplotlib.pyplot as plt
import seaborn as sns
from math import log, sqrt
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration de l'algorithme"""
    one_max_length: int = 300
    population_size: int = 20
    p_mutation: float = 1.0
    max_generations: int = 3000
    nb_runs: int = 30
    alpha: float = 0.1
    pmin: float = 0.125
    window_size: int = 10


class MutationOperators:
    """Classe regroupant les opérateurs de mutation"""

    @staticmethod
    def flip(b: int) -> int:
        return 0 if b == 1 else 1

    @staticmethod
    def one_flip(individual: List[int]) -> None:
        pos = random.randint(0, len(individual) - 1)
        individual[pos] = MutationOperators.flip(individual[pos])

    @staticmethod
    def n_flips(individual: List[int], n: int) -> None:
        positions = random.sample(range(len(individual)), n)
        for pos in positions:
            individual[pos] = MutationOperators.flip(individual[pos])

    @staticmethod
    def trois_flips(individual: List[int]) -> None:
        MutationOperators.n_flips(individual, 3)

    @staticmethod
    def cinq_flips(individual: List[int]) -> None:
        MutationOperators.n_flips(individual, 5)

    @staticmethod
    def bit_flip(individual: List[int]) -> None:
        for i in range(len(individual)):
            if random.random() < 1 / len(individual):
                individual[i] = MutationOperators.flip(individual[i])


class UCBAlgorithm:
    """Classe principale pour l'algorithme UCB"""

    def __init__(self, config: Config):
        self.config = config
        self.toolbox = self._setup_toolbox()
        self.operators = [
            MutationOperators.bit_flip,
            MutationOperators.one_flip,
            MutationOperators.trois_flips,
            MutationOperators.cinq_flips
        ]

    def _setup_toolbox(self) -> base.Toolbox:
        """Configuration initiale du toolbox DEAP"""
        toolbox = base.Toolbox()
        toolbox.register("zero", random.randint, 0, 0)

        # Création des classes de base
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Configuration du toolbox
        toolbox.register("individualCreator",
                         tools.initRepeat,
                         creator.Individual,
                         toolbox.zero,
                         self.config.one_max_length)
        toolbox.register("populationCreator",
                         tools.initRepeat,
                         list,
                         toolbox.individualCreator)
        toolbox.register("evaluate", self._one_max_fitness)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("worst", tools.selWorst, fit_attr='fitness')

        return toolbox

    def _run_generation(self, population, prob_dist, gain, count_use_op, mean_fitness, generation):
        """Exécute une génération de l'algorithme"""
        old_fitness = mean_fitness

        # Sélection des parents mode steady state
        offspring = self.toolbox.select(population, 2)
        # Clonage pour éviter de modifier les individus origineaux
        offspring = list(map(self.toolbox.clone, offspring))

        # Sélection et application de l'opérateur
        operator = random.choices(self.operators, weights=prob_dist)[0]
        for individual in offspring:
            if random.random() < self.config.p_mutation:
                operator(individual)
            del individual.fitness.values

        # Évaluation des nouveaux individus
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = self.toolbox.evaluate(ind)

        # Mise à jour de la population
        population = self._insertion_best_fitness(population, offspring)

        # Calcul des nouvelles fitness
        fitness_values = [ind.fitness.values[0] for ind in population]
        max_fitness = max(fitness_values)
        mean_fitness = sum(fitness_values) / len(population)

        # Mise à jour des gains
        current_gain = max(0, mean_fitness - old_fitness)
        operator_index = self.operators.index(operator)
        gain[operator_index].pop(0)
        gain[operator_index].append(current_gain)

        # Mise à jour des compteurs d'utilisation
        for i, op in enumerate(self.operators):
            count_use_op[i].append(1 if op == operator else 0)

        # Mise à jour des probabilités
        self._update_probabilities(prob_dist, count_use_op, gain, generation + 1)

        return {
            'max_fitness': max_fitness,
            'mean_fitness': mean_fitness
        }

    def _process_histories(self, histories: dict) -> Tuple[List[float], List[float], List[List[float]]]:
        """Traitement des historiques pour préparer l'affichage"""
        # Calcul des moyennes pour max et mean fitness
        mean_max_fitness_values = []
        mean_mean_fitness_values = []

        for g in range(self.config.max_generations):
            mean_max_fitness_values.append(
                sum(run[g] for run in histories['max_fitness']) / self.config.nb_runs
            )
            mean_mean_fitness_values.append(
                sum(run[g] for run in histories['mean_fitness']) / self.config.nb_runs
            )

        # Calcul des moyennes pour les probabilités des opérateurs
        mean_proba_op = [[] for _ in range(len(self.operators))]

        for op in range(len(self.operators)):
            for g in range(self.config.max_generations):
                mean_proba = sum(
                    run[op][g] for run in histories['proba_distrib']
                ) / self.config.nb_runs
                mean_proba_op[op].append(mean_proba)

        return mean_max_fitness_values, mean_mean_fitness_values, mean_proba_op

    @staticmethod
    def _one_max_fitness(individual: List[int]) -> Tuple[int]:
        """Calcul de la fitness"""
        return sum(individual),

    def _normalize_array(self, gain: List[List[float]]) -> List[List[float]]:
        """Normalisation des gains"""
        flattened_array = [val for sublist in gain for val in sublist]
        min_val = min(flattened_array)
        max_val = max(flattened_array)

        if max_val == min_val:
            return [[0.01] * len(array) for array in gain]

        return [[(val - min_val) / (max_val - min_val) * 0.99 + 0.01
                 for val in array] for array in gain]

    def _update_probabilities(self, prob_dist: List[float],
                              count_use_op: List[List[int]],
                              gain: List[List[float]],
                              generation: int) -> None:
        """Mise à jour des probabilités selon UCB"""
        gain_norm = self._normalize_array(gain)
        mean_gains = [sum(op_gain) / self.config.window_size for op_gain in gain_norm]
        mean_gain_all_op = sum(mean_gains)

        for idx, (gain_op, counts) in enumerate(zip(mean_gains, count_use_op)):
            sum_use_op = max(1, sum(counts))
            # UCB_i(t) = \bar{X}_i(t) + C\sqrt{\frac{\ln(t)}{n_i(t)}}
            prob_dist[idx] = (gain_op + 0.01 * sqrt(log(generation) / sum_use_op)) / mean_gain_all_op

    def _insertion_best_fitness(self, population: List, offspring: List) -> List:
        """Insertion des meilleurs individus"""
        worst = self.toolbox.worst(population, 1)
        for ind in offspring:
            if ind.fitness.values[0] > worst[0].fitness.values[0]:
                population.remove(worst[0])
                population.append(ind)
                worst = self.toolbox.worst(population, 1)
        return population

    def run(self) -> Tuple[List[float], List[float], List[List[float]]]:
        """Exécution de l'algorithme UCB"""
        histories = {
            'max_fitness': [],
            'mean_fitness': [],
            'proba_distrib': [],
            'count_use_op': []
        }

        for _ in range(self.config.nb_runs):
            run_data = self._single_run()
            for key in histories:
                histories[key].append(run_data[key])

        return self._process_histories(histories)

    def _single_run(self) -> dict:
        """Exécution d'une seule instance de l'algorithme"""
        prob_dist = [0.25] * len(self.operators)
        gain = [[0.25] * self.config.window_size for _ in range(len(self.operators))]
        count_use_op = [[] for _ in range(len(self.operators))]

        population = self.toolbox.populationCreator(n=self.config.population_size)

        # Initialisation des fitness
        for ind in population:
            ind.fitness.values = self.toolbox.evaluate(ind)

        fitness_values = [ind.fitness.values[0] for ind in population]
        mean_fitness = sum(fitness_values) / len(population)

        # Historiques pour cette exécution
        max_fitness_values = []
        mean_fitness_values = []
        proba_distrib_values = [[0.25] for _ in range(len(self.operators))]

        # Boucle principale
        for gen in range(self.config.max_generations):
            generation_data = self._run_generation(population, prob_dist, gain,
                                                   count_use_op, mean_fitness, gen)

            # Mise à jour des historiques
            max_fitness_values.append(generation_data['max_fitness'])
            mean_fitness_values.append(generation_data['mean_fitness'])
            for i, prob in enumerate(prob_dist):
                proba_distrib_values[i].append(prob)

        return {
            'max_fitness': max_fitness_values,
            'mean_fitness': mean_fitness_values,
            'proba_distrib': proba_distrib_values,
            'count_use_op': count_use_op
        }

    def plot_results(self, mean_max_fitness: List[float],
                     mean_mean_fitness: List[float],
                     mean_proba_op: List[List[float]]) -> None:
        """Affichage des résultats"""
        # Configuration de style
        sns.set_style("whitegrid")

        # Premier graphique: Fitness
        plt.figure(figsize=(10, 6))
        plt.plot(mean_max_fitness, color='blue', label='Fitness max')
        plt.plot(mean_mean_fitness, color='orange', label='Fitness moyenne')
        plt.legend()
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(f'Fitness moyenne et max sur {self.config.nb_runs} runs')
        plt.show()

        # Second graphique: Distribution des opérateurs
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'red', 'green', 'black']
        labels = ['bit-flip', '1-flip', '3-flip', '5-flip']

        for op_proba, color, label in zip(mean_proba_op, colors, labels):
            plt.plot(op_proba, color=color, label=label)

        plt.legend()
        plt.xlabel('Generation')
        plt.ylabel('Distribution')
        plt.title(f'Distribution des opérateurs sur {self.config.nb_runs} runs')
        plt.show()


def main():
    """Point d'entrée principal"""
    random.seed()  # Initialisation du générateur aléatoire

    # Configuration
    config = Config()

    # Création et exécution de l'algorithme
    ucb = UCBAlgorithm(config)
    mean_max_fitness, mean_mean_fitness, mean_proba_op = ucb.run()

    # Affichage des résultats
    ucb.plot_results(mean_max_fitness, mean_mean_fitness, mean_proba_op)


if __name__ == '__main__':
    main()
