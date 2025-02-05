"""Main entry point of the program."""
from OptimizationExperimentManager import OptimizationExperimentManager
from config import GeneticConfig
import numpy as np


def main():
    fitness_runs = []
    population_runs = []
    evaluation_runs = []

    final_optimizer = None  # Pour stocker le dernier optimizer

    for _ in range(GeneticConfig.NUM_EXPERIMENTS):
        fitness_history, pop_history, eval_history, optimizer = OptimizationExperimentManager.execute_single_optimization()
        fitness_runs.append(fitness_history)
        population_runs.append(pop_history)
        evaluation_runs.append(eval_history)
        final_optimizer = optimizer  # Garder le dernier optimizer

    fitness_array = np.array(fitness_runs)
    evaluations_array = np.array(evaluation_runs)

    OptimizationExperimentManager.create_performance_visualizations(
        fitness_array,
        population_runs,
        evaluations_array,
        final_optimizer  # Passer l'optimizer pour la matrice de migration
    )


if __name__ == "__main__":
    main()
