"""Main entry point of the program."""
from OptimizationExperimentManager import OptimizationExperimentManager
from config import GeneticConfig
import numpy as np


def main():
    fitness_runs = []
    population_runs = []
    evaluation_runs = []

    for _ in range(GeneticConfig.NUM_EXPERIMENTS):
        fitness_history, pop_history, eval_history, _ = OptimizationExperimentManager.execute_single_optimization()
        fitness_runs.append(fitness_history)
        population_runs.append(pop_history)
        evaluation_runs.append(eval_history)

    fitness_array = np.array(fitness_runs)
    evaluations_array = np.array(evaluation_runs)

    OptimizationExperimentManager.save_experiment_results(
        fitness_data=fitness_array,
        evaluation_data=evaluations_array
    )

    OptimizationExperimentManager.create_performance_visualizations(
        fitness_array,
        population_runs,
        evaluations_array
    )


if __name__ == "__main__":
    main()
