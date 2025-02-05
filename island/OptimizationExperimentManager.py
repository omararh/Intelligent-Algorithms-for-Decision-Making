import numpy as np
import matplotlib.pyplot as plt
from IslandBasedOptimizer import IslandBasedOptimizer
from config import GeneticConfig


class OptimizationExperimentManager:
    """
    Handles running multiple optimization experiments and analyzing their results.
    Includes functionality for statistical analysis and visualization.
    """

    @staticmethod
    def execute_single_optimization():
        """
        Executes a single complete optimization run and collects performance metrics.

        Returns:
            tuple: Contains histories of (fitness values, population sizes)
        """
        optimizer = OptimizationExperimentManager._initialize_optimizer()

        # Initialize tracking variables
        fitness_progression = []
        population_tracking = {island.island_name: [] for island in optimizer.islands}
        evaluation_tracking = []

        # Run optimization process
        for generation in range(GeneticConfig.MAX_GENERATIONS):
            if optimizer.global_best_fitness == GeneticConfig.GENOME_LENGTH:
                break

            optimizer.execute_generation()

            # Record current state
            OptimizationExperimentManager._record_generation_metrics(
                optimizer,
                fitness_progression,
                population_tracking,
                evaluation_tracking
            )

            # If optimal solution found, pad remaining generations
            if optimizer.global_best_fitness == GeneticConfig.GENOME_LENGTH:
                OptimizationExperimentManager._pad_metric_histories(
                    optimizer,
                    generation,
                    fitness_progression,
                    population_tracking,
                    evaluation_tracking
                )
                break

        return fitness_progression, population_tracking, evaluation_tracking, optimizer

    @staticmethod
    def _initialize_optimizer():
        """Creates and initializes a new optimizer instance."""
        optimizer = IslandBasedOptimizer()

        # Set initial island assignments
        for island in optimizer.islands:
            for individual in island.individuals:
                individual.current_island = island.island_id
                individual.source_island = island.island_id

        return optimizer

    @staticmethod
    def _record_generation_metrics(optimizer, fitness_history, population_history, eval_history):
        """Records performance metrics for the current generation."""
        fitness_history.append(optimizer.global_best_fitness)
        eval_history.append(optimizer.evaluation_count)

        for island in optimizer.islands:
            population_history[island.island_name].append(len(island.individuals))

    @staticmethod
    def _pad_metric_histories(optimizer, current_gen, fitness_history,
                              population_history, eval_history):
        """Pads metric histories with final values for remaining generations."""
        remaining_gens = range(current_gen + 1, GeneticConfig.MAX_GENERATIONS)

        for _ in remaining_gens:
            fitness_history.append(optimizer.global_best_fitness)
            eval_history.append(optimizer.evaluation_count)

            for island in optimizer.islands:
                population_history[island.island_name].append(len(island.individuals))

    @staticmethod
    def create_performance_visualizations(fitness_data, population_data, _, optimizer=None):
        """
        Creates and displays visualization plots of the experiment results.

        Args:
            fitness_data (np.array): Fitness values from all runs
            population_data (list): Population sizes per island
            optimizer: Instance of IslandBasedOptimizer pour accéder à la matrice de migration
        """
        # Calculate mean metrics
        mean_fitness = np.mean(fitness_data, axis=0)

        # Get island names from first run
        island_names = list(population_data[0].keys())

        # Calculate mean population sizes per island
        avg_populations = {name: [0.0] * GeneticConfig.MAX_GENERATIONS
                           for name in island_names}

        for name in island_names:
            for gen in range(GeneticConfig.MAX_GENERATIONS):
                avg_populations[name][gen] = np.mean([
                    run_data[name][gen] for run_data in population_data
                ])

        # Plot 1: Fitness Progression
        plt.figure(figsize=(10, 6))
        plt.plot(mean_fitness,
                 label=f"Mean Best Fitness ({GeneticConfig.NUM_EXPERIMENTS} runs)", color="purple")
        plt.xlabel("Generation")
        plt.ylabel("Fitness Value")
        plt.title("Evolution of Best Fitness Over Generations")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot 2: Population Dynamics
        colors = ['purple', 'pink', 'red', 'black']
        plt.figure(figsize=(10, 6))
        for name, color in zip(island_names, colors):
            plt.plot(avg_populations[name], label=f"{name}", color=color)
        plt.xlabel("Generation")
        plt.ylabel("Mean Population Size")
        plt.title(f"Population Dynamics per Island (Mean of {GeneticConfig.NUM_EXPERIMENTS} runs)\n" +
                  f"Learning Rate={GeneticConfig.LEARNING_RATE}, " +
                  f"Exploration Rate={GeneticConfig.EXPLORATION_RATE}")
        plt.legend()
        plt.grid(True)
        plt.show()
