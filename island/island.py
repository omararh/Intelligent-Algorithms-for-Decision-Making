import numpy as np
from individual import Individual
from config import GeneticConfig


class Island:
    def __init__(self, island_id, mutation_operator, island_name):
        """Initialize island with vectorized operations."""
        self.island_id = island_id
        self.individuals = [Individual() for _ in range(GeneticConfig.POPULATION_SIZE)]
        self.mutation_operator = mutation_operator
        self.island_name = island_name

        # Pre-allocate arrays for optimization
        self._temp_genome = np.zeros(GeneticConfig.GENOME_LENGTH, dtype=np.int8)
        self._temp_population = np.zeros(
            (GeneticConfig.POPULATION_SIZE, GeneticConfig.GENOME_LENGTH),
            dtype=np.int8
        )

    def get_best_individual(self):
        """Returns the individual with highest fitness using vectorized operations."""
        if not self.individuals:
            return None

        fitness_values = np.array([ind.calculate_fitness() for ind in self.individuals])
        best_idx = np.argmax(fitness_values)
        return self.individuals[best_idx]

    def get_best_fitness(self):
        """Returns the highest fitness value in the population."""
        best = self.get_best_individual()
        return best.calculate_fitness() if best else 0

    def evolve_population(self, optimizer):
        """
        Evolves the population using vectorized operations and batch processing.
        Implements performance optimizations for mutation and fitness evaluation.
        """
        # Process individuals in batches for better performance
        for i in range(0, len(self.individuals), GeneticConfig.BATCH_SIZE):
            batch = self.individuals[i:i + GeneticConfig.BATCH_SIZE]

            # Store original fitness values
            original_fitness = np.array([ind.calculate_fitness() for ind in batch])

            # Perform mutations in batch
            for j, individual in enumerate(batch):
                np.copyto(self._temp_genome, individual.genome)
                mutated_genome = self.mutation_operator(self._temp_genome)

                # Calculate new fitness using vectorized operation
                new_fitness = np.sum(mutated_genome)
                optimizer.evaluation_count += 1

                if new_fitness > original_fitness[j]:
                    np.copyto(individual.genome, mutated_genome)
                    individual.invalidate_fitness()
                    individual.fitness_improvement = new_fitness - original_fitness[j]
                else:
                    individual.fitness_improvement = 0

    def add_individuals(self, new_individuals):
        """Add multiple individuals efficiently."""
        self.individuals.extend(new_individuals)

    def remove_individuals(self, indices):
        """Remove multiple individuals efficiently."""
        mask = np.ones(len(self.individuals), dtype=bool)
        mask[indices] = False
        self.individuals = [ind for i, ind in enumerate(self.individuals) if mask[i]]
