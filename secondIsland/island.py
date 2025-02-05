from individual import Individual
from config import GeneticConfig


class Island:
    def __init__(self, island_id, mutation_operator, island_name):
        self.island_id = island_id
        self.individuals = [Individual() for _ in range(GeneticConfig.POPULATION_SIZE)]
        self.mutation_operator = mutation_operator
        self.island_name = island_name

    def get_best_individual(self):
        """Returns the individual with highest fitness."""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda x: x.calculate_fitness())

    def get_best_fitness(self):
        """Returns the highest fitness in the population."""
        best = self.get_best_individual()
        return best.calculate_fitness() if best else 0

    def evolve_population(self, archipelago):
        """Evolves the population using the island's mutation operator."""
        for individual in self.individuals:
            initial_fitness = individual.calculate_fitness()
            mutated_genome = self.mutation_operator(individual.genome)
            mutated_fitness = sum(mutated_genome)

            archipelago.evaluation_count += 1

            if mutated_fitness > initial_fitness:
                individual.genome = mutated_genome
            individual.fitness_improvement = individual.calculate_fitness() - initial_fitness
