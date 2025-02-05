from config import GeneticConfig


class Individual:
    def __init__(self):
        self.genome = [0 for _ in range(GeneticConfig.GENOME_LENGTH)]
        self.source_island = None
        self.current_island = None
        self.fitness_improvement = 0
        self.has_migrated = False

    def calculate_fitness(self):
        """Calculate the fitness (sum of 1s in the genome)."""
        return sum(self.genome)
