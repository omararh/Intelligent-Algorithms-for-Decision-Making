import numpy as np
from config import GeneticConfig


class Individual:
    def __init__(self):
        """Initialize individual with NumPy array for genome."""
        self.genome = np.zeros(GeneticConfig.GENOME_LENGTH, dtype=np.int8)
        self.source_island = None
        self.current_island = None
        self.fitness_improvement = 0
        self.has_migrated = False
        self._fitness = None  # Cache for fitness value

    def calculate_fitness(self):
        """Calculate fitness using vectorized NumPy operations."""
        if self._fitness is None:
            self._fitness = np.sum(self.genome)
        return self._fitness

    def invalidate_fitness(self):
        """Invalidate fitness cache when genome changes."""
        self._fitness = None

    def copy(self):
        """Create a deep copy of the individual."""
        new_ind = Individual()
        new_ind.genome = np.copy(self.genome)
        new_ind.source_island = self.source_island
        new_ind.current_island = self.current_island
        new_ind.fitness_improvement = self.fitness_improvement
        new_ind.has_migrated = self.has_migrated
        new_ind._fitness = self._fitness
        return new_ind
