from typing import List, Optional
from settings import Config
from individual_model import Individual
from mutations import MutationType, MutationOperators


class Island:
    def __init__(self, id_island: int, mutation_type: MutationType, name: str):
        self.id = id_island
        self.mutation_type = mutation_type
        self.operator = MutationOperators.get_operator(mutation_type)
        self.name = name
        self.population = [Individual() for _ in range(Config.POPULATION_SIZE)]

    def get_best_element(self) -> Optional[Individual]:
        if not self.population:
            return None
        return max(self.population, key=lambda x: x.get_fitness())

    def get_best_fitness(self) -> int:
        best = self.get_best_element()
        return best.get_fitness() if best else 0

    def local_search(self) -> None:
        """Applique la recherche locale sur la population."""
        for ind in self.population:
            fitness_before = ind.get_fitness()
            child_bits = self.operator(ind.bits)
            child_fitness = sum(child_bits)
            if child_fitness > fitness_before:
                ind.bits = child_bits
            ind.upgrade = ind.get_fitness() - fitness_before
