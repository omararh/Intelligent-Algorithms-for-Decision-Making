import random
from config import GeneticConfig


class MutationOperators:
    @staticmethod
    def single_bit_flip(genome):
        """Flips a single random bit in the genome."""
        offspring = genome[:]
        position = random.randint(0, GeneticConfig.GENOME_LENGTH - 1)
        offspring[position] = 1 - offspring[position]
        return offspring

    @staticmethod
    def triple_bit_flip(genome):
        """Flips three distinct random bits in the genome."""
        offspring = genome[:]
        positions = random.sample(range(GeneticConfig.GENOME_LENGTH), 3)
        for pos in positions:
            offspring[pos] = 1 - offspring[pos]
        return offspring

    @staticmethod
    def quintuple_bit_flip(genome):
        """Flips five distinct random bits in the genome."""
        offspring = genome[:]
        positions = random.sample(range(GeneticConfig.GENOME_LENGTH), 5)
        for pos in positions:
            offspring[pos] = 1 - offspring[pos]
        return offspring

    @staticmethod
    def uniform_bit_flip(genome):
        """Flips each bit with probability 1/genome_length."""
        offspring = genome[:]
        for i in range(GeneticConfig.GENOME_LENGTH):
            if random.random() < 1 / GeneticConfig.GENOME_LENGTH:
                offspring[i] = 1 - offspring[i]
        return offspring
