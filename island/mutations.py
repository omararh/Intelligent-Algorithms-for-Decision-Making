import random
from typing import List, Callable
from settings import Config
from enum import Enum, auto


class MutationType(Enum):
    ONE_FLIP = auto()
    THREE_FLIPS = auto()
    FIVE_FLIPS = auto()
    BIT_FLIP = auto()


class MutationOperators:
    @staticmethod
    def one_flip(bits: List[int]) -> List[int]:
        """Mutation qui inverse un seul bit aléatoire."""
        child = bits.copy()
        idx = random.randint(0, Config.VECTOR_SIZE - 1)
        child[idx] = 1 - child[idx]
        return child

    @staticmethod
    def n_flips(bits: List[int], n: int) -> List[int]:
        """Mutation qui inverse n bits aléatoires distincts."""
        child = bits.copy()
        indices = random.sample(range(Config.VECTOR_SIZE), n)
        for idx in indices:
            child[idx] = 1 - child[idx]
        return child

    @staticmethod
    def three_flips(bits: List[int]) -> List[int]:
        return MutationOperators.n_flips(bits, 3)

    @staticmethod
    def five_flips(bits: List[int]) -> List[int]:
        return MutationOperators.n_flips(bits, 5)

    @staticmethod
    def bit_flip(bits: List[int]) -> List[int]:
        """Mutation qui inverse chaque bit avec une probabilité de 1/taille."""
        child = bits.copy()
        for i in range(Config.VECTOR_SIZE):
            if random.random() < 1 / Config.VECTOR_SIZE:
                child[i] = 1 - child[i]
        return child

    @staticmethod
    def get_operator(mutation_type: MutationType) -> Callable:
        operators = {
            MutationType.ONE_FLIP: MutationOperators.one_flip,
            MutationType.THREE_FLIPS: MutationOperators.three_flips,
            MutationType.FIVE_FLIPS: MutationOperators.five_flips,
            MutationType.BIT_FLIP: MutationOperators.bit_flip
        }
        return operators[mutation_type]
