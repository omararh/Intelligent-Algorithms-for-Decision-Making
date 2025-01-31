from settings import Config
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Individual:
    bits: List[int] = field(default_factory=lambda: [0] * Config.VECTOR_SIZE)
    origin: Optional[int] = None
    current_island: Optional[int] = None
    upgrade: float = 0
    migrated: bool = False

    def get_fitness(self) -> int:
        return sum(self.bits)

    def clone(self) -> 'Individual':
        return Individual(
            bits=self.bits.copy(),
            origin=self.origin,
            current_island=self.current_island,
            upgrade=self.upgrade,
            migrated=self.migrated
        )
