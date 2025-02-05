from collections import defaultdict
import random
from island import Island
from mutations import MutationOperators
from config import GeneticConfig


class IslandBasedOptimizer:
    """
    Manages multiple populations (islands) with adaptive migration policies
    to optimize a binary genome using different mutation strategies.
    """

    def __init__(self):
        # Initialize islands with different mutation strategies
        self.islands = [
            Island(0, MutationOperators.single_bit_flip, "SingleMutation_Island"),
            Island(1, MutationOperators.triple_bit_flip, "TripleMutation_Island"),
            Island(2, MutationOperators.quintuple_bit_flip, "QuintupleMutation_Island"),
            Island(3, MutationOperators.uniform_bit_flip, "UniformMutation_Island")
        ]

        # Initialize migration probability matrix (uniform distribution initially)
        self.migration_probabilities = [
            [1.0 / GeneticConfig.NUM_ISLANDS for _ in range(GeneticConfig.NUM_ISLANDS)]
            for _ in range(GeneticConfig.NUM_ISLANDS)
        ]

        # Initialize reward matrix for migration policy adaptation
        self.migration_rewards = [
            [0.0 for _ in range(GeneticConfig.NUM_ISLANDS)]
            for _ in range(GeneticConfig.NUM_ISLANDS)
        ]

        # Track global optimization progress
        self.global_best_fitness = 0
        self.global_best_individual = None
        self.evaluation_count = 0

    def get_best_individual(self):
        """Retrieves the best individual across all islands."""
        best_candidates = []
        for island in self.islands:
            best = island.get_best_individual()
            if best is not None:
                best_candidates.append(best)

        if not best_candidates:
            return None

        return max(best_candidates, key=lambda x: x.calculate_fitness())

    def get_best_fitness(self):
        """Returns the highest fitness value found across all islands."""
        best = self.get_best_individual()
        return best.calculate_fitness() if best else 0

    def execute_generation(self):
        """Executes one complete generation of the distributed evolution process."""
        self._handle_migration()
        self._evolve_populations()
        self._update_global_best()
        self._compute_migration_rewards()
        self._adapt_migration_policy()

    def _handle_migration(self):
        """Manages the migration of individuals between islands."""
        for source_island in self.islands:
            migration_probs = self.migration_probabilities[source_island.island_id]

            for individual in source_island.individuals[:]:  # Copy for safe iteration
                if not individual.has_migrated:
                    target_island_id = self._select_target_island(migration_probs)

                    # Update individual's migration status
                    individual.source_island = source_island.island_id
                    individual.current_island = target_island_id
                    individual.has_migrated = True

                    # Perform the migration
                    source_island.individuals.remove(individual)
                    self.islands[target_island_id].individuals.append(individual)

        # Reset migration flags for next generation
        for island in self.islands:
            for individual in island.individuals:
                individual.has_migrated = False

    def _select_target_island(self, probabilities):
        """Selects a target island for migration based on probability distribution."""
        random_value = random.random()
        cumulative_prob = 0.0

        for island_id, prob in enumerate(probabilities):
            cumulative_prob += prob
            if random_value < cumulative_prob:
                return island_id

        # Fallback to random selection if numerical issues occur
        return random.randrange(GeneticConfig.NUM_ISLANDS)

    def _evolve_populations(self):
        """Evolves all island populations using their respective mutation strategies."""
        for island in self.islands:
            island.evolve_population(self)

    def _update_global_best(self):
        """Updates the global best solution if a better one is found."""
        current_best_fitness = self.get_best_fitness()
        if current_best_fitness > self.global_best_fitness:
            self.global_best_fitness = current_best_fitness
            self.global_best_individual = self.get_best_individual()

    def _compute_migration_rewards(self):
        """Computes rewards for migration paths based on fitness improvements."""
        # Reset reward matrix
        self.migration_rewards = [[0.0 for _ in range(GeneticConfig.NUM_ISLANDS)]
                                  for _ in range(GeneticConfig.NUM_ISLANDS)]

        # Group migrants by their migration path (source -> destination)
        migration_groups = defaultdict(list)
        for island in self.islands:
            for individual in island.individuals:
                if individual.source_island is not None:
                    key = (individual.source_island, individual.current_island)
                    migration_groups[key].append(individual)

        # Calculate rewards based on fitness improvements
        for source_id in range(GeneticConfig.NUM_ISLANDS):
            improvements_by_destination = {}

            # Calculate average improvement for each destination
            for dest_id in range(GeneticConfig.NUM_ISLANDS):
                migrants = migration_groups.get((source_id, dest_id), [])
                if migrants:
                    avg_improvement = sum(ind.fitness_improvement for ind in migrants) / len(migrants)
                    improvements_by_destination[dest_id] = avg_improvement
                else:
                    improvements_by_destination[dest_id] = 0.0

            # Identify and reward best performing migration paths
            best_improvement = max(improvements_by_destination.values())
            if best_improvement > 0:
                best_destinations = [
                    dest for dest, impr in improvements_by_destination.items()
                    if impr == best_improvement
                ]
                reward_value = 1.0 / len(best_destinations)
                for dest in best_destinations:
                    self.migration_rewards[source_id][dest] = reward_value

    def _adapt_migration_policy(self):
        """Updates migration probabilities based on observed rewards."""
        for i in range(GeneticConfig.NUM_ISLANDS):
            for j in range(GeneticConfig.NUM_ISLANDS):
                # Apply learning rate and exploration rate
                current_prob = self.migration_probabilities[i][j]
                reward = self.migration_rewards[i][j]

                new_prob = (1 - GeneticConfig.EXPLORATION_RATE) * (
                        GeneticConfig.LEARNING_RATE * current_prob +
                        (1 - GeneticConfig.LEARNING_RATE) * reward
                ) + GeneticConfig.EXPLORATION_RATE * GeneticConfig.MUTATION_NOISE

                self.migration_probabilities[i][j] = new_prob

            # Normalize probabilities
            row_sum = sum(self.migration_probabilities[i])
            if row_sum > 0:
                self.migration_probabilities[i] = [
                    p / row_sum for p in self.migration_probabilities[i]
                ]
            else:
                # Reset to uniform distribution if normalization fails
                self.migration_probabilities[i] = [
                    1.0 / GeneticConfig.NUM_ISLANDS for _ in range(GeneticConfig.NUM_ISLANDS)
                ]
