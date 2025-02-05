import numpy as np
from collections import defaultdict
from island import Island
from mutations import MutationOperators
from config import GeneticConfig


class IslandBasedOptimizer:
    def __init__(self):
        """Initialize optimizer with vectorized data structures."""
        # Initialize islands
        self.islands = [
            Island(0, MutationOperators.single_bit_flip, "SingleMutation_Island"),
            Island(1, MutationOperators.triple_bit_flip, "TripleMutation_Island"),
            Island(2, MutationOperators.quintuple_bit_flip, "QuintupleMutation_Island"),
            Island(3, MutationOperators.uniform_bit_flip, "UniformMutation_Island")
        ]

        # Use NumPy arrays for matrices
        self.migration_probabilities = np.full(
            (GeneticConfig.NUM_ISLANDS, GeneticConfig.NUM_ISLANDS),
            1.0 / GeneticConfig.NUM_ISLANDS
        )

        self.migration_rewards = np.zeros(
            (GeneticConfig.NUM_ISLANDS, GeneticConfig.NUM_ISLANDS)
        )

        self.global_best_fitness = 0
        self.global_best_individual = None
        self.evaluation_count = 0

        # Pre-allocate arrays for optimization
        self._temp_rewards = np.zeros_like(self.migration_rewards)
        self._temp_probs = np.zeros_like(self.migration_probabilities)

    def get_best_individual(self):
        """Returns best individual using vectorized operations."""
        best_candidates = [island.get_best_individual() for island in self.islands]
        best_candidates = [b for b in best_candidates if b is not None]

        if not best_candidates:
            return None

        fitness_values = np.array([ind.calculate_fitness() for ind in best_candidates])
        return best_candidates[np.argmax(fitness_values)]

    def get_best_fitness(self):
        """Returns highest fitness using cached values when possible."""
        best = self.get_best_individual()
        return best.calculate_fitness() if best else 0

    def execute_generation(self):
        """Executes one generation with optimized operations."""
        self._handle_migration()
        self._evolve_populations()
        self._update_global_best()
        self._compute_migration_rewards()
        self._adapt_migration_policy()

    def _handle_migration(self):
        """Optimized migration handling using batch operations."""
        # Group migrations by destination for batch processing
        migrations = defaultdict(list)
        source_islands = defaultdict(list)

        for source_island in self.islands:
            probs = self.migration_probabilities[source_island.island_id]

            # Process all potential migrants at once
            candidates = [ind for ind in source_island.individuals if not ind.has_migrated]
            if not candidates:
                continue

            # Generate all destination choices at once
            random_values = np.random.random(len(candidates))
            cumulative_probs = np.cumsum(probs)

            for ind, rand_val in zip(candidates, random_values):
                dest_id = np.searchsorted(cumulative_probs, rand_val)
                if dest_id >= GeneticConfig.NUM_ISLANDS:
                    dest_id = np.random.randint(GeneticConfig.NUM_ISLANDS)

                migrations[dest_id].append(ind)
                source_islands[dest_id].append(source_island)

        # Perform migrations in batch
        for dest_id in migrations:
            migrants = migrations[dest_id]
            sources = source_islands[dest_id]

            # Update migrants
            for migrant, source in zip(migrants, sources):
                source.individuals.remove(migrant)
                migrant.source_island = source.island_id
                migrant.current_island = dest_id
                migrant.has_migrated = True

            # Add all migrants to destination
            self.islands[dest_id].add_individuals(migrants)

        # Reset migration flags efficiently
        for island in self.islands:
            for ind in island.individuals:
                ind.has_migrated = False

    def _evolve_populations(self):
        """Evolve all populations in parallel."""
        for island in self.islands:
            island.evolve_population(self)

    def _update_global_best(self):
        """Update global best solution efficiently."""
        current_best_fitness = self.get_best_fitness()
        if current_best_fitness > self.global_best_fitness:
            self.global_best_fitness = current_best_fitness
            self.global_best_individual = self.get_best_individual()

    def _compute_migration_rewards(self):
        """
        Computes rewards using intensification strategy as described in the paper:
        Only the best island is rewarded with D[j] = 1/|B| if j ∈ B, 0 otherwise,
        where B = argmax(D_in)
        """
        # Reset reward matrix
        self.migration_rewards = np.zeros((GeneticConfig.NUM_ISLANDS, GeneticConfig.NUM_ISLANDS))

        # Group migrants by their migration path (source -> destination)
        migrants_by_path = defaultdict(list)
        for island in self.islands:
            for ind in island.individuals:
                if ind.source_island is not None:
                    path = (ind.source_island, ind.current_island)
                    migrants_by_path[path].append(ind.fitness_improvement)

        # Calculate average improvement for each source island
        improvements_by_source = defaultdict(dict)
        for (source, dest), improvements in migrants_by_path.items():
            if improvements:  # Si nous avons des améliorations pour ce chemin
                avg_improvement = sum(improvements) / len(improvements)
                improvements_by_source[source][dest] = avg_improvement

        # Pour chaque île source, identifie la/les meilleures destinations
        for source in range(GeneticConfig.NUM_ISLANDS):
            if source in improvements_by_source:
                # Trouve la meilleure valeur d'amélioration
                best_improvement = max(improvements_by_source[source].values())

                # Trouve toutes les destinations qui ont atteint cette meilleure valeur
                best_destinations = [
                    dest for dest, impr in improvements_by_source[source].items()
                    if impr == best_improvement and best_improvement > 0
                ]

                # Si nous avons des meilleures destinations, distribue la récompense
                if best_destinations:
                    reward_value = 1.0 / len(best_destinations)
                    for dest in best_destinations:
                        self.migration_rewards[source][dest] = reward_value

    def _adapt_migration_policy(self):
        """
        Update migration probabilities using the formula from the paper:
        V = (1 - β)(α.V + (1 - α).D) + β.N
        Where:
        - V is the transition vector (migration probabilities)
        - D is the reward vector
        - α is the learning rate (exploitation)
        - β is the exploration rate
        - N is a stochastic vector (noise)
        """
        # Calculate stochastic noise vector
        N = np.random.random((GeneticConfig.NUM_ISLANDS, GeneticConfig.NUM_ISLANDS))
        N = N / N.sum(axis=1)[:, np.newaxis]  # Normalize rows to sum to 1

        self._temp_probs = (
                (1 - GeneticConfig.EXPLORATION_RATE) * (
                GeneticConfig.LEARNING_RATE * self.migration_probabilities +
                (1 - GeneticConfig.LEARNING_RATE) * self.migration_rewards
        ) +
                GeneticConfig.EXPLORATION_RATE * N
        )

        # Normalize probabilities
        row_sums = np.sum(self._temp_probs, axis=1)
        self._temp_probs = self._temp_probs / row_sums[:, np.newaxis]

        # Update probabilities
        np.copyto(self.migration_probabilities, self._temp_probs)
