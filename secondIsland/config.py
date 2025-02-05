class GeneticConfig:
    GENOME_LENGTH = 1000  # Previously TAILLE_VECTEUR
    POPULATION_SIZE = 20  # Previously TAILLE_POPULATION
    MAX_GENERATIONS = 5000  # Previously MAX_ITER
    NUM_ISLANDS = 4  # Previously N
    NUM_EXPERIMENTS = 10  # Previously NB_RUNS

    # Learning parameters
    LEARNING_RATE = 0.9  # Previously alpha
    EXPLORATION_RATE = 0.1  # Previously beta
    MUTATION_NOISE = 0.01  # Previously noise
