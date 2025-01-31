class Config:
    # Paramètres généraux
    VECTOR_SIZE = 1000
    POPULATION_SIZE = 100
    MAX_ITERATIONS = 5000
    NUM_ISLANDS = 4
    NUM_RUNS = 1

    # Paramètres d'apprentissage
    ALPHA = 0.8    # Inertie
    BETA = 0.1     # Facteur d'apprentissage
    NOISE = 0.1    # Bruit pour l'exploration
