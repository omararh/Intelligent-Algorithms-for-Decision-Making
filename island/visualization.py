import matplotlib.pyplot as plt
from typing import List


def plot_fitness_history(avg_fitness: List[float], num_runs: int) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(avg_fitness, label=f"Meilleure Fitness (moyenne sur {num_runs} runs)")
    plt.xlabel('Génération')
    plt.ylabel('Fitness')
    plt.title(f'Évolution moyenne de la Meilleure Fitness sur {num_runs} runs')
    plt.show()
