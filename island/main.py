from expirement import run_experiment
from settings import Config
import matplotlib.pyplot as plt


def main():
    # On va exécuter Config.NUM_RUNS fois l'expérience
    all_fitness_runs = []  # liste de best_fitness_history pour chaque run
    all_population_runs = []  # liste de population_history pour chaque run

    for run_index in range(Config.NUM_RUNS):
        bf_history, pop_history, manager = run_experiment()
        all_fitness_runs.append(bf_history)
        all_population_runs.append(pop_history)

    # Maintenant, on agrège (moyenne) sur Config.NUM_RUNS.
    # 1) Best Fitness moyenne
    # on suppose que toutes les best_fitness_history font EXACTEMENT Config.MAX_ITERATIONS de long
    # grâce au "remplissage" si la solution est atteinte avant la fin.
    avg_best_fitness = [0.0] * Config.MAX_ITERATIONS
    for i in range(Config.MAX_ITERATIONS):
        # Faire la moyenne sur les Config.NUM_RUNS
        s = 0.0
        for r in range(Config.NUM_RUNS):
            s += all_fitness_runs[r][i]
        avg_best_fitness[i] = s / Config.NUM_RUNS

    # 2) Population moyenne pour chaque île
    # on doit faire la même chose pour chaque île
    island_names = list(all_population_runs[0].keys())  # "Ile 1flip", etc.
    avg_populations = {name: [0.0] * Config.MAX_ITERATIONS for name in island_names}
    for name in island_names:
        for gen in range(Config.MAX_ITERATIONS):
            s = 0.0
            for r in range(Config.NUM_RUNS):
                s += all_population_runs[r][name][gen]
            avg_populations[name][gen] = s / Config.NUM_RUNS

    # 3) Tracer les courbes moyennes
    # a) Meilleure fitness moyenne
    plt.figure(figsize=(12, 6))
    plt.plot(avg_best_fitness, label="Meilleure Fitness (moyenne sur " + str(Config.NUM_RUNS) + " runs)")
    plt.xlabel('Génération')
    plt.ylabel('Fitness')
    plt.title('Évolution moyenne de la Meilleure Fitness sur ' + str(Config.NUM_RUNS) + ' runs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # b) Taille de population moyenne
    plt.figure(figsize=(12, 6))
    for name in island_names:
        plt.plot(avg_populations[name], label=f"{name} (moy.)")
    plt.xlabel('Génération')
    plt.ylabel('Taille de population moyenne')
    plt.title(
        f'Taille de population par Île sur {Config.NUM_RUNS} runs - Alpha {Config.ALPHA} - Beta {Config.BETA} - Noise {Config.NOISE}')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
