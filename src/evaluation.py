from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import v_measure_score

from config import Config

args = Config.get_args()


def calculate_distance_matrix(X, individual):
    num_samples = len(X)
    matrix_distances = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            distance = safe_eval(
                individual.phenotype,
                {
                    **{f"x_{k}": X[i][k] for k in range(args.individual_size)},
                    **{f"y_{k}": X[j][k] for k in range(args.individual_size)},
                },
            )
            matrix_distances[i, j] = distance
            matrix_distances[j, i] = distance

    return matrix_distances


def evaluate_fitness(individual, X, y):
    try:
        if individual.fitness is not None:
            return individual.fitness

        matrix_distances = calculate_distance_matrix(X, individual)
        clustering = AgglomerativeClustering(n_clusters=len(set(y)), metric="precomputed", linkage="complete")
        y_pred = clustering.fit_predict(matrix_distances)

        individual.fitness = v_measure_score(y, y_pred, beta=5.0)

    except ZeroDivisionError:
        print("div")
        individual.fitness = 0
    except Exception as e:
        print(f"Erro na avaliação da fitness: {e}")
        individual.fitness = 0

    return individual.fitness


def safe_eval(expression, vars_dict):
    try:
        result = eval(expression, {}, vars_dict)
        if np.isfinite(result):
            return result
        else:
            return 0
    except Exception as e:
        print(expression)
        print(f"Erro na expressão: {e}")
        return 0


def population_evaluate_fitness(population, X, y):
    if args.multithreading:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(evaluate_fitness, ind, X, y): ind for ind in population}
            for future in as_completed(futures):
                ind = futures[future]
                ind.fitness = future.result()

    else:
        for ind in population:
            ind.fitness = evaluate_fitness(ind, X, y)
