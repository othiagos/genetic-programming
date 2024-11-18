from concurrent.futures import ProcessPoolExecutor, as_completed

import numexpr as ne
import numpy as np
from numpy import ndarray
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import v_measure_score

from config import Config, set_seed
from individual import Individual

args = Config.get_args()


def calculate_distance_matrix(X: ndarray, individual: Individual) -> ndarray:
    """
    Calculates the pairwise distance matrix for the given data and individual.

    @param X: Data points as a NumPy array of shape (n_samples, n_features).
    @param individual: The individual containing the phenotype to compute distances.
    @return: A symmetric distance matrix (ndarray) of shape (n_samples, n_samples).
    """
    num_samples = len(X)
    matrix_distances = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            local_dict = {**args.precomputed_x_dicts[i], **args.precomputed_y_dicts[j]}

            distance = safe_eval(individual.phenotype, local_dict)
            matrix_distances[i, j] = distance
            matrix_distances[j, i] = distance

    return matrix_distances


def evaluate_fitness(X: ndarray, y: ndarray, individual: Individual, seed: int = None) -> float:
    """
    Evaluates the fitness of an individual using clustering and a custom distance matrix.

    @param X: Data points as a NumPy array of shape (n_samples, n_features).
    @param y: Ground truth labels as a NumPy array of shape (n_samples,).
    @param individual: The individual whose fitness is being evaluated.
    @param seed: Random seed for reproducibility (optional).
    @return: The fitness value (float) of the individual.
    """
    if seed is not None:
        set_seed(seed)

    try:
        if individual.fitness is not None:
            return individual.fitness

        matrix_distances = calculate_distance_matrix(X, individual)
        clustering = AgglomerativeClustering(n_clusters=len(set(y)), metric="precomputed", linkage="complete")
        y_pred = clustering.fit_predict(matrix_distances)

        individual.fitness = v_measure_score(y, y_pred, beta=5.0)

    except ZeroDivisionError:
        individual.fitness = 0
    except Exception as e:
        print(f"Error in fitness evaluation: {e}")
        individual.fitness = 0

    return individual.fitness


def safe_eval(expression: str, vars_dict: dict) -> float:
    """
    Safely evaluates a mathematical expression using predefined variables.

    @param expression: The mathematical expression to evaluate as a string.
    @param vars_dict: A dictionary of variable names and their values for evaluation.
    @return: The result of the evaluation (float). Returns 0.0 if the evaluation fails.
    """
    try:
        result = ne.evaluate(expression, vars_dict)
        if np.isfinite(result):
            return result
        else:
            return 0.0
    except Exception as e:
        print(expression)
        print(f"Error in the expression: {e}")
        return 0.0


def population_evaluate_fitness(X: ndarray, y: ndarray, population: list[Individual]) -> None:
    """
    Evaluates the fitness of a population of individuals, using multithreading if enabled.

    @param X: Data points as a NumPy array of shape (n_samples, n_features).
    @param y: Ground truth labels as a NumPy array of shape (n_samples,).
    @param population: A list of individuals whose fitness will be evaluated.
    """
    if args.multithreading:
        with ProcessPoolExecutor() as executor:
            futures = {}

            for ind in population:
                args.seed += 1
                futures[executor.submit(evaluate_fitness, X, y, ind, args.seed)] = ind

            for future in as_completed(futures):
                ind = futures[future]
                ind.fitness = future.result()

    else:
        for ind in population:
            ind.fitness = evaluate_fitness(X, y, ind)
