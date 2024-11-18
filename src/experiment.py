import csv
import os

import numpy as np
from numpy import ndarray

from config import Config

args = Config.get_args()

EXPERIMENT_FOLDER = "experiment/"


def get_generation_info(population_fitness: np.ndarray) -> tuple[float, float, float, float]:
    """
    Computes statistics for a given population's fitness.

    @param population_fitness: A NumPy array containing the fitness values of the population.
    @return: A tuple containing:
        - Best fitness (float)
        - Minimum fitness (float)
        - Average fitness (float)
        - Standard deviation of fitness (float)
    """
    best_fitness = np.max(population_fitness) * 100
    min_fitness = np.min(population_fitness) * 100
    avg_fitness = np.mean(population_fitness) * 100
    std_fitness = np.std(population_fitness) * 100

    return best_fitness, min_fitness, avg_fitness, std_fitness


def process_population(train_population: ndarray, test_population: ndarray) -> dict:
    """
    Processes and computes statistics for training and test populations.

    @param train_population: Fitness values of the training population as a NumPy array.
    @param test_population: Fitness values of the test population as a NumPy array.
    @return: A dictionary containing fitness statistics for training and test populations.
    """
    train_population_fitness = np.array([float(ind) for ind in train_population])
    test_population_fitness = np.array([float(ind) for ind in test_population])

    train_best, train_min, train_avg, train_std = get_generation_info(train_population_fitness)
    test_best, test_min, test_avg, test_std = get_generation_info(test_population_fitness)

    return {
        "train_best": train_best,
        "train_avg": train_avg,
        "train_std": train_std,
        "test_best": test_best,
        "test_avg": test_avg,
        "test_std": test_std,
        "seed": args.seed_main,
    }


def get_experiment_file_name():
    """
    Generates the filename for experiment data based on configuration parameters.

    @return: The experiment filename as a string.
    """

    dataset = args.dataset.upper()
    population = args.population_size
    generation = args.generations
    mutation = int(args.mutation_prob * 100)
    tournament = args.tournament
    depth = args.depth
    return f"EXPR_{dataset}_P{population}_G{generation}_M{mutation:02d}_T{tournament}_D{depth}.csv"


def get_gen_info_file_name():
    """
    Generates the filename for generation information data based on configuration parameters.

    @return: The generation information filename as a string.
    """

    dataset = args.dataset.upper()
    population = args.population_size
    generation = args.generations
    mutation = int(args.mutation_prob * 100)
    tournament = args.tournament
    depth = args.depth
    return f"GEN_INFO_{dataset}_P{population}_G{generation}_M{mutation:02d}_T{tournament}_D{depth}.csv"


def save_info_experiment(train_population: ndarray, test_population: ndarray) -> None:
    """
    Saves experiment data to a CSV file in the experiment folder.

    @param train_population: Fitness values of the training population as a NumPy array.
    @param test_population: Fitness values of the test population as a NumPy array.
    """

    if not args.expr_file:
        return

    experiment_file = get_experiment_file_name()

    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)
    write_header = False

    expr_path = os.path.join(EXPERIMENT_FOLDER, experiment_file)
    if not os.path.exists(expr_path):
        write_header = True

    with open(expr_path, mode="a", newline="", encoding="utf-8") as expr_file:
        population_data = process_population(train_population, test_population)

        writer = csv.DictWriter(expr_file, fieldnames=population_data.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(population_data)


def save_gen_info(generation: int, best: float, min: float, avg: float, std: float) -> None:
    """
    Saves generation statistics to a CSV file in the experiment folder.

    @param generation: The current generation number.
    @param best: The best fitness value in the generation.
    @param min: The minimum fitness value in the generation.
    @param avg: The average fitness value in the generation.
    @param std: The standard deviation of fitness values in the generation.
    """

    if not args.gen_file:
        return

    gen_info_data = {
        "generation": generation,
        "best_fitness": best,
        "min_fitness": min,
        "avg_fitness": avg,
        "std_fitness": std,
    }

    gen_info_file = get_gen_info_file_name()

    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)
    write_header = False

    gen_path = os.path.join(EXPERIMENT_FOLDER, gen_info_file)
    if not os.path.exists(gen_path):
        write_header = True

    with open(gen_path, mode="a", newline="", encoding="utf-8") as gen_file:

        writer = csv.DictWriter(gen_file, fieldnames=gen_info_data.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(gen_info_data)
