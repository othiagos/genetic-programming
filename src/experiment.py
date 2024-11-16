import csv
import os

import numpy as np
from numpy import ndarray

from config import Config
from info import get_generation_info

args = Config.get_args()

EXPERIMENT_FOLDER = "experiment/"
EXPERIMENT_FILE = args.expr_file if args.expr_file is None else args.expr_file + ".csv"


def process_population(train_population: ndarray, test_population: ndarray) -> dict:
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


def save_info_experiment(train_population: ndarray, test_population: ndarray) -> None:

    if EXPERIMENT_FILE is None:
        return

    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)
    write_header = False

    expr_path = os.path.join(EXPERIMENT_FOLDER, EXPERIMENT_FILE)
    if not os.path.exists(expr_path):
        write_header = True

    with open(expr_path, mode="a", newline="", encoding="utf-8") as expr_file:
        population_data = process_population(train_population, test_population)

        writer = csv.DictWriter(expr_file, fieldnames=population_data.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(population_data)
