from time import monotonic as time

import numpy as np

from experiment import get_generation_info, save_gen_info
from individual import Individual


def time_info(time: float) -> None:
    """
    Prints the elapsed time of a process.

    @param time: Elapsed time in seconds.
    """
    print(f"\n[{time:.6f}s]", end=" ")


def generation_info(best: float, min: float, avg: float, std: float) -> None:
    """
    Prints statistical information about a generation.

    @param best: Best fitness in the population.
    @param min: Worst fitness in the population.
    @param avg: Average fitness in the population.
    @param std: Standard deviation of fitness in the population.
    """
    print(f"BEST {best:7.4f}% ", end="| ")
    print(f"WORSE {min:7.4f}% ", end="| ")
    print(f"AVG {avg:7.4f}% ", end="| ")
    print(f"STD {std:7.4f}%", end="")


def print_train_info(population: list[Individual], generation: int, instant: float) -> None:
    """
    Prints and saves training generation information, including fitness statistics and elapsed time.

    @param population: List of individuals in the current population.
    @param generation: Current generation number.
    @param instant: Starting time of the generation.
    """
    population_fitness = np.array([float(ind) for ind in population])

    best_fitness, min_fitness, avg_fitness, std_fitness = get_generation_info(population_fitness)
    save_gen_info(generation, best_fitness, min_fitness, avg_fitness, std_fitness)

    gen_time = time() - instant
    time_info(gen_time)
    print(f"GENERATION {generation:04d} ", end="| ")
    generation_info(best_fitness, min_fitness, avg_fitness, std_fitness)


def print_test_info(population: list[Individual], instant: float) -> None:
    """
    Prints and saves test generation information, including fitness statistics and elapsed time.

    @param population: List of individuals in the current population.
    @param instant: Starting time of the generation.
    """
    population_fitness = np.array([float(ind) for ind in population])

    best_fitness, min_fitness, avg_fitness, std_fitness = get_generation_info(population_fitness)
    save_gen_info(-1, best_fitness, min_fitness, avg_fitness, std_fitness)

    gen_time = time() - instant
    time_info(gen_time)
    print(f"GENERATION TEST ", end="| ")
    generation_info(best_fitness, min_fitness, avg_fitness, std_fitness)
    print()
