
from time import monotonic as time
import numpy as np
from individual import Individual


def get_generation_info(population_fitness: np.ndarray) -> tuple[float, float, float, float]:
    best_fitness = np.max(population_fitness) * 100
    min_fitness = np.min(population_fitness) * 100
    avg_fitness = np.mean(population_fitness) * 100
    std_fitness = np.std(population_fitness) * 100

    return best_fitness, min_fitness, avg_fitness, std_fitness


def time_info(time: float) -> None:
    print(f"\n[{time:.6f}s]", end=" ")


def generation_info(best: float, min: float, avg: float, std: float) -> None:
    print(f"BEST {best:7.4f}% ", end="| ")
    print(f"WORSE {min:7.4f}% ", end="| ")
    print(f"AVG {avg:7.4f}% ", end="| ")
    print(f"STD {std:7.4f}%", end="")


def print_train_info(population: list[Individual], generation: int, instant: float) -> None:
    population_fitness = np.array([float(ind) for ind in population])

    best_fitness, min_fitness, avg_fitness, std_fitness = get_generation_info(population_fitness)

    gen_time = time() - instant
    time_info(gen_time)
    print(f"GENERATION {generation:04d} ", end="| ")
    generation_info(best_fitness, min_fitness, avg_fitness, std_fitness)


def print_test_info(population: list[Individual], instant: float) -> None:
    population_fitness = np.array([float(ind) for ind in population])

    best_fitness, min_fitness, avg_fitness, std_fitness = get_generation_info(population_fitness)

    gen_time = time() - instant
    time_info(gen_time)
    print(f"GENERATION TEST ", end="| ")
    generation_info(best_fitness, min_fitness, avg_fitness, std_fitness)
    print()