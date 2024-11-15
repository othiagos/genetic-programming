from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from time import monotonic as time

import numpy as np
from numpy import ndarray
from numpy.random import choice, randint, random
from scipy.spatial.distance import cosine

from config import LEN_OPERATOR, Config
from evaluation import evaluate_fitness, population_evaluate_fitness
from individual import Individual

args = Config.get_args()


def generate_initial_population(size: int) -> list[Individual]:
    return [Individual(depth=args.depth, individual_size=args.individual_size) for _ in range(size)]


def tournament_selection(population: list[Individual], k: int) -> Individual:
    if k > len(population):
        k = len(population)

    selected = choice(population, k, replace=False)
    selected = np.sort(selected)[::-1]
    best = selected[0]
    population.remove(best)
    return best


def crossover(parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
    swap_index = [randint(parent1.genotype_len)]
    genotype_len = parent1.genotype_len
    genotype1 = deepcopy(parent1.genotype)
    genotype2 = deepcopy(parent2.genotype)

    while len(swap_index) > 0:
        i = swap_index.pop(0)

        if 2 * i + 2 < genotype_len:
            swap_index.append(2 * i + 1)
            swap_index.append(2 * i + 2)

        genotype1[i], genotype2[i] = genotype2[i], genotype1[i]

    child1 = mutate(Individual(genotype1, args.depth, args.individual_size))
    child2 = mutate(Individual(genotype2, args.depth, args.individual_size))

    return child1, child2


def mutate(individual: Individual) -> Individual:
    if random() < args.mutation_prob:

        swap_index = [randint(individual.genotype_len)]
        genotype_len = individual.genotype_len
        genotype = individual.genotype

        while len(swap_index) > 0:
            i = swap_index.pop(0)

            if 2 * i + 2 < genotype_len:
                swap_index.append(2 * i + 1)
                swap_index.append(2 * i + 2)

            if i >= individual.genotype_len // 2:
                genotype[i] = randint(args.individual_size)
            else:
                genotype[i] = randint(LEN_OPERATOR)

        individual.update_genotype(genotype)

    return individual


def crowding(parents: tuple[Individual, Individual], children: tuple[Individual, Individual]) -> list[Individual]:
    best_Individuals = []
    dist1 = cosine(parents[0].genotype, children[0].genotype)
    dist2 = cosine(parents[1].genotype, children[0].genotype)

    if dist1 < dist2:
        if parents[0].fitness < children[0].fitness:
            best_Individuals.append(children[0])
        else:
            best_Individuals.append(parents[0])

        if parents[1].fitness < children[1].fitness:
            best_Individuals.append(children[1])
        else:
            best_Individuals.append(parents[1])
    else:
        if parents[1].fitness < children[0].fitness:
            best_Individuals.append(children[0])
        else:
            best_Individuals.append(parents[1])

        if parents[0].fitness < children[1].fitness:
            best_Individuals.append(children[1])
        else:
            best_Individuals.append(parents[0])

    return best_Individuals


def process_crossover_crowding(X: ndarray, y: ndarray, parent1: Individual, parent2: Individual) -> list[Individual]:
    child1, child2 = crossover(parent1, parent2)
    child1.fitness = evaluate_fitness(X, y, child1)
    child2.fitness = evaluate_fitness(X, y, child2)

    return crowding((parent1, parent2), (child1, child2))


def new_generation(X: ndarray, y: ndarray, population: list[Individual]) -> list[Individual]:
    new_population = []

    if args.multithreading:

        parents = []
        for _ in range(len(population) // 2):
            parent1 = tournament_selection(population, args.tournament)
            parent2 = tournament_selection(population, args.tournament)
            parents.append((parent1, parent2))

        with ProcessPoolExecutor() as executor:
            futures = []

            for parent1, parent2 in parents:
                fn_args = (X, y, parent1, parent2)
                futures.append(executor.submit(process_crossover_crowding, *fn_args))

            for future in as_completed(futures):
                best_individuals = future.result()
                new_population.extend(best_individuals)

        return new_population

    for _ in range(len(population) // 2):
        parent1 = tournament_selection(population, args.tournament)
        parent2 = tournament_selection(population, args.tournament)

        best_individuals = process_crossover_crowding(X, y, parent1, parent2)
        new_population.extend(best_individuals)

    return new_population


def get_generation_info(population_fitness: ndarray) -> tuple[float, float, float, float]:
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


def genetic_programming_train(X: ndarray, y: ndarray) -> list[Individual]:
    population = generate_initial_population(args.population_size)

    gen_start_time = time()
    population_evaluate_fitness(X, y, population)

    for generation in range(args.generations):
        # population.sort(reverse=True)
        # print()
        # for ind in population:
        #     print(ind.genotype)
        #     print(ind.phenotype)
        #     print()

        print_train_info(population, generation, gen_start_time)

        gen_start_time = time()

        new_population = new_generation(X, y, population)
        population = new_population

    return population


def genetic_programming_test(X: ndarray, y: ndarray, population: list[Individual]) -> None:
    gen_start_time = time()

    for ind in population:
        ind.fitness = None

    population_evaluate_fitness(X, y, population)

    print_test_info(population, gen_start_time)
