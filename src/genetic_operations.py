from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from time import monotonic as time

import numpy as np
from numpy import ndarray
from numpy.random import choice, randint, random
from scipy.spatial.distance import cosine

from config import LEN_OPERATOR, Config, set_seed
from evaluation import evaluate_fitness, population_evaluate_fitness
from individual import Individual
from info import print_test_info, print_train_info

args = Config.get_args()


def generate_initial_population(size: int) -> list[Individual]:
    """
    Generates an initial population of individuals.

    @param size: Number of individuals in the population.
    @return list[Individual]: A list of newly created individuals.
    """
    return [Individual(depth=args.depth, individual_size=args.individual_size) for _ in range(size)]


def tournament_selection(population: list[Individual], k: int) -> Individual:
    """
    Selects the best individual from a random subset of the population.

    @param population: The current population of individuals.
    @param k: Number of individuals to participate in the tournament.
    @return Individual: The best individual from the subset.
    """
    if k > len(population):
        k = len(population)

    selected = choice(population, k, replace=False)
    selected = np.sort(selected)[::-1]
    best = selected[0]
    population.remove(best)
    return best


def crossover(parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
    """
    Performs crossover between two parents to generate two offspring.

    @param parent1: The first parent individual.
    @param parent2: The second parent individual.
    @return tuple[Individual, Individual]: Two offspring resulting from crossover.
    """
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
    """
    Applies mutation to an individual with a certain probability.

    @param individual: The individual to mutate.
    @return Individual: The mutated individual.
    """
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
    """
    Selects the best individuals based on genetic similarity and fitness.

    @param parents: A tuple containing two parent individuals.
    @param children: A tuple containing two child individuals.
    @return list[Individual]: A list of two individuals chosen for the next generation.
    """
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


def process_crossover_crowding(
    X: ndarray, y: ndarray, parent1: Individual, parent2: Individual, seed: int = None
) -> list[Individual]:
    """
    Performs crossover, mutation, and crowding with fitness evaluation.

    @param X: Training data.
    @param y: Labels for the training data.
    @param parent1: The first parent individual.
    @param parent2: The second parent individual.
    @param seed: Random seed for reproducibility.
    @return list[Individual]: Two best individuals selected from parents and offspring.
    """
    if seed is not None:
        set_seed(seed)

    child1, child2 = crossover(parent1, parent2)
    child1.fitness = evaluate_fitness(X, y, child1)
    child2.fitness = evaluate_fitness(X, y, child2)

    return crowding((parent1, parent2), (child1, child2))


def new_generation(X: ndarray, y: ndarray, population: list[Individual]) -> list[Individual]:
    """
    Generates a new population from the current population.

    @param X: Training data.
    @param y: Labels for the training data.
    @param population: The current population of individuals.
    @return list[Individual]: The new population.
    """
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
                args.seed += 1
                fn_args = (X, y, parent1, parent2, args.seed)
                futures.append(executor.submit(process_crossover_crowding, *fn_args))

            for future in futures:
                best_individuals = future.result()
                new_population.extend(best_individuals)

        return new_population

    for _ in range(len(population) // 2):
        parent1 = tournament_selection(population, args.tournament)
        parent2 = tournament_selection(population, args.tournament)

        best_individuals = process_crossover_crowding(X, y, parent1, parent2)
        new_population.extend(best_individuals)

    return new_population


def genetic_programming_train(X: ndarray, y: ndarray) -> list[Individual]:
    """
    Trains the genetic programming algorithm.

    @param X: Training data.
    @param y: Labels for the training data.
    @return list[Individual]: The final population after all generations.
    """
    population = generate_initial_population(args.population_size)

    gen_start_time = time()
    population_evaluate_fitness(X, y, population)

    for generation in range(args.generations):

        print_train_info(population, generation, gen_start_time)

        gen_start_time = time()

        population = new_generation(X, y, population)

    return population


def genetic_programming_test(X: ndarray, y: ndarray, population: list[Individual]) -> None:
    """
    Evaluates the final population on test data and prints results.

    @param X: Test data.
    @param y: Labels for the test data.
    @param population: The population of individuals to evaluate.
    """
    gen_start_time = time()

    for ind in population:
        ind.fitness = None

    population_evaluate_fitness(X, y, population)

    print_test_info(population, gen_start_time)

    return population
