import argparse
import csv
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import v_measure_score
from sklearn.preprocessing import normalize

# Constants
OPERATOR = ["+", "-", "*", "/"]
LEN_OPERATOR = len(OPERATOR)

DATASET_PATHS = {
    "cancer": {"train": "data/breast_cancer_coimbra_train.csv", "test": "data/breast_cancer_coimbra_test.csv"},
    "wine": {"train": "data/wineRed-train.csv", "test": "data/wineRed-test.csv"},
}


class Individual:
    def __init__(self, genotype=None, depth=5, terminal_size=1):
        self.fitness = None
        self.genotype_len = 2**depth - 1
        self.genotype = self.genotype_vector(genotype, terminal_size)
        self.phenotype = self.phenotype_expr()

    def update_genotype(self, genotype):
        self.fitness = None
        self.genotype = genotype
        self.phenotype = self.phenotype_expr()

    def genotype_vector(self, genotype=None, terminal=None):
        if genotype is not None:
            return np.array(genotype)

        genotype_vector = np.ones((self.genotype_len,), dtype=np.int32)

        for i in range(self.genotype_len):
            if i >= self.genotype_len // 2:
                genotype_vector[i] = random.randrange(0, terminal)
            else:
                genotype_vector[i] = random.randrange(0, LEN_OPERATOR)

        return genotype_vector

    def get_expr_node(self, i):
        if i < self.genotype_len // 2:
            op = self.operator(self.genotype[i])
            right = self.get_expr_node(2 * i + 1)
            left = self.get_expr_node(2 * i + 2)

            if op == "/":
                return f"{left} {op} ({right} + 1e-6)"

            return f"({left} {op} {right})"
        else:
            return self.terminal(i)

    def phenotype_expr(self):

        return self.get_expr_node(0)

    def terminal(self, i):
        index = self.genotype[i]
        return f"abs(x_{index} - y_{index})"

    def operator(self, number):
        return OPERATOR[number]

    def __gt__(self, other):
        if self.fitness == None:
            return False

        if other.fitness == None:
            return True

        return self.fitness > other.fitness


def generate_initial_population(size, depth, individual_size):
    return [Individual(depth=depth, terminal_size=individual_size) for _ in range(size)]


def calculate_distance_matrix(X, individual, individual_size):
    num_samples = len(X)
    matrix_distances = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            distance = safe_eval(
                individual.phenotype,
                {
                    **{f"x_{k}": X[i][k] for k in range(individual_size)},
                    **{f"y_{k}": X[j][k] for k in range(individual_size)},
                },
            )
            matrix_distances[i, j] = distance
            matrix_distances[j, i] = distance

    return matrix_distances


def evaluate_fitness(individual, X, y, individual_size):
    try:
        if individual.fitness is not None:
            return individual.fitness

        matrix_distances = calculate_distance_matrix(X, individual, individual_size)
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
            print("ing")
            return 0
    except Exception as e:
        print(expression)
        print(f"Erro na expressão: {e}")
        return 0


def tournament_selection(population, k):
    if k > len(population):
        k = len(population)

    selected = random.sample(population, k)
    selected.sort(reverse=True)
    best = selected[0]
    population.remove(best)
    return best


def crossover(parent1, parent2, mutation_prob, depth, individual_size):
    swap_index = [random.randrange(0, parent1.genotype_len)]
    genotype_len = parent1.genotype_len
    genotype1 = deepcopy(parent1.genotype)
    genotype2 = deepcopy(parent2.genotype)

    while len(swap_index) > 0:
        i = swap_index.pop(0)

        if 2 * i + 2 < genotype_len:
            swap_index.append(2 * i + 1)
            swap_index.append(2 * i + 2)

        genotype1[i], genotype2[i] = genotype2[i], genotype1[i]

    child1 = Individual(genotype1, depth, individual_size)
    child2 = Individual(genotype2, depth, individual_size)

    child1 = mutate(child1, mutation_prob, individual_size)
    child2 = mutate(child2, mutation_prob, individual_size)

    return child1, child2


def mutate(individual, mutation_prob, num_terminal):
    if random.random() < mutation_prob:

        swap_index = [random.randrange(0, individual.genotype_len)]
        genotype_len = individual.genotype_len
        genotype = individual.genotype

        while len(swap_index) > 0:
            i = swap_index.pop(0)

            if 2 * i + 2 < genotype_len:
                swap_index.append(2 * i + 1)
                swap_index.append(2 * i + 2)

            if i >= individual.genotype_len // 2:
                genotype[i] = random.randrange(0, num_terminal)
            else:
                genotype[i] = random.randrange(0, LEN_OPERATOR)

        individual.update_genotype(genotype)

    return individual


def population_evaluate_fitness(population, X, y, individual_size, multithreading=False):
    if multithreading:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(evaluate_fitness, ind, X, y, individual_size): ind for ind in population}
            for future in as_completed(futures):
                ind = futures[future]
                ind.fitness = future.result()

    else:
        for ind in population:
            ind.fitness = evaluate_fitness(ind, X, y, individual_size)


def crowding(parents, children):
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


def process_crossover_and_crowding(X, y, parent1, parent2, mutation_prob, depth, individual_size):
    child1, child2 = crossover(parent1, parent2, mutation_prob, depth, individual_size)

    child1.fitness = evaluate_fitness(child1, X, y, individual_size)
    child2.fitness = evaluate_fitness(child2, X, y, individual_size)

    return crowding((parent1, parent2), (child1, child2))


def new_generation(X, y, population, k, mutation_prob, depth, individual_size, multithreading=False):
    if multithreading:
        with ProcessPoolExecutor() as executor:
            futures = []
            new_population = []
            for _ in range(len(population) // 2):
                parent1 = tournament_selection(population, k)
                parent2 = tournament_selection(population, k)

                args = (X, y, parent1, parent2, mutation_prob, depth, individual_size)
                futures.append(executor.submit(process_crossover_and_crowding, *args))

            for future in as_completed(futures):
                best_individuals = future.result()
                new_population.extend(best_individuals)

        return new_population

    new_population = []

    for _ in range(len(population) // 2):
        parent1 = tournament_selection(population, k)
        parent2 = tournament_selection(population, k)

        best_individuals = process_crossover_and_crowding(X, y, parent1, parent2, mutation_prob, depth, individual_size)
        new_population.extend(best_individuals)

    return new_population


def genetic_programming_train(
    X, y, population_size, generations, mutation_prob, depth, individual_size, multithreading, k
):
    population = generate_initial_population(population_size, depth, individual_size)

    t0 = time.time()
    population_evaluate_fitness(population, X, y, individual_size, multithreading)

    for generation in range(generations):
        # for ind in population:
        #     print(ind.genotype)
        #     print(ind.phenotype)
        #     print()

        best_fitness = float(np.max([ind.fitness for ind in population if ind.fitness is not None]))
        avg_fitness = float(np.mean([ind.fitness for ind in population if ind.fitness is not None]))

        gen_time = time.time() - t0
        print(
            f"\nGeração {generation}: Melhor {best_fitness:.3f}, Média {avg_fitness:.3f}, Tempo: {gen_time:.3f}s",
            end="",
        )

        t0 = time.time()

        new_population = new_generation(X, y, population, k, mutation_prob, depth, individual_size, multithreading)
        new_population.sort(reverse=True)
        population = new_population[:population_size]

        population_evaluate_fitness(population, X, y, individual_size, multithreading)
    population.sort(reverse=True)
    return population


def genetic_programming_test(X, y, population, individual_size, multithreading):
    for ind in population:
        ind.fitness = None

    if multithreading:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(evaluate_fitness, ind, X, y, individual_size): ind for ind in population}
            for future in as_completed(futures):
                ind = futures[future]
                ind.fitness = future.result()
    else:
        for ind in population:
            ind.fitness = evaluate_fitness(ind, X, y, individual_size)

    best_fitness = float(np.max([ind.fitness for ind in population if ind.fitness is not None]))
    avg_fitness = float(np.mean([ind.fitness for ind in population if ind.fitness is not None]))

    print(f"\nGeração test: Melhor {best_fitness:.3f}, Média {avg_fitness:.3f}")


def load_data(dataset_name):
    paths = DATASET_PATHS[dataset_name]

    # Load training data
    X_train, y_train = [], []
    with open(paths["train"], newline="") as csv_file_train:
        reader = csv.DictReader(csv_file_train)
        for row in reader:
            values = list(row.values())
            X_train.append([float(v) for v in values[:-1]])
            y_train.append(float(values[-1]) - 1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Load test data
    X_test, y_test = [], []
    with open(paths["test"], newline="") as csv_file_test:
        reader = csv.DictReader(csv_file_test)
        for row in reader:
            values = list(row.values())
            X_test.append([float(v) for v in values[:-1]])
            y_test.append(float(values[-1]) - 1)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def main():
    help_description = "Configurações do algoritmo de programação genética"
    help_population_size = "Tamanho da população"
    help_generations = "Número de gerações"
    help_mutation_prob = "Probabilidade de mutação"
    help_depth = "Tamanho máximo da árvores dos indivíduos"
    help_seed = "Semente para o gerador de números aleatórios"
    help_multithreading = "Usar múltiplas threads para avaliação de fitness (0 para não, qualquer valor para sim)"
    help_tournament = "Número de indivíduos a serem selecionados no torneio"
    help_dataset = "Escolha da base de dados: 'cancer' ou 'wine'"

    parser = argparse.ArgumentParser(description=help_description)
    parser.add_argument("--population_size", type=int, default=30, help=help_population_size)
    parser.add_argument("--generations", type=int, default=30, help=help_generations)
    parser.add_argument("--mutation_prob", type=float, default=0.05, help=help_mutation_prob)
    parser.add_argument("--depth", type=int, help=help_depth)
    parser.add_argument("--seed", type=int, help=help_seed)
    parser.add_argument("--multithreading", type=int, default=0, help=help_multithreading)
    parser.add_argument("--tournament", type=int, default=3, help=help_tournament)
    parser.add_argument("--dataset", type=str, choices=["cancer", "wine"], required=True, help=help_dataset)

    args = parser.parse_args()

    set_seed(args.seed)

    X_train, y_train, X_test, y_test = load_data(args.dataset)
    individual_size = len(X_train[0])

    X_train_normalized = normalize(X_train, norm="max")
    X_test_normalized = normalize(X_test, norm="max")

    train_population = genetic_programming_train(
        X_train_normalized,
        y_train,
        args.population_size,
        args.generations,
        args.mutation_prob,
        args.depth,
        individual_size,
        args.multithreading != 0,
        args.tournament,
    )

    genetic_programming_test(X_test_normalized, y_test, train_population, individual_size, args.multithreading != 0)


if __name__ == "__main__":
    main()
