import argparse
import csv
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import v_measure_score

# Constants
OPERATOR = ["+", "-", "*", "/"]
LEN_OPERATOR = len(OPERATOR)


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

        genotype_vector = np.ones((self.genotype_len,), dtype=np.int8)

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


def share_distance(individual1, individual2):
    distance = 0

    for i in range(len(individual1.genotype)):
        if individual1.genotype[i] != individual2.genotype[i]:
            distance += 1

    return distance


def fitness_sharing(individual, population):
    share_threshold = individual.genotype_len // 2

    sum_sh = 0
    for ind in population:
        diff = share_distance(individual, ind)

        if diff < share_threshold:
            sum_sh += 1 - (diff / share_threshold)

    return sum_sh


def evaluate_fitness(individual, X, y, individual_size):
    try:
        # Check if need calc individual fitness
        if individual.fitness != None:
            return individual.fitness

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

        clustering = AgglomerativeClustering(n_clusters=len(set(y)), metric="precomputed", linkage="average")
        labels = clustering.fit_predict(matrix_distances)

        individual.fitness = v_measure_score(y, labels)
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
    selected = random.sample(population, k)
    selected.sort(reverse=True)
    return selected[0]


def crossover(parent1, parent2, crossover_prob, depth, individual_size):
    if random.random() < crossover_prob:

        swap_index = [random.randrange(0, parent1.genotype_len)]
        genotype_len = parent1.genotype_len
        genotype1 = parent1.genotype
        genotype2 = parent2.genotype

        while len(swap_index) > 0:
            i = swap_index.pop(0)

            if 2 * i + 2 < genotype_len:
                swap_index.append(2 * i + 1)
                swap_index.append(2 * i + 2)

            genotype1[i], genotype2[i] = genotype2[i], genotype1[i]

        return Individual(genotype1, depth, individual_size), Individual(genotype2, depth, individual_size)

    return parent1, parent2


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


def population_evaluate_fitness(population, X, y, individual_size, use_multithreading=False):
    if use_multithreading:
        # Fitness
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(evaluate_fitness, ind, X, y, individual_size): ind for ind in population}
            for future in as_completed(futures):
                ind = futures[future]
                ind.fitness = future.result()

        # Fitness sharing
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(fitness_sharing, ind, population): ind for ind in population if ind.fitness != 0}
            for future in as_completed(futures):
                ind = futures[future]
                nc = future.result()
                ind.fitness /= nc if nc != 0 else 1
    else:
        # Fitness
        for ind in population:
            ind.fitness = evaluate_fitness(ind, X, y, individual_size)

        # Fitness sharing
        for ind in population:
            if ind.fitness != 0:
                nc = fitness_sharing(ind, population)
                ind.fitness /= nc if nc != 0 else 1


def genetic_programming(
    X,
    y,
    population_size,
    generations,
    crossover_prob,
    mutation_prob,
    elitism,
    depth,
    individual_size,
    use_multithreading,
    k,
):
    population = generate_initial_population(population_size, depth, individual_size)

    population_evaluate_fitness(population, X, y, individual_size, use_multithreading)

    # for ind in population:
    #     print(ind.genotype)
    #     print(ind.phenotype)
    #     print()
    # return
    for generation in range(generations):

        best_fitness = float(np.max([ind.fitness for ind in population if ind.fitness is not None]))
        avg_fitness = float(np.mean([ind.fitness for ind in population if ind.fitness is not None]))
        print(f"Geração {generation}: Melhor {best_fitness:.3}, Média {avg_fitness:.3}")

        new_population = []
        for _ in range(len(population) // 2):
            parent1 = tournament_selection(population, k)
            parent2 = tournament_selection(population, k)
            child1, child2 = crossover(parent1, parent2, crossover_prob, depth, individual_size)
            child1 = mutate(child1, mutation_prob, individual_size)
            child2 = mutate(child2, mutation_prob, individual_size)

            new_population.extend([child1, child2])

        if elitism:
            population.sort(reverse=True)
            new_population = population[:2] + new_population

        new_population.sort(reverse=True)
        population = new_population[:population_size]

        population_evaluate_fitness(new_population, X, y, individual_size, use_multithreading)
    population.sort(reverse=True)
    return population[0]


def test(X, y, population, individual_size, use_multithreading):
    if use_multithreading:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(evaluate_fitness, ind, X, y, individual_size): ind for ind in population}
            for future in as_completed(futures):
                ind = futures[future]
                ind.fitness = future.result()
    else:
        for ind in population:
            ind.fitness = evaluate_fitness(ind, X, y, individual_size)

    best_fitness = np.max(population)
    print(f"Geração test: Melhor {best_fitness.fitness:.3}")


def main():
    help_description = "Configurações do algoritmo de programação genética"
    help_population_size = "Tamanho da população"
    help_generations = "Número de gerações"
    help_crossover_prob = "Probabilidade de crossover"
    help_mutation_prob = "Probabilidade de mutação"
    help_elitism = "Se deve usar elitismo (0 para não, qualquer outro número para sim)"
    help_depth = "Tamanho máximo da árvores dos indivíduos"
    help_seed = "Semente para o gerador de números aleatórios"
    help_multithreading = "Usar múltiplas threads para avaliação de fitness (0 para não, qualquer valor para sim)"
    help_tournament = "Número de indivíduos a serem selecionados no torneio"

    parser = argparse.ArgumentParser(description=help_description)
    parser.add_argument("--population_size", type=int, default=30, help=help_population_size)
    parser.add_argument("--generations", type=int, default=30, help=help_generations)
    parser.add_argument("--crossover_prob", type=float, default=0.9, help=help_crossover_prob)
    parser.add_argument("--mutation_prob", type=float, default=0.05, help=help_mutation_prob)
    parser.add_argument("--elitism", type=int, default=1, help=help_elitism)
    parser.add_argument("--depth", type=int, help=help_depth)
    parser.add_argument("--seed", type=int, help=help_seed)
    parser.add_argument("--multithreading", type=int, default=0, help=help_multithreading)
    parser.add_argument("--tournament", type=int, default=3, help=help_tournament)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    else:
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))

    # Carregar os dados de treino
    X_train, y_train = [], []
    with open("data/breast_cancer_coimbra_train.csv", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            values = list(row.values())
            X_train.append([float(v) for v in values[:-1]])
            y_train.append(float(values[-1]))

    individual_size = len(X_train[0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Executar o algoritmo de programação genética
    best_individual = genetic_programming(
        X_train,
        y_train,
        args.population_size,
        args.generations,
        args.crossover_prob,
        args.mutation_prob,
        args.elitism != 0,  # Convert the integer to a boolean
        args.depth,
        individual_size,
        args.multithreading != 0,  # Convert the integer to a boolean
        args.tournament,
    )

    # Carregar os dados de teste
    X_test, y_test = [], []
    with open("data/breast_cancer_coimbra_test.csv", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            values = list(row.values())
            X_test.append([float(v) for v in values[:-1]])
            y_test.append(float(values[-1]))

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    test(X_test, y_test, [best_individual], individual_size, args.multithreading != 0)


if __name__ == "__main__":
    main()
