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
CONSTANT = [f"{number:.3}" for number in np.arange(-2, 2, 0.05)]
LEN_OPERATOR = len(OPERATOR)
LEN_CONSTANT = len(CONSTANT)


class Individual:
    def __init__(self, genotype=None, depth=1):
        self.fitness = 0
        self.depth = depth - 1
        self.max_depth = self.depth
        self.genotype = self.genotype_vector(genotype)
        self.phenotype = self.phenotype_expr(self.genotype)

    def update_genotype(self, i, value):
        self.fitness = 0
        self.depth = self.max_depth
        self.genotype[i] = value
        self.phenotype = self.phenotype_expr(self.genotype)

    def genotype_vector(self, genotype=None):
        if genotype is not None:
            return genotype

        g_vector = []

        for _ in range(self.depth + 1):
            g_vector.append(random.randrange(0, LEN_CONSTANT))
            g_vector.append(random.randrange(0, LEN_OPERATOR))

        return g_vector

    def phenotype_expr(self, genotype):
        genotype = genotype[::-1]

        phenotype = self.terminal(genotype[0:2])
        for i in range(2, len(genotype), 2):
            phenotype = self.terminal(genotype[i : i + 2]) + " + " + phenotype

        return phenotype

    def terminal(self, numbers):
        index = self.depth
        self.depth -= 1

        const = self.constant(numbers[1])
        op = self.operator(numbers[0])

        if op == "/":
            return f"{const} {op} (abs(xi_{index} - xj_{index}) + 1e-6)"

        return f"{const} {op} abs(xi_{index} - xj_{index})"

    def operator(self, number):
        return OPERATOR[number]

    def constant(self, number):
        return CONSTANT[number]


def generate_initial_population(size, individual_size):
    return [Individual(depth=individual_size) for _ in range(size)]


def evaluate_fitness(individual, X, y, max_individual_size):
    try:
        num_samples = len(X)
        distances = np.zeros((num_samples, num_samples))

        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                distance = safe_eval(
                    individual.phenotype,
                    {
                        **{f"xi_{k}": X[i][k] for k in range(max_individual_size)},
                        **{f"xj_{k}": X[j][k] for k in range(max_individual_size)},
                    },
                )
                distances[i, j] = distance
                distances[j, i] = distance

        clustering = AgglomerativeClustering(n_clusters=len(set(y)), metric="precomputed", linkage="complete")
        labels = clustering.fit_predict(distances)

        individual.fitness = v_measure_score(y, labels)
    except ZeroDivisionError:
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
    except (ZeroDivisionError, Exception):
        return 0


def crossover(parent1, parent2, crossover_prob):
    if random.random() < crossover_prob:
        i = random.randrange(0, len(parent1.genotype) - 1)
        j = random.randrange(i + 1, len(parent1.genotype))

        genotype1 = parent1.genotype[:i] + parent2.genotype[i:j] + parent1.genotype[j:]
        genotype2 = parent2.genotype[:i] + parent1.genotype[i:j] + parent2.genotype[j:]
        return Individual(genotype1), Individual(genotype2)

    return parent1, parent2


def mutate(individual, mutation_prob):
    if random.random() < mutation_prob:
        i = random.randrange(0, len(individual.genotype))
        if i % 2 == 0:
            new_value = random.randrange(0, LEN_CONSTANT)
        else:
            new_value = random.randrange(0, LEN_OPERATOR)
        individual.update_genotype(i, new_value)
    return individual


def genetic_programming(
    X, y, population_size, generations, crossover_prob, mutation_prob, elitism, max_individual_size, use_multithreading
):  
    population = generate_initial_population(population_size, max_individual_size)

    for generation in range(generations):
        if use_multithreading:
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(evaluate_fitness, ind, X, y, max_individual_size): ind for ind in population}
                for future in as_completed(futures):
                    ind = futures[future]
                    ind.fitness = future.result()
        else:
            for ind in population:
                ind.fitness = evaluate_fitness(ind, X, y, max_individual_size)
        for p in population:
            print(p.genotype)

        best_fitness = np.max([ind.fitness for ind in population if ind.fitness is not None])
        worst_fitness = np.min([ind.fitness for ind in population if ind.fitness is not None])
        avg_fitness = np.mean([ind.fitness for ind in population if ind.fitness is not None])
        print(f"Geração {generation}: Melhor {best_fitness:.3}, Pior {worst_fitness:.3}, Média {avg_fitness:.3}")

        new_population = []
        for _ in range(len(population) // 2):
            parents1 = tournament_selection(population)
            child1, child2 = crossover(parents1[0], parents1[1], crossover_prob)
            new_population.extend([mutate(child1, mutation_prob), mutate(child2, mutation_prob)])

        if elitism:
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            new_population = population[:2] + new_population

        population.sort(key=lambda ind: ind.fitness, reverse=True)
        population = new_population[: len(population)]


    population.sort(key=lambda ind: ind.fitness, reverse=True)
    return population[0]


def tournament_selection(population):
    k = max(2, len(population) // 10)
    selected = random.sample(population, k)
    selected.sort(key=lambda ind: ind.fitness, reverse=True)
    return selected[:2]


def test(X, y, population, max_individual_size, use_multithreading):
    if use_multithreading:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(evaluate_fitness, ind, X, y, max_individual_size): ind for ind in population}
            for future in as_completed(futures):
                ind = futures[future]
                ind.fitness = future.result()
    else:
        for ind in population:
            ind.fitness = evaluate_fitness(ind, X, y, max_individual_size)

    best_fitness = np.max([ind.fitness for ind in population if ind.fitness is not None])
    print(f"Geração test: Melhor {best_fitness:.3}")


def main():
    parser = argparse.ArgumentParser(description="Configurações do algoritmo de programação genética")
    parser.add_argument("--population_size", type=int, default=30, help="Tamanho da população")
    parser.add_argument("--generations", type=int, default=30, help="Número de gerações")
    parser.add_argument("--crossover_prob", type=float, default=0.9, help="Probabilidade de crossover")
    parser.add_argument("--mutation_prob", type=float, default=0.05, help="Probabilidade de mutação")
    parser.add_argument(
        "--elitism", type=int, default=1, help="Se deve usar elitismo (0 para não, qualquer outro número para sim)"
    )
    parser.add_argument("--individual_size", type=int, help="Tamanho máximo dos indivíduos (tamanho de X_train)")
    parser.add_argument("--seed", type=int, help="Semente para o gerador de números aleatórios")
    parser.add_argument(
        "--use_multithreading",
        type=int,
        default=0,
        help="Usar múltiplas threads para avaliação de fitness (0 para não, qualquer outro número para sim)",
    )
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
        args.individual_size,
        args.use_multithreading != 0,  # Convert the integer to a boolean
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

    test(X_test, y_test, [best_individual], args.individual_size, args.use_multithreading != 0)


if __name__ == "__main__":
    main()
