import csv
import random

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import completeness_score, homogeneity_score

# Configurações globais
MAX_INDIVIDUAL_SIZE = 9
POPULATION_SIZES = [30, 50, 100, 500]
GENERATIONS = [30, 50, 100, 500]
CROSSOVER_PROBABILITIES = [0.9, 0.6]
MUTATION_PROBABILITIES = [0.05, 0.3]
ELITISM = True

# random.seed(137)

OPERATOR = ["+", "-", "*", "/"]
CONSTANT = [f"{number:.3}" for number in np.arange(-2, 2, 0.05)]
LEN_OPERATOR = len(OPERATOR)
LEN_CONSTANT = len(CONSTANT)


# 1. Representação de um Indivíduo
class Individual:
    def __init__(self, genotype=None, depth=MAX_INDIVIDUAL_SIZE - 1):
        self.fitness = None
        self.depth = depth
        self.genotype = self.genotype_vector(genotype)
        self.phenotype = self.phenotype_expr(self.genotype)

    def update_gen(self):
        self.depth = MAX_INDIVIDUAL_SIZE - 1
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

        phenotype = self.terminal(genotype[0 : 2])
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


# 2. Geração da População Inicial
def generate_initial_population(size):
    return [Individual() for _ in range(size)]


# 3. Função de Avaliação (Fitness)
def evaluate_fitness(individual, X, y):
    try:
        # Calcula a matriz de distâncias usando a expressão do indivíduo
        distances = np.array(
            [
                [
                    safe_eval(
                        individual.phenotype,
                        {
                            **{f"xi_{i}": xi[i] for i in range(MAX_INDIVIDUAL_SIZE)},
                            **{f"xj_{i}": xj[i] for i in range(MAX_INDIVIDUAL_SIZE)},
                        },
                    )
                    for xj in X
                ]
                for xi in X
            ]
        )

        # print(individual.genotype)
        # print(individual.phenotype)
        # print()


        clustering = AgglomerativeClustering(n_clusters=len(set(y)), metric="precomputed", linkage="average")
        labels = clustering.fit_predict(distances)
        homogeneity = float(homogeneity_score(y, labels))
        completeness = float(completeness_score(y, labels))
        beta = 1
        individual.fitness = (1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness)
    except Exception as e:
        individual.fitness = 0  # Penaliza expressões que falham
    return individual.fitness


# Função para avaliar expressões de forma segura, evitando divisão por zero
def safe_eval(expression, vars_dict):
    try:
        result = eval(expression, {}, vars_dict)
        if np.isfinite(result):
            return result
        else:
            return 0
    except ZeroDivisionError as e:
        print(e)
        return 0
    except Exception as e:
        print(e)
        return 0


# 4. Operadores Genéticos
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_PROBABILITIES[0]:
        i = random.randrange(0, len(parent1.genotype))

        genotype = parent1.genotype[:i] + parent2.genotype[i:]
        return Individual(genotype)
    
    return parent1


def mutate(individual):
    if random.random() < MUTATION_PROBABILITIES[0]:
        i = random.randrange(0, len(individual.genotype))

        new_value = None
        if i % 2 == 0:
            new_value = random.randrange(0, LEN_CONSTANT)
        else:
            new_value = random.randrange(0, LEN_OPERATOR)

        individual.genotype[i] = new_value
        individual.update_gen()
    return individual


# 5. Algoritmo de Programação Genética
def genetic_programming(X, y):
    population = generate_initial_population(POPULATION_SIZES[2])

    # for p in population:
    #     print(p.genotype)
    #     print(p.phenotype)
    #     print()

    # return
    for generation in range(GENERATIONS[0]):
        for individual in population:
            evaluate_fitness(individual, X, y)

        # for p in population:
            # print(p.genotype)
            # print(p.phenotype)
            # print()

        new_population = []
        for _ in range(len(population) // 2):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.extend([mutate(child1), mutate(child2)])

        if ELITISM:
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            new_population = population[: len(population) // 10] + new_population

        population = new_population[: len(population)]

        best_fitness = max(ind.fitness for ind in population if ind.fitness is not None)
        worst_fitness = min(ind.fitness for ind in population if ind.fitness is not None)
        avg_fitness = np.mean([ind.fitness for ind in population if ind.fitness is not None])
        print(f"Geração {generation}: Melhor {best_fitness}, Pior {worst_fitness}, Média {avg_fitness}")


# 6. Seleção por Torneio
def tournament_selection(population, k=2):
    selected = random.sample(population, k)
    selected.sort(key=lambda ind: ind.fitness, reverse=True)
    return selected[0]


# Exemplo de execução
if __name__ == "__main__":
    n = 7  # Número de variáveis por elemento (atualizado para 7)
    X_train = []
    y_train = []

    # Carregando os dados do CSV
    with open("data/breast_cancer_coimbra_test.csv", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            values = list(row.values())
            X_train.append([float(v) for v in values[:-1]])  # Convertendo para float
            y_train.append(float(values[-1]))  # Supondo que os rótulos são valores float

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    genetic_programming(X_train, y_train)
