import numpy as np
from numpy import ndarray
from numpy.random import randint

from config import LEN_OPERATOR, OPERATOR


class Individual:
    def __init__(self, genotype: ndarray = None, depth: int = 5, individual_size: int = 1):
        self.fitness = None
        self.genotype_len = 2**depth - 1
        self.genotype = self.genotype_vector(genotype, individual_size)
        self.phenotype = self.phenotype_expr()

    def update_genotype(self, genotype: ndarray) -> None:
        self.fitness = None
        self.genotype = genotype
        self.phenotype = self.phenotype_expr()

    def genotype_vector(self, genotype: ndarray = None, len_terminal: int = 1) -> ndarray:
        if genotype is not None:
            return np.array(genotype)

        genotype_vector = np.ones((self.genotype_len,), dtype=np.int32)

        for i in range(self.genotype_len):
            if i >= self.genotype_len // 2:
                genotype_vector[i] = randint(len_terminal)
            else:
                genotype_vector[i] = randint(LEN_OPERATOR)

        return genotype_vector

    def get_expr_node(self, i: int) -> str:
        if i < self.genotype_len // 2:
            op = self.operator(self.genotype[i])
            right = self.get_expr_node(2 * i + 1)
            left = self.get_expr_node(2 * i + 2)

            if op == "/":
                return f"{left} {op} ({right} + 1e-6)"

            return f"({left} {op} {right})"
        else:
            return self.terminal(i)

    def phenotype_expr(self) -> str:

        return self.get_expr_node(0)

    def terminal(self, i: int) -> str:
        index = self.genotype[i]
        return f"abs(x_{index} - y_{index})"

    def operator(self, number: int) -> str:
        return OPERATOR[number]

    def __gt__(self, other) -> bool:
        if self.fitness == None:
            return False

        if other.fitness == None:
            return True

        return self.fitness > other.fitness
