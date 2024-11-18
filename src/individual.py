import numpy as np
from numpy import ndarray
from numpy.random import randint

from config import LEN_OPERATOR, OPERATOR


class Individual:
    """
    Represents an individual in a genetic programming algorithm, with a genotype, phenotype, and fitness.

    @param genotype: Initial genotype as a NumPy array. If None, a random genotype is generated.
    @param depth: Depth of the binary tree used for the individual's expression. Default is 5.
    @param individual_size: Size of the individual (number of terminal variables). Default is 1.
    """

    def __init__(self, genotype: ndarray = None, depth: int = 5, individual_size: int = 1):
        self.fitness = None
        self.genotype_len = 2**depth - 1
        self.genotype = self.genotype_vector(genotype, individual_size)
        self.phenotype = self.phenotype_expr()

    def update_genotype(self, genotype: ndarray) -> None:
        """
        Updates the genotype and recalculates the phenotype.

        @param genotype: New genotype as a NumPy array.
        """
        self.fitness = None
        self.genotype = genotype
        self.phenotype = self.phenotype_expr()

    def genotype_vector(self, genotype: ndarray = None, len_terminal: int = 1) -> ndarray:
        """
        Generates or validates a genotype vector.

        @param genotype: Existing genotype as a NumPy array. If None, generates a random genotype.
        @param len_terminal: Number of terminal variables available. Default is 1.
        @return: Genotype vector as a NumPy array.
        """
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
        """
        Recursively constructs the expression tree for the phenotype.

        @param i: Index of the current node in the binary tree.
        @return: String representation of the expression for the current node.
        """
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
        """
        Constructs the phenotype expression based on the genotype.

        @return: Phenotype expression as a string.
        """
        return self.get_expr_node(0)

    def terminal(self, i: int) -> str:
        """
        Retrieves the terminal node expression for a given index.

        @param i: Index of the terminal node.
        @return: Terminal node expression as a string.
        """
        index = self.genotype[i]
        return f"abs(x_{index} - y_{index})"

    def operator(self, number: int) -> str:
        """
        Maps a number to the corresponding operator.

        @param number: Index of the operator.
        @return: Operator as a string.
        """
        return OPERATOR[number]

    def __gt__(self, other) -> bool:
        """
        Compares two individuals based on their fitness.

        @param other: Another individual.
        @return: True if the current individual's fitness is greater, False otherwise.
        """
        if self.fitness == None:
            return False

        if other.fitness == None:
            return True

        return self.fitness > other.fitness

    def __float__(self) -> float:
        """
        Converts the individual's fitness to a float.

        @return: Fitness value as a float.
        """
        return self.fitness
