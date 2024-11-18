import argparse
from argparse import Namespace
from time import monotonic_ns as time

import numpy as np

# Constants
OPERATOR = ["+", "-", "*", "/"]
LEN_OPERATOR = len(OPERATOR)


class Config:
    """
    Singleton class to manage configuration arguments.

    @method get_args: Retrieves the parsed arguments as a Namespace.
    """

    _instance = None

    @classmethod
    def get_args(self) -> Namespace:
        """
        Retrieves the parsed command-line arguments.

        @return Namespace: Parsed arguments.
        """
        if self._instance is None:
            self._instance = parser.parse_args()
        return self._instance


def set_seed(seed: int) -> None:
    """
    Sets the random seed for NumPy's random number generator.

    @param seed: Seed value for reproducibility.
    """
    np.random.seed(seed)


help_description = "Genetic programming algorithm configurations"
help_population_size = "Population size"
help_generations = "Number of generations"
help_mutation_prob = "Mutation probability"
help_depth = "Maximum tree depth of the individuals"
help_seed = "Seed for the random number generator"
help_multithreading = "Use multiple threads for fitness evaluation (0 for no, any value for yes)"
help_tournament = "Number of individuals to select in the tournament"
help_dataset = "Choose the dataset: 'cancer' or 'wine'"
help_csv_file = "Name of the CSV file where results will be saved (optional)"
help_gen_csv_file = "Name of the CSV file to save generation results (optional)"


parser = argparse.ArgumentParser(description=help_description)
parser.add_argument("--population_size", type=int, default=30, help=help_population_size)
parser.add_argument("--generations", type=int, default=30, help=help_generations)
parser.add_argument("--mutation_prob", type=float, default=0.05, help=help_mutation_prob)
parser.add_argument("--depth", type=int, help=help_depth)
parser.add_argument("--seed", type=int, help=help_seed)
parser.add_argument("--multithreading", type=int, default=1, help=help_multithreading)
parser.add_argument("--tournament", type=int, default=3, help=help_tournament)
parser.add_argument("--dataset", type=str, choices=["cancer", "wine"], required=True, help=help_dataset)
parser.add_argument("--expr_file", action="store_true", help=help_csv_file)
parser.add_argument("--gen_file", action="store_true", help=help_gen_csv_file)


args = Config.get_args()
args.multithreading = args.multithreading != 0

if args.seed is None:
    args.seed = time() % 2**32 - 1

args.seed_main = args.seed
set_seed(args.seed_main)
