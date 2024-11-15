import argparse
from argparse import Namespace

import numpy as np

# Constants
OPERATOR = ["+", "-", "*", "/"]
LEN_OPERATOR = len(OPERATOR)

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


def set_seed(seed: int) -> None:
    np.random.seed(seed)


class Config:
    _instance = None

    @classmethod
    def get_args(self) -> Namespace:
        if self._instance is None:
            self._instance = parser.parse_args()
        return self._instance


args = Config.get_args()
set_seed(args.seed)
