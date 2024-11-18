from copy import deepcopy

from config import Config
from data import load_data
from experiment import save_info_experiment
from genetic_operations import genetic_programming_test, genetic_programming_train


def main():
    args = Config.get_args()

    X_train, y_train, X_test, y_test = load_data(args.dataset)
    train_population = genetic_programming_train(X_train, y_train)
    test_population = genetic_programming_test(X_test, y_test, deepcopy(train_population))
    save_info_experiment(train_population, test_population)


if __name__ == "__main__":
    main()
