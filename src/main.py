from config import Config
from data import load_data
from genetic_operations import genetic_programming_test, genetic_programming_train

def main():

    args = Config.get_args()

    X_train, y_train, X_test, y_test = load_data(args.dataset)
    args.individual_size = len(X_train[0])
    args.multithreading = args.multithreading != 0

    train_population = genetic_programming_train(X_train, y_train)
    genetic_programming_test(X_test, y_test, train_population)


if __name__ == "__main__":
    main()
