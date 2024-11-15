import csv

import numpy as np
from numpy import ndarray
from sklearn.preprocessing import normalize

from config import Config

DATASET_PATHS = {
    "cancer": {"train": "data/breast_cancer_coimbra_train.csv", "test": "data/breast_cancer_coimbra_test.csv"},
    "wine": {"train": "data/wineRed-train.csv", "test": "data/wineRed-test.csv"},
}

args = Config.get_args()


def load_data(dataset_name: str) -> tuple[ndarray, ndarray, ndarray, ndarray]:
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

    X_train_normalized = normalize(X_train, norm="l2")
    X_test_normalized = normalize(X_test, norm="l2")

    num_samples = len(X_train)
    individual_size = len(X_train[0])
    precomputed_x_dicts = [{f"x_{k}": X_train[i][k] for k in range(individual_size)} for i in range(num_samples)]
    precomputed_y_dicts = [{f"y_{k}": X_train[j][k] for k in range(individual_size)} for j in range(num_samples)]

    args.precomputed_x_dicts = precomputed_x_dicts
    args.precomputed_y_dicts = precomputed_y_dicts

    return X_train_normalized, y_train, X_test_normalized, y_test
