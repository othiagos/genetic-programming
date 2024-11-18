import argparse
import csv
import os
from collections import OrderedDict

csv_mean_file = "experiment/DATE_EXPR_MEAN.csv"


def calculate_column_means(csv_file):
    """
    Calculates the mean of each column in a CSV file, excluding the 'seed' column.

    @param csv_file: The path to the CSV file.
    @return dict: A dictionary containing the column names as keys and their respective mean values as values.
    """
    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        columns = {key: [] for key in reader.fieldnames}

        for row in reader:
            for key, value in row.items():
                if key != "seed":  # Ignores the 'seed' column
                    columns[key].append(float(value))

        column_means = {key: sum(values) / len(values) for key, values in columns.items() if key != "seed"}
        return column_means


def save_means_to_csv(input_file, column_means):
    """
    Saves the calculated column means to a CSV file.

    @param input_file: The original input CSV file used for mean calculation.
    @param column_means: A dictionary containing the column names and their calculated mean values.
    """
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    column_means["file"] = base_name

    # Reorganizes so that "file" is the first column
    ordered_means = OrderedDict([("file", column_means.pop("file"))] + list(column_means.items()))

    write_header = False

    if not os.path.exists(csv_mean_file):
        write_header = True

    with open(csv_mean_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ordered_means.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(ordered_means)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str)
    args = parser.parse_args()

    column_means = calculate_column_means(args.csv_file)
    save_means_to_csv(args.csv_file, column_means)
