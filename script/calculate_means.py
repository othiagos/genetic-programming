import argparse
import csv
import os
from collections import OrderedDict

csv_mean_file = "experiment/DATE_EXPR_MEAN.csv"


def calculate_column_means(csv_file):
    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        columns = {key: [] for key in reader.fieldnames}

        for row in reader:
            for key, value in row.items():
                if key != "seed":  # Ignora a coluna 'seed'
                    columns[key].append(float(value))

        column_means = {key: sum(values) / len(values) for key, values in columns.items() if key != "seed"}
        return column_means


def save_means_to_csv(input_file, column_means):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    column_means["file"] = base_name

    # Reorganiza para que "file" seja a primeira coluna
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
