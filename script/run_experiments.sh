#!/usr/bin/bash

# Name of the Python script
PYTHON_SCRIPT="src/main.py"
SCRIPT_MEAN="script/calculate_means.py"

# Checks if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: File '$PYTHON_SCRIPT' not found!"
    exit 1
fi

seed=137

# Arguments received
DS="cancer"
DS_UPCASE=$(echo $DS | tr 'a-z' 'A-Z')
P=$1
G=$2
M=$3
T=$4
D=$5

MM=$(awk "BEGIN {print $M * 100}")
MM=$(printf "%02d" $MM)

CSV="EXPR_${DS_UPCASE}_P${P}_G${G}_M${MM}_T${T}_D${D}.csv"
echo $CSV

# Loop to execute 10 times
for i in {1..10}; do
    echo "Execution $i:"
    # Passing the arguments directly to the Python script
    python3 "$PYTHON_SCRIPT" --dataset "$DS" --population_size "$P" --generations "$G" --mutation_prob "$M" --tournament "$T" --depth "$D" --expr_file --seed $((seed + i))
    # Checks if there was an error in the execution
    if [ $? -ne 0 ]; then
        echo "Error in execution $i"
    fi
    echo "Execution $i completed"
    echo "---------------------------"
done

python3 "$SCRIPT_MEAN" experiment/"$CSV"
