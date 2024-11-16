#!/usr/bin/zsh

# Nome do script Python
PYTHON_SCRIPT="src/main.py"
SCRIPT_MEAN="script/calculate_means.py"

# Verifica se o script Python existe
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Erro: Arquivo '$PYTHON_SCRIPT' não encontrado!"
    exit 1
fi

seed=137

# Argumentos recebidos
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

# Loop para executar 10 vezes
for i in {1..10}; do
    echo "Execução $i:"
    # Passando os argumentos diretamente ao script Python
    python3 "$PYTHON_SCRIPT" --dataset "$DS" --population_size "$P" --generations "$G" --mutation_prob "$M" --tournament "$T" --depth "$D" --expr_file --seed $((seed + i))
    # Verifica se houve erro na execução
    if [ $? -ne 0 ]; then
        echo "Erro na execução $i"
    fi
    echo "Execução $i concluída"
    echo "---------------------------"
done

python3 "$SCRIPT_MEAN" experiment/"$CSV"
