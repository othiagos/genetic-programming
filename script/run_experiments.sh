#!/usr/bin/zsh

# Nome do script Python
PYTHON_SCRIPT="src/main.py"

# Verifica se o script Python existe
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Erro: Arquivo '$PYTHON_SCRIPT' não encontrado!"
    exit 1
fi

# Loop para executar 10 vezes
for i in {1..10}; do
    echo "Execução $i:"
    # Passando os argumentos diretamente ao script Python
    python3 "$PYTHON_SCRIPT" "$@"
    # Verifica se houve erro na execução
    if [ $? -ne 0 ]; then
        echo "Erro na execução $i"
    fi
    echo "Execução $i concluída"
    echo "--------------------------"
done
