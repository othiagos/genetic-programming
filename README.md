
# Algoritmo de Programação Genética

---

## Requisitos

Antes de executar o programa, certifique-se de ter o seguinte instalado:

- Python 3.12.6 (Testado nessa versão)
- Bibliotecas Python necessárias (instale usando `pip`):
  ```bash
  pip install -r requirements.txt
  ```

---

## Uso

Execute o programa com os seguintes argumentos de linha de comando:

### Argumentos Obrigatórios:
- `--dataset`: Especifique o conjunto de dados a ser usado. Opções: `cancer` ou `wine`.

### Argumentos Opcionais:
- `--population_size`: Especifique o tamanho da população (padrão: 30).
- `--generations`: Especifique o número de gerações (padrão: 30).
- `--mutation_prob`: Defina a probabilidade de mutação como um valor decimal (padrão: 0.05).
- `--depth`: Defina a profundidade máxima das árvores dos indivíduos.
- `--seed`: Defina uma semente para o gerador de números aleatórios para garantir reprodutibilidade.
- `--multithreading`: Habilite o multithreading para avaliação de fitness (`0` para não, qualquer outro valor para sim; padrão: 1).
- `--tournament`: Especifique o número de indivíduos para seleção no torneio (padrão: 3).
- `--expr_file`: Salve os resultados gerais em um arquivo CSV (opcional).
- `--gen_file`: Salve os resultados de cada geração em um arquivo CSV (opcional).

### Comando de Exemplo
```bash
python genetic_program.py --dataset cancer --population_size 50 --generations 100 --mutation_prob 0.1 --depth 5 --seed 42 --multithreading 1 --tournament 5 --expr_file --gen_file
```

---

## Observações

### Como interpretar o nome do experimento

Os nomes dos experimentos seguem um formato padronizado que indica as configurações utilizadas no algoritmo. Por exemplo: `EXPR_CANCER_P25_G30_M05_T2_D7`.

- **PXX**: Tamanho da população utilizado. Neste exemplo, `P25` indica uma população de 25 indivíduos.
- **GXX**: Número de gerações utilizadas. Aqui, `G30` indica 30 gerações.
- **MXX**: Taxa de mutação utilizada. Em `M05`, a taxa de mutação é de 0,05 (5%).
- **TXX**: Tamanho do torneio utilizado. No exemplo, `T2` indica que dois indivíduos foram selecionados para o torneio.
- **DXX**: Profundidade máxima das árvores utilizadas. Em `D7`, a profundidade máxima é 7.
