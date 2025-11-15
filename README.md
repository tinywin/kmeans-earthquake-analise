# K-Means – Análise de Clusters em Dados de Terremotos

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-KMeans-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-data%20analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-graphs-informational)](https://matplotlib.org/)
[![Plotly](https://img.shields.io/badge/plotly-interactive%20maps-3F4F75?logo=plotly)](https://plotly.com/)
[![Kaggle Dataset](https://img.shields.io/badge/dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/warcoder/earthquake-dataset)

---

Este projeto aplica o algoritmo de **K-Means** para identificar padrões em dados de terremotos históricos (1995-2023).

O modelo usa técnicas de machine learning não supervisionado para agrupar eventos sísmicos com base em características como **magnitude**, **profundidade**, **localização geográfica** e outras variáveis.
Além de clusterizar os dados, o sistema calcula o **coeficiente de silhueta** para determinar automaticamente o número ideal de clusters e gera **visualizações interativas** em mapas geográficos.

---

## O que foi feito (explicação simples)

1. Foram coletados dados reais de **terremotos** entre 1995 e 2023
   (fonte: [Kaggle Earthquake Dataset](https://www.kaggle.com/datasets/warcoder/earthquake-dataset)).
2. As variáveis numéricas foram **padronizadas** (StandardScaler).
3. Foi aplicado o algoritmo **K-Means** para diferentes valores de k (número de clusters).
4. Foi calculado o **coeficiente de silhueta** para identificar o melhor k automaticamente.
5. Foram gerados **gráficos de clusters**, **mapas interativos** e **relatórios completos**.
6. Comparação com **Gaussian Mixture Model (GMM)** para validação dos resultados.

---

## O que é K-Means

O **K-Means** é um algoritmo de aprendizado não supervisionado que agrupa dados em k clusters.
Neste projeto, ele analisa variáveis como **magnitude**, **profundidade**, **latitude**, **longitude** e outras features para identificar padrões geográficos e sísmicos naturais.

O **coeficiente de silhueta** mede a qualidade dos clusters (valores próximos de 1 indicam boa separação).

---

## Métricas e avaliação

| Métrica                       | Explicação                                                     |
| :---------------------------- | :------------------------------------------------------------- |
| **Coeficiente de Silhueta**   | Mede a coesão interna e separação entre clusters (0 a 1).      |
| **Inércia**                   | Soma das distâncias quadradas ao centroide mais próximo.       |
| **Comparação K-Means vs GMM** | Contraste entre clustering rígido (K-Means) e probabilístico.  |
| **PCA (2 componentes)**       | Redução dimensional para visualização bidimensional.           |

---

## Como usar

```powershell
pip install -r requirements.txt
python kmeans_clustering.py --csv earthquake_data.csv --k-min 2 --k-max 10 --salvar-imagens
```

O script lê o CSV, testa diferentes valores de k, calcula a silhueta, identifica o melhor k e gera relatórios em `outputs/`.

---

## Estrutura do projeto

```
earthquake_1995-2023.csv
earthquake_data.csv
kmeans_clustering.py
README.md
requirements.txt
outputs/
 ├── resultados_clusters.csv
 ├── matriz_correlacao.png
 ├── pairplot.png
 ├── silhueta_vs_k.png
 ├── clusters_pca.png
 ├── silhueta_detalhada.png
 ├── clusters_mapa_lat_lon.png
 └── clusters_mapa_plotly.html
```

---

## Parâmetros da linha de comando

| Parâmetro          | Descrição                                                      | Padrão                  |
| :----------------- | :------------------------------------------------------------- | :---------------------- |
| `--csv`            | Caminho do arquivo CSV de entrada                              | `earthquake_data.csv`   |
| `--columns`        | Lista de colunas numéricas a utilizar (opcional)               | Todas as colunas        |
| `--k-min`          | Menor valor de k a testar                                      | `2`                     |
| `--k-max`          | Maior valor de k a testar                                      | `10`                    |
| `--no-plot`        | Não exibir gráficos (apenas salvar)                            | `False`                 |
| `--salvar-imagens` | Salvar todas as figuras na pasta `outputs/`                    | `False`                 |
| `--pairplot`       | Gerar pairplot (pode ser lento em datasets grandes)            | `False`                 |

---

## Exemplos de uso

```powershell
# Testar k de 2 a 6, sem exibir gráficos
python kmeans_clustering.py --csv earthquake_data.csv --k-min 2 --k-max 6 --no-plot

# Usar apenas colunas específicas
python kmeans_clustering.py --csv earthquake_data.csv --columns magnitude depth latitude longitude --k-min 2 --k-max 8

# Gerar e salvar todas as visualizações
python kmeans_clustering.py --csv earthquake_data.csv --columns magnitude depth latitude longitude --k-min 2 --k-max 8 --salvar-imagens

# Incluir pairplot na análise
python kmeans_clustering.py --csv earthquake_data.csv --columns magnitude depth latitude longitude --k-min 2 --k-max 8 --salvar-imagens --pairplot
```

---

## Pipeline de execução

### 1. Carregamento dos dados

* Lê o arquivo CSV especificado
* Seleciona colunas numéricas (todas por padrão ou as especificadas)
* Remove linhas com valores ausentes

### 2. Análise exploratória

* Matriz de correlação entre variáveis
* Pairplot (opcional) com até 5 colunas e 1000 amostras máximo

### 3. Pré-processamento

* Padronização com StandardScaler (média 0, desvio padrão 1)
* Normalização essencial para o desempenho do K-Means

### 4. Busca do melhor k

* Testa valores de k no intervalo definido (`k-min` a `k-max`)
* Calcula o coeficiente de silhueta para cada k
* Identifica o k que maximiza a silhueta média
* Gera gráfico comparativo `silhueta_vs_k.png`

### 5. Treinamento final

* Aplica K-Means com o melhor k encontrado
* Atribui rótulo de cluster para cada amostra
* Calcula centróides finais

### 6. Avaliação e comparação

* Gera resumo estatístico por cluster
* Compara K-Means com Gaussian Mixture Model (GMM)
* Exporta resultados para `resultados_clusters.csv`

### 7. Visualizações

* **PCA 2D:** Projeção bidimensional dos clusters
* **Silhueta detalhada:** Qualidade de cada amostra por cluster
* **Mapa estático:** Latitude × Longitude colorido por cluster
* **Mapa interativo:** Visualização geográfica com Plotly (HTML)

---

## Gráficos gerados

| Arquivo                         | Descrição                                    | Interpretação                          |
| :------------------------------ | :------------------------------------------- | :------------------------------------- |
| `matriz_correlacao.png`         | Heatmap de correlação entre variáveis        | Identifica relações lineares           |
| `pairplot.png`                  | Distribuições e relações bivariadas          | Análise exploratória visual            |
| `silhueta_vs_k.png`             | Coeficiente de silhueta para cada k          | Melhor k = maior silhueta média        |
| `clusters_pca.png`              | Clusters projetados em 2D via PCA            | Separação visual dos grupos            |
| `silhueta_detalhada.png`        | Silhueta individual por amostra              | Qualidade interna dos clusters         |
| `clusters_mapa_lat_lon.png`     | Mapa estático com clusters geográficos       | Distribuição espacial dos clusters     |
| `clusters_mapa_plotly.html`     | Mapa interativo mundial                      | Exploração interativa dos padrões      |

---

## Observações técnicas

* **StandardScaler:** z-score normalização para todas as features
* **Coeficiente de Silhueta:** métrica principal para seleção de k
* **PCA:** redução dimensional para visualização (não afeta o clustering)
* **K-Means:** algoritmo de Lloyd com `random_state=42` para reprodutibilidade
* **GMM:** modelo probabilístico de comparação com k componentes
* **Plotly:** mapas interativos com `scatter_geo` para análise geográfica

---

## Autoria e créditos

* **Autora:** Laura Barbosa Henrique (`@tinywin`)
* **Instituição:** Universidade Federal do Tocantins (UFT)
* **Disciplina:** Inteligência Artificial — 2025/02
* **Docente:** Prof. Dr. Alexandre Rossini
* **Contato:** `laura.henrique@mail.uft.edu.br`

**Dataset:**
["Earthquake Dataset"](https://www.kaggle.com/datasets/warcoder/earthquake-dataset)
Autor: **warcoder** — Kaggle

---

## Configuração do ambiente

### Windows / PowerShell

1. Criar e ativar ambiente virtual (opcional, mas recomendado):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependências:

```powershell
pip install -r requirements.txt
```

---

## Licença e uso

Projeto **educacional**, sem fins comerciais.
Código e experimentos liberados para **aprendizado e pesquisa**, respeitando os termos do Kaggle.

---

## Resumo simples

> "Implementei o algoritmo K-Means para identificar padrões em dados de terremotos históricos.
> O sistema automaticamente encontra o número ideal de clusters usando o coeficiente de silhueta e gera visualizações interativas em mapas geográficos.
> Os resultados mostram agrupamentos naturais de eventos sísmicos por características geográficas e físicas."

---

## Conclusão

O modelo K-Means identificou **padrões geográficos e sísmicos coerentes** nos dados de terremotos.
A análise revelou agrupamentos naturais baseados em:

* Distribuição geográfica (latitude/longitude)
* Características físicas (magnitude/profundidade)
* Padrões temporais e espaciais

O coeficiente de silhueta permitiu selecionar automaticamente o número ideal de clusters, garantindo boa separação e coesão interna dos grupos formados.