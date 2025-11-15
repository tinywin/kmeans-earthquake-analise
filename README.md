## K-Means Clustering com Coeficiente de Silhueta

Este projeto implementa a tarefa de **clustering com K-Means** usando um conjunto de dados em CSV
(`earthquake_data.csv`, baseado em dados de terremotos). O objetivo é:

- aplicar **K-Means** para agrupar os dados;
- testar diferentes valores de **k** (número de clusters);
- usar o **coeficiente de silhueta** para escolher o melhor valor de k;
- apresentar os resultados em um **script Python (.py)**, sem uso de notebook (`.ipynb`).

O código utiliza **Python**, **Scikit-Learn**, **Matplotlib**, **Seaborn** e **Plotly** para
realizar o agrupamento, calcular métricas e gerar visualizações.

Os dados utilizados foram obtidos a partir do seguinte conjunto de dados público no Kaggle:

- **Earthquake Dataset** – disponível em: https://www.kaggle.com/datasets/warcoder/earthquake-dataset
	(autor do dataset: *warcoder*).

---

## Autoria

- **Autora:** Laura Barbosa Henrique (@tinywin)  
- **Instituição:** Universidade Federal do Tocantins (UFT)  
- **Disciplina:** Inteligência Artificial — 2025/02  
- **Docente:** Prof. Dr. Alexandre Rossini  
- **Contato:** laura.henrique@mail.uft.edu.br

---

## Arquivos principais

- `kmeans_clustering.py`: script principal que carrega o CSV, aplica K-Means,
	calcula o coeficiente de silhueta para vários valores de k, treina o modelo final
	e gera visualizações (PCA, mapas, silhueta etc.).
- `earthquake_data.csv`: base de dados em formato CSV usada como exemplo para o clustering.
- `requirements.txt`: lista de dependências Python usadas no projeto.
- Pasta `outputs/`: diretório onde são salvos **resultados** e **imagens** geradas pelo script:
  - `outputs/resultados_clusters.csv`: dados com coluna extra `cluster`.
  - `outputs/matriz_correlacao.png`: heatmap de correlação.
  - `outputs/pairplot.png`: pairplot (se ativado).
  - `outputs/silhueta_vs_k.png`: silhueta média em função de k.
  - `outputs/clusters_pca.png`: clusters projetados em 2D via PCA.
  - `outputs/silhueta_detalhada.png`: gráfico de silhueta por ponto para o melhor k.
	- `outputs/clusters_mapa_lat_lon.png`: mapa simples latitude x longitude colorido pelos clusters.
	- `outputs/clusters_mapa_plotly.html`: mapa geográfico interativo (Plotly scatter_geo).

---

## Como configurar o ambiente (Windows / PowerShell)

1. (Opcional, mas recomendado) Crie e ative um ambiente virtual:

	```powershell
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	```

2. Instale as dependências:

	```powershell
	pip install -r requirements.txt
	```

---

## Como executar o script de K-Means

Dentro da pasta do projeto (`kmeans-2025`), execute no PowerShell:

```powershell
python kmeans_clustering.py --csv earthquake_data.csv --k-min 2 --k-max 10
```

### Parâmetros principais

- `--csv`: caminho para o arquivo CSV (padrão: `earthquake_data.csv`).
- `--colunas` / `--columns`: lista de colunas numéricas a utilizar. Se não informar,
  o script usa todas as colunas numéricas do CSV.
- `--k-min`: menor valor de k a ser testado (padrão: 2). Deve ser ≥ 2.
- `--k-max`: maior valor de k a ser testado (padrão: 10). Deve ser ≥ `k-min`.
- `--no-plot`: se for passado, não mostra gráficos, apenas a saída em texto
  (não afeta o salvamento de CSVs, mas evita abrir janelas de figura).
- `--salvar-imagens`: salva as figuras geradas na pasta `outputs/` (formato PNG).
- `--pairplot`: gera também um pairplot das primeiras colunas (pode ser mais lento). Quando
  ativado junto com `--salvar-imagens`, o pairplot é salvo em `outputs/pairplot.png`.

### Exemplos de uso

```powershell
# Exemplo 1: usar todas as colunas numéricas e testar k de 2 a 6, sem gráficos
python kmeans_clustering.py --csv earthquake_data.csv --k-min 2 --k-max 6 --no-plot

# Exemplo 2: usar apenas algumas colunas específicas
python kmeans_clustering.py --csv earthquake_data.csv --columns magnitude depth latitude longitude --k-min 2 --k-max 8

# Exemplo 3: gerar gráficos e salvar imagens na pasta outputs/ usando o Python da venv
.\.venv\Scripts\python.exe kmeans_clustering.py --csv earthquake_data.csv --columns magnitude depth latitude longitude --k-min 2 --k-max 8 --salvar-imagens

# Exemplo 4: ativar também o pairplot (pode demorar um pouco mais)
python kmeans_clustering.py --csv earthquake_data.csv --columns magnitude depth latitude longitude --k-min 2 --k-max 8 --salvar-imagens --pairplot
```

---

## O que o script faz (pipeline)

1. **Carregamento dos dados**
	- Lê o arquivo CSV informado em `--csv`.
	- Se `--colunas/--columns` não for passado, seleciona apenas colunas numéricas.
	- Remove linhas com valores ausentes nas colunas selecionadas.

2. **Análise exploratória com imagens (opcional)**
	- Gera uma **matriz de correlação** (`matriz_correlacao.png`).
	- Opcionalmente, gera um **pairplot** com até 5 colunas e, no máximo, 1000 amostras
	  (para não ficar pesado).

3. **Pré-processamento**
	- Padroniza as features com `StandardScaler` (média 0, desvio padrão 1),
	  algo importante para o K-Means.

4. **Busca do melhor k via coeficiente de silhueta**
	- Para cada valor de k no intervalo [`k-min`, `k-max`]:
	  - treina um modelo `KMeans`;
	  - calcula o **coeficiente de silhueta médio** para esse k;
	  - guarda os valores de silhueta.
	- Gera o gráfico `silhueta_vs_k.png` se `--salvar-imagens` estiver ativo.
	- Escolhe o **melhor k** como aquele que maximiza a silhueta média.

5. **Treinamento final do modelo**
	- Reutiliza o mesmo `StandardScaler` ajustado na fase de seleção de k.
	- Treina um novo K-Means com `n_clusters = melhor_k`.
	- Atribui um rótulo de cluster para cada amostra.

6. **Saídas numéricas**
	- Imprime no terminal a tabela com `k` e a respectiva silhueta média.
	- Mostra o valor de k considerado ideal.
	- Salva `outputs/resultados_clusters.csv` com todas as colunas usadas e
	  uma coluna extra `cluster`.
	- Imprime um **resumo por cluster** (médias das variáveis em cada grupo).
	- Compara a silhueta do **K-Means** com a silhueta de um modelo
	  **Gaussian Mixture Model (GMM)** com o mesmo número de componentes `k`.

7. **Visualizações finais (opcionais)**
	- **Clusters em 2D com PCA**: gráfico `clusters_pca.png`, projetando os dados em duas
	  componentes principais e colorindo por cluster.
	- **Gráfico de silhueta detalhado**: `silhueta_detalhada.png`, mostrando a silhueta
	  de cada amostra em cada cluster, com a linha vertical indicando a silhueta média.
	- **Mapa latitude x longitude (estático)**: `clusters_mapa_lat_lon.png`, exibindo os
	  terremotos em um plano geográfico simples (lon x lat), coloridos pelos clusters.
	- **Mapa geográfico interativo (Plotly)**: `clusters_mapa_plotly.html`, permitindo
	  explorar os clusters sobre um mapa mundi interativo (zoom, pan etc.).