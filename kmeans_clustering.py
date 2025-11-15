"""
Trabalho: Agrupamento (Clustering) com K-Means

Coeficiente de silhueta:
- Mede quão bem cada ponto está situado dentro do seu cluster em comparação
    com os outros clusters.
- Varia de -1 a 1:
    * próximo de 1: ponto bem agrupado
    * próximo de 0: ponto na fronteira entre clusters
    * valor negativo: ponto possivelmente no cluster errado

Neste código, escolhemos o valor de k que maximiza a silhueta média.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


# ==================================================
# Carregar dados
# ==================================================
def carregar_dados(caminho_csv: str, colunas: List[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv)

    if colunas is None:
        df = df.select_dtypes(include=["number"]).dropna()
        if df.empty:
            raise ValueError("O CSV não possui colunas numéricas utilizáveis.")
        return df

    missing = [c for c in colunas if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas não encontradas no CSV: {missing}")

    return df[colunas].dropna()


# ==================================================
# K-Means com silhueta
# ==================================================
def executar_kmeans_silhueta(
    dados: pd.DataFrame,
    k_range: List[int],
    random_state: int = 42,
) -> Tuple[int, List[int], List[float], StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(dados)
    ks_validos, silhuetas = [], []

    for k in k_range:
        if k <= 1 or k >= len(dados):
            continue

        modelo = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = modelo.fit_predict(X)
        score = silhouette_score(X, labels)

        ks_validos.append(k)
        silhuetas.append(score)

    if not ks_validos:
        raise ValueError("Nenhum valor válido de k para calcular silhueta.")

    melhor_k = ks_validos[int(np.argmax(silhuetas))]
    return melhor_k, ks_validos, silhuetas, scaler


# ==================================================
# Visualizações
# ==================================================
def plotar_silhueta(ks, silhuetas, salvar: bool = False, caminho: str | None = None):
    plt.figure(figsize=(8, 4))
    plt.plot(ks, silhuetas, marker="o")
    plt.title("Silhueta Média para Diferentes Valores de k")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Silhueta média")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if salvar and caminho is not None:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(caminho, dpi=300)
    plt.show()


def plotar_silhueta_detalhada(X, labels, k, salvar: bool = False, caminho: str | None = None):
    """Gráfico de silhueta por ponto, clássico em trabalhos de clustering."""
    sil_vals = silhouette_samples(X, labels)
    sil_media = silhouette_score(X, labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_lower = 10
    for c in np.unique(labels):
        vals_c = sil_vals[labels == c]
        vals_c.sort()
        size_c = vals_c.shape[0]
        y_upper = y_lower + size_c

        cor = plt.cm.tab10(float(c) / max(labels))
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            vals_c,
            facecolor=cor,
            edgecolor=cor,
            alpha=0.7,
        )

        ax.text(-0.05, y_lower + 0.5 * size_c, str(c))
        y_lower = y_upper + 10

    ax.axvline(x=sil_media, color="red", linestyle="--", label=f"média = {sil_media:.3f}")
    ax.set_title(f"Gráfico de Silhueta por Cluster (k={k})")
    ax.set_xlabel("Valor da silhueta")
    ax.set_ylabel("Índice da amostra (agrupado por cluster)")
    ax.legend()
    plt.tight_layout()
    if salvar and caminho is not None:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(caminho, dpi=300)
    plt.show()


def plotar_correlacao(df, salvar: bool = False, caminho: str | None = None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="viridis", fmt=".2f")
    plt.title("Matriz de Correlação")
    plt.tight_layout()
    if salvar and caminho is not None:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(caminho, dpi=300)
    plt.show()


def plotar_clusters_pca(X, labels, melhor_k, salvar: bool = False, caminho: str | None = None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=40, alpha=0.8
    )

    # Centroides no espaço PCA
    centroids = []
    for c in np.unique(labels):
        centroids.append(X_pca[labels == c].mean(axis=0))
    centroids = np.vstack(centroids)
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        s=120,
        marker="X",
        edgecolor="white",
        linewidth=1.5,
        label="Centroides",
    )

    plt.title(f"Clusters (PCA 2D) com centroides - k={melhor_k}")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    plt.tight_layout()
    if salvar and caminho is not None:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(caminho, dpi=300)
    plt.show()


def plotar_mapa_clusters(dados, labels, salvar: bool = False, caminho: str | None = None):
    """Mapa 2D latitude x longitude colorido pelos clusters."""
    if "latitude" not in dados.columns or "longitude" not in dados.columns:
        print("> Latitude/longitude não disponíveis; pulando mapa de clusters.")
        return

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        dados["longitude"],
        dados["latitude"],
        c=labels,
        cmap="tab10",
        s=30,
        alpha=0.8,
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Clusters no espaço geográfico (lat x lon)")
    plt.grid(alpha=0.3)
    plt.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    plt.tight_layout()
    if salvar and caminho is not None:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(caminho, dpi=300)
    plt.show()


def plotar_mapa_plotly(dados, labels, salvar_html: bool = False, caminho: str | None = None):
    """Mapa geográfico interativo usando Plotly scatter_geo."""
    if "latitude" not in dados.columns or "longitude" not in dados.columns:
        print("> Latitude/longitude não disponíveis; pulando mapa Plotly.")
        return

    df_plot = dados.copy()
    df_plot["cluster"] = labels.astype(str)

    fig = px.scatter_geo(
        df_plot,
        lat="latitude",
        lon="longitude",
        color="cluster",
        title="Clusters em mapa geográfico (Plotly)",
        projection="natural earth",
    )

    if salvar_html and caminho is not None:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(caminho)

    fig.show()


# ==================================================
# Main
# ==================================================
def main():
    parser = argparse.ArgumentParser(
        description="Clustering com K-Means e visualizações avançadas."
    )
    parser.add_argument("--csv", type=str, required=True, help="Arquivo CSV.")
    parser.add_argument(
        "--colunas",
        "--columns",
        dest="colunas",
        type=str,
        nargs="*",
        default=None,
        help="Nomes das colunas numéricas a utilizar (se vazio, usa todas).",
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Não mostrar gráficos (apenas resultados numéricos).",
    )
    parser.add_argument(
        "--salvar-imagens",
        action="store_true",
        help="Salvar as figuras geradas em arquivos PNG.",
    )
    parser.add_argument("--pairplot", action="store_true",
                        help="Gera também pairplot (pode ser lento).")
    args = parser.parse_args()

    print("=== AGRUPAMENTO K-MEANS COM VISUALIZAÇÕES ===")
    print(f"> Lendo arquivo: {args.csv}")
    if args.colunas:
        print(f"> Usando colunas: {args.colunas}")
    else:
        print("> Usando todas as colunas numéricas disponíveis")

    # Diretório de saída para arquivos gerados
    output_dir = Path("outputs")

    dados = carregar_dados(args.csv, args.colunas)

    print(f"Dados carregados: {dados.shape[0]} linhas, {dados.shape[1]} colunas")

    # Correlação
    if not args.no_plot:
        print("\n> Gerando matriz de correlação...")
        plotar_correlacao(
            dados,
            salvar=args.salvar_imagens,
            caminho=str(output_dir / "matriz_correlacao.png"),
        )

    # Pairplot
    if args.pairplot and not args.no_plot:
        max_cols = 5
        cols_to_plot = dados.columns[:max_cols]
        print(f"\n> Gerando pairplot para colunas: {list(cols_to_plot)}")

        if len(dados) > 1000:
            dados_plot = dados.sample(1000, random_state=42)
            print(
                f"> Dataset possui {len(dados)} linhas, amostrando 1000 para o pairplot...")
        else:
            dados_plot = dados

        g = sns.pairplot(dados_plot[cols_to_plot])
        if args.salvar_imagens:
            output_dir.mkdir(parents=True, exist_ok=True)
            g.savefig(output_dir / "pairplot.png", dpi=300)
        plt.show()

    # Seleção de k
    print("\n> Normalizando dados e avaliando diferentes valores de k...")
    if args.k_min < 2:
        raise ValueError("k-min deve ser pelo menos 2.")
    if args.k_max < args.k_min:
        raise ValueError("k-max deve ser maior ou igual a k-min.")

    intervalo_k = list(range(args.k_min, args.k_max + 1))
    melhor_k, ks, silhuetas, scaler = executar_kmeans_silhueta(dados, intervalo_k)

    print("\nResultados da silhueta:")
    for k, s in zip(ks, silhuetas):
        print(f"k={k:2d} → silhueta média = {s:.4f}")

    print(f"\nMelhor k: {melhor_k}")

    if not args.no_plot:
        plotar_silhueta(
            ks,
            silhuetas,
            salvar=args.salvar_imagens,
            caminho=str(output_dir / "silhueta_vs_k.png"),
        )

    # Treinar modelo final (K-Means)
    X = scaler.transform(dados)
    modelo = KMeans(n_clusters=melhor_k, random_state=42, n_init="auto")
    labels = modelo.fit_predict(X)

    # Salvar resultados em CSV
    resultado = dados.copy()
    resultado["cluster"] = labels
    output_dir.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(output_dir / "resultados_clusters.csv", index=False)
    print("\n> Resultados com rótulo de cluster salvos em 'outputs/resultados_clusters.csv'")

    # Resumo por cluster
    print("\nResumo por cluster (médias das variáveis):")
    print(resultado.groupby("cluster").mean(numeric_only=True))

    # Comparação com GMM usando o mesmo número de componentes
    print("\n> Comparando com Gaussian Mixture Model (GMM)...")
    gmm = GaussianMixture(n_components=melhor_k, random_state=42)
    labels_gmm = gmm.fit_predict(X)
    sil_gmm = silhouette_score(X, labels_gmm)
    print(f"Silhueta com K-Means (k={melhor_k}): {max(silhuetas):.4f}")
    print(f"Silhueta com GMM     (k={melhor_k}): {sil_gmm:.4f}")

    # Imagens adicionais
    if not args.no_plot:
        print("\n> Plotando clusters usando PCA...")
        plotar_clusters_pca(
            X,
            labels,
            melhor_k,
            salvar=args.salvar_imagens,
            caminho=str(output_dir / "clusters_pca.png"),
        )

        print("> Plotando clusters em lat x lon (mapa)...")
        plotar_mapa_clusters(
            resultado,
            labels,
            salvar=args.salvar_imagens,
            caminho=str(output_dir / "clusters_mapa_lat_lon.png"),
        )

        print("> Gerando gráfico de silhueta detalhado para o melhor k...")
        plotar_silhueta_detalhada(
            X,
            labels,
            melhor_k,
            salvar=args.salvar_imagens,
            caminho=str(output_dir / "silhueta_detalhada.png"),
        )

        # Mapa geográfico interativo (Plotly scatter_geo) salvo em HTML, se desejar
        if args.salvar_imagens:
            print("> Gerando mapa geográfico interativo (Plotly scatter_geo)...")
            try:
                plotar_mapa_plotly(
                    resultado,
                    labels,
                    salvar_html=True,
                    caminho=str(output_dir / "clusters_mapa_plotly.html"),
                )
            except Exception as e:
                print("[AVISO] Não foi possível gerar o mapa Plotly:", e)


def _run():
    try:
        main()
    except ValueError as e:
        print("\n[ERRO]", e)
    except Exception as e:
        print("\n[ERRO INESPERADO]")
        print(type(e).__name__, ":", e)


if __name__ == "__main__":
    _run()