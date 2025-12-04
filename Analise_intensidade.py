"""
Analise_intensidade_simplificado.py

K-Means na forma mais simples possível:
- Usa apenas duas features: inflacao_score e atividade_score
- Normaliza com StandardScaler
- Aplica KMeans com K=3
- Mapeia clusters para Fraco/Moderado/Forte via magnitude
- Gera visualização 2D e imprime métricas básicas

Objetivo: ter uma versão enxuta, fácil de entender e rodar.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def carregar_historico(path_csv='historico_quadrantes.csv'):
    try:
        df = pd.read_csv(path_csv, parse_dates=['data'])
        print(f"✓ {len(df)} observações carregadas de '{path_csv}'")
        return df
    except FileNotFoundError:
        print("❌ Arquivo 'historico_quadrantes.csv' não encontrado.")
        print("💡 Rode antes 'visualizacao_analise_historica/analise_historica.py'.")
        raise


def preparar_features_simples(df):
    """Cria apenas duas features: inflacao_score e atividade_score."""
    features = df[['inflacao_score', 'atividade_score']].copy()
    return features


def treinar_kmeans_simples(features, k=3):
    """Normaliza e treina KMeans simples."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X)

    sil = silhouette_score(X, labels)
    print(f"\n✅ Treinado KMeans simples (K={k})")
    print(f"   • Silhouette Score: {sil:.3f}")

    return labels, scaler, kmeans, X

def determinar_quadrante(row):
    x = row['inflacao_score']
    y = row['atividade_score']
    if y > 0 and x < 0:
        return 'Q1'
    elif y > 0 and x > 0:
        return 'Q2'
    elif y < 0 and x > 0:
        return 'Q3'
    else:
        return 'Q4'

def treinar_kmeans_por_quadrante(df):
    """
    Para cada quadrante (Q1..Q4), aplica KMeans com K=3 e mapeia intensidade
    por magnitude. Retorna labels globais 0..11, e um array com intensidades.
    """
    df_local = df.copy()
    df_local['quadrante_simpl'] = df_local.apply(determinar_quadrante, axis=1)

    global_labels = np.full(len(df_local), -1, dtype=int)
    intensidades = np.array([''] * len(df_local), dtype=object)
    cluster_global_id = 0

    quadrantes = ['Q1', 'Q2', 'Q3', 'Q4']
    nome_int = ['Fraco', 'Moderado', 'Forte']

    for q in quadrantes:
        idx = df_local['quadrante_simpl'] == q
        subset = df_local.loc[idx, ['inflacao_score', 'atividade_score']]
        if subset.empty:
            continue
        scaler = StandardScaler()
        X = scaler.fit_transform(subset)
        km = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
        lbl = km.fit_predict(X)

        tmp = subset.copy()
        tmp['cluster'] = lbl
        tmp['magnitude'] = np.sqrt(tmp['inflacao_score']**2 + tmp['atividade_score']**2)
        ordem = tmp.groupby('cluster')['magnitude'].mean().sort_values().index.tolist()
        mapeamento_int = {cl: nome_int[i] for i, cl in enumerate(ordem)}

        local_to_global = {cl: cluster_global_id + i for i, cl in enumerate(ordem)}
        mapped_global = [local_to_global[c] for c in lbl]

        global_labels[np.where(idx)[0]] = mapped_global
        intensidades[np.where(idx)[0]] = [mapeamento_int[c] for c in lbl]

        cluster_global_id += 3

    scaler_all = StandardScaler()
    X_all = scaler_all.fit_transform(df_local[['inflacao_score','atividade_score']])
    if (global_labels >= 0).all():
        sil = silhouette_score(X_all, global_labels)
    else:
        sil = np.nan
    print(f"\n✅ KMeans por quadrante (12 clusters no total)")
    if not np.isnan(sil):
        print(f"   • Silhouette global: {sil:.3f}")
    else:
        print("   • Silhouette global: N/A (faltam rótulos)")

    return global_labels, intensidades


def mapear_intensidade_por_magnitude(df, labels):
    """Mapeia clusters para Fraco/Moderado/Forte pela magnitude (distância da origem)."""
    temp = df[['inflacao_score', 'atividade_score']].copy()
    temp['cluster'] = labels
    temp['magnitude'] = np.sqrt(temp['inflacao_score']**2 + temp['atividade_score']**2)

    ordem = temp.groupby('cluster')['magnitude'].mean().sort_values().index.tolist()
    nomes = ['Fraco', 'Moderado', 'Forte'] if len(ordem) == 3 else [f'Nivel_{i}' for i in range(len(ordem))]

    mapping = {cl: nomes[i] for i, cl in enumerate(ordem)}
    return mapping


def visualizar_clusters_simples(df, labels, mapping, salvar=True):
    """Scatter 2D simples por intensidade."""
    df_plot = df[['inflacao_score', 'atividade_score']].copy()
    df_plot['cluster'] = labels
    df_plot['intensidade'] = df_plot['cluster'].map(mapping)

    cores = {'Fraco': 'lightblue', 'Moderado': 'orange', 'Forte': 'red'}

    plt.figure(figsize=(12, 8))
    for intensidade, grupo in df_plot.groupby('intensidade'):
        plt.scatter(grupo['inflacao_score'], grupo['atividade_score'],
                    c=cores.get(intensidade, 'gray'), label=intensidade,
                    alpha=0.6, s=100)

    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.xlabel('Score de Inflação')
    plt.ylabel('Score de Atividade Econômica')
    plt.title('Intensidade dos Regimes (K-Means Simples)')
    # Rótulos de quadrantes (posições relativas ao eixo, sempre dentro do gráfico)
    ax = plt.gca()
    plt.text(0.15, 0.85, 'Q1: GOLDILOCKS', fontsize=11, alpha=0.35,
             ha='center', va='center', fontweight='bold', transform=ax.transAxes)
    plt.text(0.85, 0.85, 'Q2: REFLAÇÃO', fontsize=11, alpha=0.35,
             ha='center', va='center', fontweight='bold', transform=ax.transAxes)
    plt.text(0.85, 0.15, 'Q3: ESTAGFLAÇÃO', fontsize=11, alpha=0.35,
             ha='center', va='center', fontweight='bold', transform=ax.transAxes)
    plt.text(0.15, 0.15, 'Q4: DEFLAÇÃO/RECESSÃO', fontsize=11, alpha=0.35,
             ha='center', va='center', fontweight='bold', transform=ax.transAxes)
    plt.legend(title='Intensidade', loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    if salvar:
        plt.savefig('clusters_intensidade.png', dpi=300)
        print("📊 Gráfico salvo em 'clusters_intensidade.png'")

    plt.show()


def main():
    print("\n" + "="*70)
    print(" "*10 + "K-MEANS SIMPLES: INFLAÇÃO × ATIVIDADE")
    print("="*70)

    df = carregar_historico()
    # Versão com 3 clusters por quadrante (12 no total)
    labels12, intensidades12 = treinar_kmeans_por_quadrante(df)

    dist = pd.Series(labels12).value_counts().sort_index()
    print("\n📊 Distribuição de clusters (0..11):")
    for cl, ct in dist.items():
        pct = ct / len(labels12) * 100
        print(f"   Cluster {cl:2}: {ct:3} ({pct:5.1f}%)")

    # Visualização por intensidade (cores)
    print("\n🖼️ Gerando visualização simples por intensidade...")
    df_vis = df[['inflacao_score','atividade_score']].copy()
    df_vis['intensidade'] = intensidades12
    cores = {'Fraco': 'lightblue', 'Moderado': 'orange', 'Forte': 'red'}
    plt.figure(figsize=(12,8))
    for nome, grupo in df_vis.groupby('intensidade'):
        plt.scatter(grupo['inflacao_score'], grupo['atividade_score'],
                    c=cores.get(nome, 'gray'), label=nome,
                    alpha=0.6, s=100)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.xlabel('Score de Inflação')
    plt.ylabel('Score de Atividade Econômica')
    plt.title('K-Means Simples: 3 clusters por quadrante (12 no total)')
    # Rótulos de quadrantes (posições relativas ao eixo, sempre dentro do gráfico)
    ax2 = plt.gca()
    plt.text(0.15, 0.85, 'Q1: GOLDILOCKS', fontsize=11, alpha=0.35,
             ha='center', va='center', fontweight='bold', transform=ax2.transAxes)
    plt.text(0.85, 0.85, 'Q2: REFLAÇÃO', fontsize=11, alpha=0.35,
             ha='center', va='center', fontweight='bold', transform=ax2.transAxes)
    plt.text(0.85, 0.15, 'Q3: ESTAGFLAÇÃO', fontsize=11, alpha=0.35,
             ha='center', va='center', fontweight='bold', transform=ax2.transAxes)
    plt.text(0.15, 0.15, 'Q4: DEFLAÇÃO/RECESSÃO', fontsize=11, alpha=0.35,
             ha='center', va='center', fontweight='bold', transform=ax2.transAxes)
    plt.legend(title='Intensidade', loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('clusters_intensidade.png', dpi=300)
    print("📊 Gráfico salvo em 'clusters_intensidade.png'")
    plt.show()

    out = df.copy()
    out['cluster_12'] = labels12
    out['intensidade_12'] = intensidades12
    out.to_csv('historico_intensidade_12_simples.csv', index=False)
    print("\n💾 Resultados salvos em 'historico_intensidade_12_simples.csv'")

    print("\n✅ Concluído.")


if __name__ == '__main__':
    main()
