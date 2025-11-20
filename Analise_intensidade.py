"""
Analise_intensidade.py

Aplica K-Means para classificar intensidade dos regimes macroeconômicos.
Identifica se um regime é Fraco, Moderado ou Forte baseado em análise histórica.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# Ignorar avisos
warnings.simplefilter(action='ignore', category=FutureWarning)


class AnalisadorIntensidade:
    """
    Classe para análise de intensidade de regimes via K-Means.
    
    Attributes:
        n_clusters (int): Número de clusters (padrão: 3 = Fraco/Moderado/Forte)
        kmeans: Modelo K-Means treinado
        scaler: StandardScaler para normalização
        mapeamento_intensidades: Dict mapeando clusters para intensidades
    """
    
    def __init__(self, n_clusters=3):
        """
        Inicializa o analisador.
        
        Args:
            n_clusters (int): Número de níveis de intensidade (padrão: 3)
        """
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.mapeamento_intensidades = None
        self.features_names = None
    
    def preparar_features(self, df):
        """
        Cria features para clusterização a partir do histórico.
        
        Features criadas:
        - magnitude: distância euclidiana da origem (força total do sinal)
        - inflacao_abs: valor absoluto do score de inflação
        - atividade_abs: valor absoluto do score de atividade
        - inflacao_vol: volatilidade do score de inflação (janela 20 dias)
        - atividade_vol: volatilidade do score de atividade (janela 20 dias)
        - consistencia: % de dias no mesmo quadrante (janela 20 dias)
        
        Args:
            df (DataFrame): Histórico com colunas [inflacao_score, atividade_score, quadrante]
        
        Returns:
            DataFrame com 6 features normalizadas
        """
        features = pd.DataFrame()
        
        # Feature 1: Magnitude (distância da origem)
        # Quanto mais longe de (0,0), mais forte o sinal
        features['magnitude'] = np.sqrt(
            df['inflacao_score']**2 + 
            df['atividade_score']**2
        )
        
        # Features 2 e 3: Valores absolutos
        # Intensidade independente da direção
        features['inflacao_abs'] = df['inflacao_score'].abs()
        features['atividade_abs'] = df['atividade_score'].abs()
        
        # Features 4 e 5: Volatilidade (janela de 20 dias)
        # Sinal volátil = menos confiável
        features['inflacao_vol'] = (
            df['inflacao_score']
            .rolling(window=20, min_periods=5)
            .std()
            .fillna(0)
        )
        features['atividade_vol'] = (
            df['atividade_score']
            .rolling(window=20, min_periods=5)
            .std()
            .fillna(0)
        )
        
        # Feature 6: Consistência do quadrante
        # Se mudou de quadrante recentemente = sinal fraco
        def calcular_consistencia(serie):
            """Calcula % de dias no mesmo quadrante (últimos 20)"""
            if len(serie) < 5:
                return 0.5  # Valor neutro
            return (serie == serie.iloc[-1]).sum() / len(serie)
        
        features['consistencia'] = (
            df['quadrante']
            .rolling(window=20, min_periods=5)
            .apply(calcular_consistencia)
            .fillna(0.5)
        )
        
        self.features_names = features.columns.tolist()
        
        return features
    
    def encontrar_k_otimo(self, features_scaled, max_k=6, verbose=True):
        """
        Usa Elbow Method e Silhouette Score para encontrar K ótimo.
        
        Args:
            features_scaled: Features normalizadas
            max_k (int): Máximo de clusters para testar
            verbose (bool): Se True, imprime resultados
        
        Returns:
            DataFrame com métricas para cada K testado
        """
        resultados = {'K': [], 'Inertia': [], 'Silhouette': []}
        
        for k in range(2, max_k + 1):
            if verbose:
                print(f"Testando K={k}...", end=' ')
            
            kmeans = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            labels = kmeans.fit_predict(features_scaled)
            
            inertia = kmeans.inertia_
            silhouette = silhouette_score(features_scaled, labels)
            
            resultados['K'].append(k)
            resultados['Inertia'].append(inertia)
            resultados['Silhouette'].append(silhouette)
            
            if verbose:
                print(f"Inertia={inertia:.2f}, Silhouette={silhouette:.3f}")
        
        df_resultados = pd.DataFrame(resultados)
        
        if verbose:
            # Plotar gráficos
            self._plotar_k_otimo(df_resultados)
        
        return df_resultados
    
    def _plotar_k_otimo(self, df_resultados):
        """
        Cria gráficos de Elbow Method e Silhouette Score.
        
        Args:
            df_resultados: DataFrame com métricas por K
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow Method
        ax1.plot(df_resultados['K'], df_resultados['Inertia'], 
                'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Número de Clusters (K)', fontsize=12)
        ax1.set_ylabel('Inércia (WCSS)', fontsize=12)
        ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette Score
        ax2.plot(df_resultados['K'], df_resultados['Silhouette'], 
                'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Número de Clusters (K)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Marcar melhor K
        melhor_k = df_resultados.loc[df_resultados['Silhouette'].idxmax(), 'K']
        melhor_silhouette = df_resultados['Silhouette'].max()
        ax2.scatter([melhor_k], [melhor_silhouette], 
                   color='green', s=200, zorder=5, marker='*',
                   label=f'Melhor K={melhor_k}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('k_otimo_analise.png', dpi=300)
        print("\n✓ Gráficos salvos em 'k_otimo_analise.png'")
        plt.show()
    
    def treinar(self, df_historico, verbose=True):
        """
        Pipeline completo de treinamento do K-Means.
        
        Passos:
        1. Preparar features
        2. Normalizar
        3. Treinar K-Means
        4. Mapear clusters para intensidades
        
        Args:
            df_historico: DataFrame com histórico de quadrantes
            verbose (bool): Se True, imprime progresso
        
        Returns:
            tuple: (labels, features)
        """
        if verbose:
            print("\n🔬 Iniciando treinamento K-Means...\n")
        
        # Passo 1: Preparar features
        if verbose:
            print("1️⃣ Preparando features...")
        features = self.preparar_features(df_historico)
        if verbose:
            print(f"   ✓ {len(features.columns)} features criadas: {list(features.columns)}")
        
        # Passo 2: Normalizar
        if verbose:
            print("\n2️⃣ Normalizando features...")
        features_scaled = self.scaler.fit_transform(features)
        if verbose:
            print(f"   ✓ Features padronizadas (média≈0, std≈1)")
        
        # Passo 3: Treinar K-Means
        if verbose:
            print(f"\n3️⃣ Treinando K-Means (K={self.n_clusters})...")
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        labels = self.kmeans.fit_predict(features_scaled)
        
        # Calcular métricas
        silhouette = silhouette_score(features_scaled, labels)
        
        if verbose:
            print(f"   ✓ Modelo treinado")
            print(f"   ✓ Silhouette Score: {silhouette:.3f}")
            print(f"   ✓ Centróides salvos: {self.kmeans.cluster_centers_.shape}")
            
            # Distribuição dos clusters
            print(f"\n📊 Distribuição dos clusters:")
            unique, counts = np.unique(labels, return_counts=True)
            for cluster, count in zip(unique, counts):
                pct = (count / len(labels)) * 100
                print(f"   Cluster {cluster}: {count:3} observações ({pct:5.1f}%)")
        
        # Passo 4: Mapear intensidades
        if verbose:
            print("\n4️⃣ Mapeando clusters para intensidades...")
        self.mapeamento_intensidades = self._mapear_intensidades(features, labels, verbose)
        
        if verbose:
            print("\n✅ Treinamento completo!\n")
        
        return labels, features
    
    def _mapear_intensidades(self, features, labels, verbose=True):
        """
        Mapeia clusters numéricos para rótulos semânticos (Fraco/Moderado/Forte).
        
        Lógica: Cluster com maior magnitude média = Forte
        
        Args:
            features: DataFrame de features
            labels: Array de rótulos de cluster
            verbose (bool): Se True, imprime mapeamento
        
        Returns:
            dict: Mapeamento {cluster: intensidade}
        """
        df_temp = features.copy()
        df_temp['cluster'] = labels
        
        # Calcular magnitude média por cluster
        magnitude_media = df_temp.groupby('cluster')['magnitude'].mean()
        
        if verbose:
            print(f"\n   Magnitude média por cluster:")
        
        # Ordenar clusters por magnitude (menor → maior)
        clusters_ordenados = magnitude_media.sort_values().index.tolist()
        
        # Criar mapeamento
        intensidades = ['Fraco', 'Moderado', 'Forte'] if self.n_clusters == 3 else [f'Nivel_{i}' for i in range(self.n_clusters)]
        mapeamento = {}
        
        for i, cluster in enumerate(clusters_ordenados):
            mapeamento[cluster] = intensidades[i]
            mag = magnitude_media[cluster]
            if verbose:
                print(f"   Cluster {cluster} (mag={mag:.3f}) → {intensidades[i]}")
        
        return mapeamento
    
    def classificar(self, inflacao_score, atividade_score, features_contexto=None):
        """
        Classifica intensidade de uma nova observação.
        
        Args:
            inflacao_score (float): Score de inflação atual
            atividade_score (float): Score de atividade atual
            features_contexto (dict, optional): Dict com volatilidades e consistência
                Formato: {'inflacao_vol': float, 'atividade_vol': float, 'consistencia': float}
        
        Returns:
            dict: Classificação completa com intensidade, cluster e magnitude
        """
        if self.kmeans is None:
            raise ValueError("Modelo não treinado! Execute treinar() primeiro.")
        
        # Preparar features do ponto atual
        features_atual = {
            'magnitude': np.sqrt(inflacao_score**2 + atividade_score**2),
            'inflacao_abs': abs(inflacao_score),
            'atividade_abs': abs(atividade_score),
        }
        
        # Adicionar contexto se disponível, senão usar valores default
        if features_contexto:
            features_atual['inflacao_vol'] = features_contexto.get('inflacao_vol', 0.05)
            features_atual['atividade_vol'] = features_contexto.get('atividade_vol', 0.05)
            features_atual['consistencia'] = features_contexto.get('consistencia', 0.7)
        else:
            # Valores default (médios)
            features_atual['inflacao_vol'] = 0.05
            features_atual['atividade_vol'] = 0.05
            features_atual['consistencia'] = 0.7
        
        # Converter para array (mesma ordem do treinamento!)
        X = np.array([[
            features_atual['magnitude'],
            features_atual['inflacao_abs'],
            features_atual['atividade_abs'],
            features_atual['inflacao_vol'],
            features_atual['atividade_vol'],
            features_atual['consistencia']
        ]])
        
        # Normalizar usando o scaler treinado
        X_scaled = self.scaler.transform(X)
        
        # Prever cluster
        cluster = self.kmeans.predict(X_scaled)[0]
        
        # Mapear para intensidade
        intensidade = self.mapeamento_intensidades[cluster]
        
        # Calcular distância ao centróide (medida de confiança)
        distancia = np.linalg.norm(X_scaled - self.kmeans.cluster_centers_[cluster])
        
        return {
            'intensidade': intensidade,
            'cluster': int(cluster),
            'magnitude': features_atual['magnitude'],
            'distancia_centroide': float(distancia),
            'features': features_atual
        }
    
    def adicionar_intensidades_ao_historico(self, df_historico, labels):
        """
        Adiciona colunas de cluster e intensidade ao histórico.
        
        Args:
            df_historico: DataFrame original
            labels: Array de rótulos de cluster
        
        Returns:
            DataFrame com colunas adicionais [cluster, intensidade]
        """
        df_resultado = df_historico.copy()
        df_resultado['cluster'] = labels
        df_resultado['intensidade'] = [self.mapeamento_intensidades[c] for c in labels]
        
        return df_resultado
    
    def visualizar_clusters(self, df_resultado, salvar=True):
        """
        Cria visualização 2D dos clusters no espaço Inflação × Atividade.
        
        Args:
            df_resultado: DataFrame com colunas [inflacao_score, atividade_score, intensidade]
            salvar (bool): Se True, salva gráfico em arquivo
        """
        plt.figure(figsize=(12, 8))
        
        # Cores por intensidade
        cores = {'Fraco': 'lightblue', 'Moderado': 'orange', 'Forte': 'red'}
        
        # Scatter plot por intensidade
        intensidades = df_resultado['intensidade'].unique()
        for intensidade in intensidades:
            mask = df_resultado['intensidade'] == intensidade
            plt.scatter(
                df_resultado.loc[mask, 'inflacao_score'],
                df_resultado.loc[mask, 'atividade_score'],
                c=cores.get(intensidade, 'gray'),
                label=intensidade,
                alpha=0.6,
                s=100
            )
        
        # Linhas de separação dos quadrantes
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
        
        # Labels dos quadrantes
        max_x = df_resultado['inflacao_score'].abs().max() * 0.7
        max_y = df_resultado['atividade_score'].abs().max() * 0.7
        plt.text(max_x, max_y, 'Q2:\nREFLAÇÃO', fontsize=9, alpha=0.3, ha='center', va='center')
        plt.text(-max_x, max_y, 'Q1:\nGOLDILOCKS', fontsize=9, alpha=0.3, ha='center', va='center')
        plt.text(-max_x, -max_y, 'Q4:\nDEFLAÇÃO', fontsize=9, alpha=0.3, ha='center', va='center')
        plt.text(max_x, -max_y, 'Q3:\nESTAGFLAÇÃO', fontsize=9, alpha=0.3, ha='center', va='center')
        
        plt.xlabel('Score de Inflação', fontsize=12)
        plt.ylabel('Score de Atividade Econômica', fontsize=12)
        plt.title('Classificação de Intensidade dos Regimes (K-Means)', fontsize=14, fontweight='bold')
        plt.legend(title='Intensidade', fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig('clusters_intensidade.png', dpi=300)
            print("📊 Gráfico salvo em 'clusters_intensidade.png'")
        
        plt.show()


def main():
    """
    Função principal para executar análise de intensidade.
    """
    print("\n" + "="*70)
    print(" "*15 + "ANÁLISE DE INTENSIDADE VIA K-MEANS")
    print("="*70)
    
    # 1. Carregar histórico
    print("\n1️⃣ Carregando histórico de quadrantes...")
    try:
        df_historico = pd.read_csv('historico_quadrantes.csv', parse_dates=['data'])
        print(f"   ✓ {len(df_historico)} observações carregadas")
        print(f"   ✓ Período: {df_historico['data'].min().date()} a {df_historico['data'].max().date()}")
    except FileNotFoundError:
        print("\n   ❌ Arquivo 'historico_quadrantes.csv' não encontrado!")
        print("   💡 Execute primeiro 'visualizacao_analise_historica/analise_historica.py'")
        return
    
    # 2. Criar analisador
    print("\n2️⃣ Criando analisador de intensidade...")
    analisador = AnalisadorIntensidade(n_clusters=3)
    print("   ✓ Analisador criado (K=3: Fraco/Moderado/Forte)")
    
    # 3. Treinar modelo
    labels, features = analisador.treinar(df_historico, verbose=True)
    
    # 4. Adicionar intensidades ao histórico
    print("\n5️⃣ Adicionando intensidades ao histórico...")
    df_final = analisador.adicionar_intensidades_ao_historico(df_historico, labels)
    
    # 5. Estatísticas
    print("\n📊 DISTRIBUIÇÃO DE INTENSIDADES:")
    print("-" * 70)
    dist = df_final['intensidade'].value_counts()
    for intensidade, count in dist.items():
        pct = (count / len(df_final)) * 100
        print(f"   {intensidade:10} {count:3} períodos ({pct:5.1f}%)")
    
    # 6. Estatísticas por quadrante
    print("\n📈 INTENSIDADE POR QUADRANTE:")
    print("-" * 70)
    crosstab = pd.crosstab(df_final['quadrante'], df_final['intensidade'], normalize='index') * 100
    print(crosstab.round(1))
    
    # 7. Salvar resultados
    print("\n6️⃣ Salvando resultados...")
    df_final.to_csv('historico_com_intensidade.csv', index=False)
    print("   ✓ Resultados salvos em 'historico_com_intensidade.csv'")
    
    # 8. Visualizar
    print("\n7️⃣ Gerando visualização...")
    analisador.visualizar_clusters(df_final, salvar=True)
    
    # 9. Classificar observação atual
    print("\n8️⃣ Classificando regime atual...")
    ultima = df_final.iloc[-1]
    
    # Calcular features de contexto (volatilidade e consistência recentes)
    ultimas_20 = df_final.tail(20)
    features_contexto = {
        'inflacao_vol': ultimas_20['inflacao_score'].std(),
        'atividade_vol': ultimas_20['atividade_score'].std(),
        'consistencia': (ultimas_20['quadrante'] == ultima['quadrante']).sum() / 20
    }
    
    resultado = analisador.classificar(
        ultima['inflacao_score'],
        ultima['atividade_score'],
        features_contexto
    )
    
    # 10. Exibir resultado final
    print("\n" + "="*70)
    print(" "*20 + "🎯 REGIME ATUAL")
    print("="*70)
    print(f"\n📅 Data: {ultima['data'].date()}")
    print(f"📊 Quadrante: {ultima['quadrante']}")
    print(f"💪 Intensidade: {resultado['intensidade']}")
    print(f"📈 Magnitude do Sinal: {resultado['magnitude']:.3f}")
    print(f"🎯 Cluster: {resultado['cluster']}")
    print(f"📏 Distância ao Centróide: {resultado['distancia_centroide']:.3f}")
    print(f"🔍 Consistência (20 dias): {features_contexto['consistencia']:.1%}")
    print("\n" + "="*70)
    print("\n✅ Análise completa!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
