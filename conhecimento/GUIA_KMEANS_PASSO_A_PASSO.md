# 🎓 Guia Completo: K-Means para Análise de Intensidade de Regimes

## 📚 Índice
1. [Visão Macro do Processo](#visão-macro)
2. [Teoria do K-Means](#teoria)
3. [Implementação Passo a Passo](#implementação)
4. [Aplicação ao Projeto](#aplicação)

---

## 🗺️ Visão Macro do Processo {#visão-macro}

### **Fluxo Completo da Fase 4:**

```
┌─────────────────────────────────────────────────────────────┐
│                    FASE 4: K-MEANS                          │
└─────────────────────────────────────────────────────────────┘

ENTRADA                    PROCESSAMENTO                 SAÍDA
───────                    ─────────────                 ─────

📊 Histórico de     →  1️⃣ Preparar Features      →  📈 Features
   Quadrantes              (magnitude, vol, etc)        Normalizadas
   (CSV)                                                
                                                         
                       2️⃣ Encontrar K Ótimo      →  📊 Gráficos
                          (Elbow + Silhouette)          de Validação
                                                         
                       3️⃣ Treinar K-Means        →  🤖 Modelo
                          (clustering)                   Treinado
                                                         
                       4️⃣ Mapear Intensidades    →  🏷️ Labels
                          (Fraco/Moderado/Forte)         Semânticos
                                                         
                       5️⃣ Classificar Atual      →  ✅ Regime +
                          (predict)                      Intensidade
```

### **Analogia Didática:**

Imagine que você tem **centenas de fotos de tempestades** e quer classificá-las automaticamente em:
- 🟢 **Chuva Leve** (fraca)
- 🟡 **Temporal** (moderado)  
- 🔴 **Furacão** (forte)

**Problema:** Você não sabe qual foto é qual categoria.

**Solução K-Means:** 
1. O algoritmo analisa **características** de cada foto (velocidade do vento, volume de chuva, nuvens)
2. Agrupa fotos **similares** automaticamente
3. Você rotula cada grupo depois (ex: "Grupo 1 = Chuva Leve")

**No nosso caso:**
- Fotos = Dias de análise histórica
- Características = Inflação, Atividade, Volatilidade
- Grupos = Fraco, Moderado, Forte

---

## 🧠 Teoria do K-Means {#teoria}

### **O que é Clustering?**

**Clustering** é agrupar dados **similares** sem ter rótulos prévios (aprendizado não-supervisionado).

**Exemplo Visual:**

```
Antes do K-Means (só pontos):        Depois do K-Means (3 clusters):

    •     •  •                            🔴     🔴  🔴
  •   •      •                          🔴   🔴      🔴
       •  •                                  🔴  🔴
                                      
    •    •                                🟡    🟡
  •  •     •                            🟡  🟡     🟡
                                      
      •  •   •                              🔵  🔵   🔵
    •      •                              🔵      🔵
```

---

### **Como Funciona o K-Means?**

#### **Passo 1: Escolher K (número de clusters)**
```
K = 3  →  Queremos 3 grupos (Fraco, Moderado, Forte)
```

#### **Passo 2: Inicializar Centróides Aleatórios**
```
Centróide = ponto central de um cluster

Exemplo em 2D (Inflação × Atividade):

  Atividade ↑
      |
    3 |     C1 ●
      |              
    2 |          C2 ●
      |
    1 |  C3 ●
      |
    0 +──────────────→ Inflação
        0   1   2   3

C1, C2, C3 = centróides iniciais (aleatórios)
```

#### **Passo 3: Atribuir Pontos ao Centróide Mais Próximo**
```
Para cada ponto, calcular distância euclidiana:

d = √[(x₁ - x₂)² + (y₁ - y₂)²]

Exemplo:
Ponto A = (1.5, 2.5)
C1 = (2, 3)   →  d = √[(1.5-2)² + (2.5-3)²] = 0.71
C2 = (3, 2)   →  d = √[(1.5-3)² + (2.5-2)²] = 1.58
C3 = (1, 1)   →  d = √[(1.5-1)² + (2.5-1)²] = 1.58

Resultado: Ponto A pertence a C1 (menor distância)
```

#### **Passo 4: Recalcular Centróides**
```
Novo centróide = média de todos os pontos do cluster

Cluster C1: pontos (1, 2), (2, 3), (1.5, 2.5)
Novo C1 = ( (1+2+1.5)/3 , (2+3+2.5)/3 ) = (1.5, 2.5)
```

#### **Passo 5: Repetir até Convergência**
```
Critério de parada:
- Centróides não mudam mais, OU
- Máximo de iterações atingido (ex: 300)
```

---

### **Métricas de Qualidade do Clustering**

#### **1. Inércia (Within-Cluster Sum of Squares)**
```
Inércia = Soma das distâncias² de cada ponto ao seu centróide

Quanto MENOR, melhor (pontos mais próximos dos centros)

Fórmula:
Σ (distância do ponto ao centróide)²
```

#### **2. Silhouette Score (Coesão vs. Separação)**
```
Range: -1 a +1

+1 = Clusters muito bem separados
 0 = Clusters sobrepostos
-1 = Pontos no cluster errado

Fórmula para cada ponto i:
s(i) = (b - a) / max(a, b)

Onde:
a = distância média aos pontos do MESMO cluster
b = distância média aos pontos do cluster MAIS PRÓXIMO
```

**Exemplo Visual:**
```
Silhouette = 0.8 (BOM)          Silhouette = 0.2 (RUIM)

  🔴🔴🔴                          🔴🔵🔴
  🔴🔴🔴                          🔵🔴🔵
              🔵🔵🔵               🔴🔵🔵
              🔵🔵🔵               🔵🔴🔴

Bem separados                   Misturados
```

---

### **Como Escolher K (Número de Clusters)?**

#### **Método 1: Elbow Method (Cotovelo)**
```
Plotar Inércia vs. K

Inércia
   |
   |╲
   | ╲
   |  ╲_____ ← "Cotovelo" (K ótimo)
   |      ───────
   +──────────────→ K
      2  3  4  5  6

Escolher o ponto onde a curva "dobra"
```

#### **Método 2: Silhouette Score**
```
Plotar Silhouette vs. K

Score
   |      ●
   |    ●   ●
   |  ●       ●
   |●           ●
   +──────────────→ K
      2  3  4  5  6

Escolher o K com MAIOR score
```

---

## 💻 Implementação Passo a Passo {#implementação}

### **PASSO 1: Carregar Dados Históricos**

#### **Teoria:**
Precisamos de um histórico de observações passadas para o K-Means "aprender" os padrões.

#### **Código:**
```python
import pandas as pd
import numpy as np

# Carregar histórico de quadrantes (gerado pela análise_historica.py)
df_historico = pd.read_csv('historico_quadrantes.csv', parse_dates=['data'])

print(f"✓ Carregado: {len(df_historico)} observações históricas")
print(f"✓ Período: {df_historico['data'].min()} a {df_historico['data'].max()}")
print("\nColunas disponíveis:")
print(df_historico.columns.tolist())
```

**Exemplo de Output:**
```
✓ Carregado: 245 observações históricas
✓ Período: 2020-03-15 a 2025-11-20
Colunas disponíveis:
['data', 'quadrante', 'inflacao_score', 'atividade_score']
```

#### **O que esperar:**
DataFrame com estrutura:
```
     data        quadrante            inflacao_score  atividade_score
0    2020-03-15  Q3: ESTAGFLAÇÃO      0.45           -0.32
1    2020-03-20  Q3: ESTAGFLAÇÃO      0.52           -0.28
...
```

---

### **PASSO 2: Criar Features para Clustering**

#### **Teoria:**
Features = características numéricas que o K-Means usará para agrupar.

**Por que não usar apenas os scores?**
- Precisamos capturar **intensidade** e **estabilidade** do regime
- Scores sozinhos não dizem se o sinal é volátil ou consistente

**Features que vamos criar:**

| Feature | O que Mede | Fórmula |
|---------|-----------|---------|
| **magnitude** | Força total do sinal | √(inflação² + atividade²) |
| **inflacao_abs** | Direção de inflação | \|inflação_score\| |
| **atividade_abs** | Direção de atividade | \|atividade_score\| |
| **inflacao_vol** | Instabilidade inflação | std(últimos 20 dias) |
| **atividade_vol** | Instabilidade atividade | std(últimos 20 dias) |
| **consistencia** | Estabilidade quadrante | % dias no mesmo quadrante |

#### **Código:**
```python
def preparar_features(df):
    """
    Cria features para K-Means.
    
    Args:
        df: DataFrame com colunas [inflacao_score, atividade_score, quadrante]
    
    Returns:
        DataFrame com 6 features normalizadas
    """
    features = pd.DataFrame()
    
    # Feature 1: Magnitude (distância da origem)
    # Teoria: Quanto mais longe de (0,0), mais forte o sinal
    features['magnitude'] = np.sqrt(
        df['inflacao_score']**2 + 
        df['atividade_score']**2
    )
    
    # Feature 2 e 3: Valores absolutos
    # Teoria: Queremos intensidade independente da direção
    features['inflacao_abs'] = df['inflacao_score'].abs()
    features['atividade_abs'] = df['atividade_score'].abs()
    
    # Feature 4 e 5: Volatilidade (janela de 20 dias)
    # Teoria: Sinal volátil = menos confiável
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
    # Teoria: Se mudou de quadrante recentemente = sinal fraco
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
    
    return features

# Aplicar
features = preparar_features(df_historico)
print("\n✓ Features criadas:")
print(features.describe())
```

**Exemplo de Output:**
```
✓ Features criadas:
         magnitude  inflacao_abs  atividade_abs  inflacao_vol  atividade_vol  consistencia
count    245.00      245.00        245.00         245.00        245.00         245.00
mean     0.52        0.31          0.35           0.08          0.06           0.72
std      0.23        0.18          0.19           0.04          0.03           0.21
min      0.05        0.01          0.02           0.00          0.00           0.20
max      1.15        0.78          0.82           0.25          0.18           1.00
```

**Interpretação:**
- Magnitude média = 0.52 → Sinal moderado
- Consistência média = 72% → Regime costuma ser estável

---

### **PASSO 3: Normalizar Features (Padronização)**

#### **Teoria:**
K-Means usa **distância euclidiana**, que é sensível à escala.

**Problema sem normalização:**
```
Feature A: magnitude = 0.8    (range 0-1)
Feature B: volatilidade = 150  (range 0-500)

Distância será dominada por Feature B!
```

**Solução: StandardScaler**
```
Fórmula: z = (x - μ) / σ

Onde:
x = valor original
μ = média
σ = desvio padrão

Resultado: todos os valores com média=0 e std=1
```

#### **Código:**
```python
from sklearn.preprocessing import StandardScaler

def normalizar_features(features):
    """
    Normaliza features usando StandardScaler.
    
    Teoria: Transforma cada coluna para média=0 e std=1
    
    Returns:
        features_scaled: array numpy normalizado
        scaler: objeto para normalizar novos dados
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print("\n✓ Features normalizadas:")
    print(f"   Shape: {features_scaled.shape}")
    print(f"   Média: {features_scaled.mean(axis=0).round(3)}")
    print(f"   Std: {features_scaled.std(axis=0).round(3)}")
    
    return features_scaled, scaler

# Aplicar
features_scaled, scaler = normalizar_features(features)
```

**Exemplo de Output:**
```
✓ Features normalizadas:
   Shape: (245, 6)
   Média: [0. 0. 0. 0. 0. 0.]  ← Todas próximas de 0
   Std: [1. 1. 1. 1. 1. 1.]    ← Todas = 1
```

---

### **PASSO 4: Encontrar K Ótimo (Elbow + Silhouette)**

#### **Teoria:**
Testar diferentes valores de K (2 a 8) e escolher o melhor.

**Critérios:**
1. **Elbow**: Onde a inércia para de cair muito
2. **Silhouette**: Onde o score é máximo

#### **Código:**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def encontrar_k_otimo(features_scaled, max_k=8):
    """
    Testa K de 2 até max_k e plota métricas.
    
    Returns:
        DataFrame com resultados
    """
    resultados = {'K': [], 'Inertia': [], 'Silhouette': []}
    
    for k in range(2, max_k + 1):
        print(f"Testando K={k}...", end=' ')
        
        # Treinar K-Means
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,  # Seed para reprodutibilidade
            n_init=10,        # 10 inicializações diferentes
            max_iter=300      # Máximo de iterações
        )
        labels = kmeans.fit_predict(features_scaled)
        
        # Calcular métricas
        inertia = kmeans.inertia_
        silhouette = silhouette_score(features_scaled, labels)
        
        resultados['K'].append(k)
        resultados['Inertia'].append(inertia)
        resultados['Silhouette'].append(silhouette)
        
        print(f"Inertia={inertia:.2f}, Silhouette={silhouette:.3f}")
    
    df_resultados = pd.DataFrame(resultados)
    
    # Plotar gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow Method
    ax1.plot(df_resultados['K'], df_resultados['Inertia'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Número de Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inércia (WCSS)', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette Score
    ax2.plot(df_resultados['K'], df_resultados['Silhouette'], 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Número de Clusters (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('k_otimo_analise.png', dpi=300)
    print("\n✓ Gráficos salvos em 'k_otimo_analise.png'")
    plt.show()
    
    return df_resultados

# Executar
df_k_otimo = encontrar_k_otimo(features_scaled, max_k=6)
print("\n📊 Resultados:")
print(df_k_otimo)
```

**Exemplo de Output:**
```
Testando K=2... Inertia=1245.32, Silhouette=0.421
Testando K=3... Inertia=845.67, Silhouette=0.487
Testando K=4... Inertia=623.45, Silhouette=0.452
Testando K=5... Inertia=512.34, Silhouette=0.398
Testando K=6... Inertia=445.23, Silhouette=0.356

✓ Gráficos salvos em 'k_otimo_analise.png'

📊 Resultados:
   K    Inertia  Silhouette
0  2    1245.32      0.421
1  3     845.67      0.487  ← Melhor Silhouette
2  4     623.45      0.452
3  5     512.34      0.398
4  6     445.23      0.356
```

**Interpretação:**
- **K=3** tem o melhor Silhouette (0.487)
- Cotovelo está entre K=3 e K=4
- ✅ **Escolha: K=3** (Fraco, Moderado, Forte)

---

### **PASSO 5: Treinar K-Means Final**

#### **Código:**
```python
def treinar_kmeans_final(features_scaled, k=3):
    """
    Treina modelo final com K escolhido.
    
    Returns:
        kmeans: modelo treinado
        labels: rótulos de cluster para cada observação
    """
    print(f"\n🤖 Treinando K-Means com K={k}...")
    
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    
    labels = kmeans.fit_predict(features_scaled)
    
    # Métricas finais
    silhouette = silhouette_score(features_scaled, labels)
    
    print(f"✓ Modelo treinado!")
    print(f"✓ Silhouette Score: {silhouette:.3f}")
    print(f"✓ Centróides salvos: {kmeans.cluster_centers_.shape}")
    print(f"\n📊 Distribuição dos clusters:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        pct = (count / len(labels)) * 100
        print(f"   Cluster {cluster}: {count:3} observações ({pct:5.1f}%)")
    
    return kmeans, labels

# Treinar
kmeans_final, labels = treinar_kmeans_final(features_scaled, k=3)
```

**Exemplo de Output:**
```
🤖 Treinando K-Means com K=3...
✓ Modelo treinado!
✓ Silhouette Score: 0.487
✓ Centróides salvos: (3, 6)

📊 Distribuição dos clusters:
   Cluster 0:  89 observações ( 36.3%)
   Cluster 1: 102 observações ( 41.6%)
   Cluster 2:  54 observações ( 22.0%)
```

---

### **PASSO 6: Mapear Clusters para Intensidades**

#### **Teoria:**
Os clusters são apenas números (0, 1, 2). Precisamos interpretá-los como "Fraco/Moderado/Forte".

**Lógica:**
- Cluster com **menor magnitude média** = Fraco
- Cluster com **magnitude intermediária** = Moderado  
- Cluster com **maior magnitude média** = Forte

#### **Código:**
```python
def mapear_intensidades(features, labels):
    """
    Mapeia clusters numéricos para rótulos semânticos.
    
    Lógica: Cluster com maior magnitude = Forte
    
    Returns:
        dict: mapeamento {cluster: intensidade}
    """
    print("\n🏷️ Mapeando clusters para intensidades...")
    
    # Adicionar labels às features
    df_temp = features.copy()
    df_temp['cluster'] = labels
    
    # Calcular magnitude média por cluster
    magnitude_media = df_temp.groupby('cluster')['magnitude'].mean()
    print("\n📊 Magnitude média por cluster:")
    print(magnitude_media.sort_values())
    
    # Ordenar clusters por magnitude (menor → maior)
    clusters_ordenados = magnitude_media.sort_values().index.tolist()
    
    # Criar mapeamento
    intensidades = ['Fraco', 'Moderado', 'Forte']
    mapeamento = {}
    
    for i, cluster in enumerate(clusters_ordenados):
        mapeamento[cluster] = intensidades[i]
        mag = magnitude_media[cluster]
        print(f"   Cluster {cluster} (mag={mag:.3f}) → {intensidades[i]}")
    
    return mapeamento

# Aplicar
mapeamento = mapear_intensidades(features, labels)
```

**Exemplo de Output:**
```
🏷️ Mapeando clusters para intensidades...

📊 Magnitude média por cluster:
cluster
0    0.28
1    0.52
2    0.85

   Cluster 0 (mag=0.283) → Fraco
   Cluster 1 (mag=0.524) → Moderado
   Cluster 2 (mag=0.847) → Forte
```

---

### **PASSO 7: Adicionar Intensidades ao Histórico**

#### **Código:**
```python
def adicionar_intensidades(df_historico, labels, mapeamento):
    """
    Adiciona colunas de cluster e intensidade ao histórico.
    """
    df_resultado = df_historico.copy()
    df_resultado['cluster'] = labels
    df_resultado['intensidade'] = [mapeamento[c] for c in labels]
    
    print("\n✅ Intensidades adicionadas ao histórico!")
    print("\n📊 Distribuição Final:")
    dist = df_resultado['intensidade'].value_counts()
    for intensidade, count in dist.items():
        pct = (count / len(df_resultado)) * 100
        print(f"   {intensidade:10} {count:3} períodos ({pct:5.1f}%)")
    
    return df_resultado

# Aplicar
df_final = adicionar_intensidades(df_historico, labels, mapeamento)

# Salvar
df_final.to_csv('historico_com_intensidade.csv', index=False)
print("\n✓ Resultados salvos em 'historico_com_intensidade.csv'")
```

**Exemplo de Output:**
```
✅ Intensidades adicionadas ao histórico!

📊 Distribuição Final:
   Moderado    102 períodos ( 41.6%)
   Fraco        89 períodos ( 36.3%)
   Forte        54 períodos ( 22.0%)

✓ Resultados salvos em 'historico_com_intensidade.csv'
```

---

### **PASSO 8: Classificar Observação Atual**

#### **Teoria:**
Agora que temos o modelo treinado, podemos classificar **novos** dados.

#### **Código:**
```python
def classificar_atual(inflacao_score, atividade_score, features_contexto, 
                      scaler, kmeans, mapeamento):
    """
    Classifica intensidade de uma nova observação.
    
    Args:
        inflacao_score: score atual de inflação
        atividade_score: score atual de atividade
        features_contexto: dict com volatilidades e consistência
        scaler: StandardScaler treinado
        kmeans: modelo K-Means treinado
        mapeamento: dict cluster → intensidade
    
    Returns:
        dict com classificação completa
    """
    # 1. Criar features
    features_atual = {
        'magnitude': np.sqrt(inflacao_score**2 + atividade_score**2),
        'inflacao_abs': abs(inflacao_score),
        'atividade_abs': abs(atividade_score),
        'inflacao_vol': features_contexto.get('inflacao_vol', 0.05),
        'atividade_vol': features_contexto.get('atividade_vol', 0.05),
        'consistencia': features_contexto.get('consistencia', 0.7)
    }
    
    # 2. Converter para array (mesma ordem do treinamento!)
    X = np.array([list(features_atual.values())])
    
    # 3. Normalizar (usar mesmo scaler do treinamento)
    X_scaled = scaler.transform(X)
    
    # 4. Prever cluster
    cluster = kmeans.predict(X_scaled)[0]
    
    # 5. Mapear para intensidade
    intensidade = mapeamento[cluster]
    
    # 6. Calcular distância ao centróide (confiança)
    distancia = np.linalg.norm(X_scaled - kmeans.cluster_centers_[cluster])
    
    resultado = {
        'intensidade': intensidade,
        'cluster': int(cluster),
        'magnitude': features_atual['magnitude'],
        'distancia_centroide': distancia,
        'features': features_atual
    }
    
    return resultado

# Exemplo de uso
# (Pegar última observação do histórico)
ultima = df_historico.iloc[-1]

resultado_atual = classificar_atual(
    inflacao_score=ultima['inflacao_score'],
    atividade_score=ultima['atividade_score'],
    features_contexto={
        'inflacao_vol': 0.06,
        'atividade_vol': 0.04,
        'consistencia': 0.8
    },
    scaler=scaler,
    kmeans=kmeans_final,
    mapeamento=mapeamento
)

print("\n" + "="*60)
print(" "*15 + "🎯 CLASSIFICAÇÃO ATUAL")
print("="*60)
print(f"\n📊 Regime: {ultima['quadrante']}")
print(f"💪 Intensidade: {resultado_atual['intensidade']}")
print(f"📈 Magnitude do Sinal: {resultado_atual['magnitude']:.3f}")
print(f"🎯 Cluster: {resultado_atual['cluster']}")
print(f"📏 Distância ao Centróide: {resultado_atual['distancia_centroide']:.3f}")
print("\n✅ Classificação completa!")
print("="*60)
```

**Exemplo de Output:**
```
============================================================
               🎯 CLASSIFICAÇÃO ATUAL
============================================================

📊 Regime: Q1: GOLDILOCKS
💪 Intensidade: Forte
📈 Magnitude do Sinal: 0.823
🎯 Cluster: 2
📏 Distância ao Centróide: 0.245

✅ Classificação completa!
============================================================
```

---

## 🎯 Aplicação ao Projeto {#aplicação}

### **Arquivo Final: `Analise_intensidade.py`**

Agora vamos **integrar todos os passos** em uma classe reutilizável:

```python
"""
Analise_intensidade.py

Aplica K-Means para classificar intensidade dos regimes macroeconômicos.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path


class AnalisadorIntensidade:
    """
    Classe completa para análise de intensidade via K-Means.
    """
    
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.mapeamento_intensidades = None
    
    def preparar_features(self, df):
        """PASSO 2: Criar features para clustering"""
        features = pd.DataFrame()
        
        features['magnitude'] = np.sqrt(
            df['inflacao_score']**2 + df['atividade_score']**2
        )
        features['inflacao_abs'] = df['inflacao_score'].abs()
        features['atividade_abs'] = df['atividade_score'].abs()
        
        features['inflacao_vol'] = (
            df['inflacao_score'].rolling(20, min_periods=5).std().fillna(0)
        )
        features['atividade_vol'] = (
            df['atividade_score'].rolling(20, min_periods=5).std().fillna(0)
        )
        
        def calc_consistencia(serie):
            if len(serie) < 5:
                return 0.5
            return (serie == serie.iloc[-1]).sum() / len(serie)
        
        features['consistencia'] = (
            df['quadrante'].rolling(20, min_periods=5)
            .apply(calc_consistencia).fillna(0.5)
        )
        
        return features
    
    def encontrar_k_otimo(self, features_scaled, max_k=6):
        """PASSO 4: Encontrar K ótimo"""
        resultados = {'K': [], 'Inertia': [], 'Silhouette': []}
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            
            resultados['K'].append(k)
            resultados['Inertia'].append(kmeans.inertia_)
            resultados['Silhouette'].append(silhouette_score(features_scaled, labels))
        
        return pd.DataFrame(resultados)
    
    def treinar(self, df_historico):
        """PASSOS 2, 3, 5, 6: Pipeline completo de treinamento"""
        print("\n🔬 Iniciando treinamento K-Means...\n")
        
        # Passo 2: Features
        print("1️⃣ Preparando features...")
        features = self.preparar_features(df_historico)
        print(f"   ✓ {len(features.columns)} features criadas")
        
        # Passo 3: Normalizar
        print("\n2️⃣ Normalizando features...")
        features_scaled = self.scaler.fit_transform(features)
        print(f"   ✓ Features padronizadas (média≈0, std≈1)")
        
        # Passo 5: Treinar
        print(f"\n3️⃣ Treinando K-Means (K={self.n_clusters})...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(features_scaled)
        
        silhouette = silhouette_score(features_scaled, labels)
        print(f"   ✓ Modelo treinado")
        print(f"   ✓ Silhouette Score: {silhouette:.3f}")
        
        # Passo 6: Mapear intensidades
        print("\n4️⃣ Mapeando intensidades...")
        self.mapeamento_intensidades = self._mapear(features, labels)
        
        print("\n✅ Treinamento completo!\n")
        
        return labels, features
    
    def _mapear(self, features, labels):
        """PASSO 6: Mapear clusters → intensidades"""
        df_temp = features.copy()
        df_temp['cluster'] = labels
        
        magnitude_media = df_temp.groupby('cluster')['magnitude'].mean()
        clusters_ordenados = magnitude_media.sort_values().index.tolist()
        
        intensidades = ['Fraco', 'Moderado', 'Forte']
        mapeamento = {}
        
        for i, cluster in enumerate(clusters_ordenados):
            mapeamento[cluster] = intensidades[i]
            print(f"   Cluster {cluster} → {intensidades[i]}")
        
        return mapeamento
    
    def classificar(self, inflacao_score, atividade_score, features_contexto=None):
        """PASSO 8: Classificar nova observação"""
        if self.kmeans is None:
            raise ValueError("Modelo não treinado! Execute treinar() primeiro.")
        
        features_atual = {
            'magnitude': np.sqrt(inflacao_score**2 + atividade_score**2),
            'inflacao_abs': abs(inflacao_score),
            'atividade_abs': abs(atividade_score),
            'inflacao_vol': features_contexto.get('inflacao_vol', 0.05) if features_contexto else 0.05,
            'atividade_vol': features_contexto.get('atividade_vol', 0.05) if features_contexto else 0.05,
            'consistencia': features_contexto.get('consistencia', 0.7) if features_contexto else 0.7
        }
        
        X = np.array([list(features_atual.values())])
        X_scaled = self.scaler.transform(X)
        
        cluster = self.kmeans.predict(X_scaled)[0]
        intensidade = self.mapeamento_intensidades[cluster]
        
        return {
            'intensidade': intensidade,
            'cluster': int(cluster),
            'magnitude': features_atual['magnitude']
        }


# ============================================================================
# SCRIPT DE EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    # Carregar histórico
    df_historico = pd.read_csv('historico_quadrantes.csv', parse_dates=['data'])
    print(f"✓ Carregado: {len(df_historico)} observações")
    
    # Criar analisador
    analisador = AnalisadorIntensidade(n_clusters=3)
    
    # Treinar
    labels, features = analisador.treinar(df_historico)
    
    # Adicionar ao histórico
    df_final = df_historico.copy()
    df_final['cluster'] = labels
    df_final['intensidade'] = [analisador.mapeamento_intensidades[c] for c in labels]
    
    # Estatísticas
    print("\n📊 Distribuição de Intensidades:")
    dist = df_final['intensidade'].value_counts()
    for intensidade, count in dist.items():
        pct = (count / len(df_final)) * 100
        print(f"   {intensidade:10} {count:3} períodos ({pct:5.1f}%)")
    
    # Salvar
    df_final.to_csv('historico_com_intensidade.csv', index=False)
    print("\n✓ Resultados salvos em 'historico_com_intensidade.csv'")
    
    # Classificar observação atual
    ultima = df_final.iloc[-1]
    resultado = analisador.classificar(
        ultima['inflacao_score'],
        ultima['atividade_score']
    )
    
    print("\n" + "="*60)
    print(" "*15 + "🎯 REGIME ATUAL")
    print("="*60)
    print(f"\n📊 Quadrante: {ultima['quadrante']}")
    print(f"💪 Intensidade: {resultado['intensidade']}")
    print(f"📈 Magnitude: {resultado['magnitude']:.3f}")
    print("="*60)
```

---

## 📚 Resumo dos Conceitos

| Conceito | O que é | Por que usar |
|----------|---------|--------------|
| **K-Means** | Algoritmo de clustering | Agrupa dados similares automaticamente |
| **Centróide** | Centro de um cluster | Representa o "típico" daquele grupo |
| **Inércia** | Soma das distâncias² aos centros | Mede compactação dos clusters |
| **Silhouette** | Coesão vs. Separação | Valida qualidade dos clusters |
| **StandardScaler** | Normalização z-score | Iguala escalas das features |
| **Features** | Características numéricas | Dados que o K-Means usa para agrupar |

---

## ✅ Checklist de Implementação

- [ ] **Passo 1**: Carregar histórico de quadrantes
- [ ] **Passo 2**: Criar 6 features (magnitude, abs, vol, consistência)
- [ ] **Passo 3**: Normalizar features com StandardScaler
- [ ] **Passo 4**: Testar K de 2 a 6 (Elbow + Silhouette)
- [ ] **Passo 5**: Treinar K-Means final (K=3)
- [ ] **Passo 6**: Mapear clusters → Fraco/Moderado/Forte
- [ ] **Passo 7**: Adicionar intensidades ao histórico
- [ ] **Passo 8**: Classificar observação atual
- [ ] **Passo 9**: Salvar resultados e visualizar

---

## 🎓 Exercícios Práticos

### **Exercício 1: Entender Features**
Calcule manualmente as features para esta observação:
```
inflacao_score = 0.6
atividade_score = -0.4
```

Resposta:
- magnitude = √(0.6² + 0.4²) = √0.52 = 0.72
- inflacao_abs = 0.6
- atividade_abs = 0.4

### **Exercício 2: Interpretar Silhouette**
Se Silhouette = 0.15, os clusters estão bem separados?

Resposta: Não. 0.15 é baixo (próximo de 0), indicando sobreposição.

### **Exercício 3: Escolher K**
Dados:
```
K=2: Silhouette=0.35, Inertia=1200
K=3: Silhouette=0.52, Inertia=800
K=4: Silhouette=0.48, Inertia=650
```
Qual K escolher?

Resposta: K=3 (melhor Silhouette e cotovelo na inércia)

---

## 🚀 Próximos Passos

Após dominar o K-Means:

1. **Fase 5**: Criar trading rules por quadrante + intensidade
2. **Fase 6**: Backtesting das estratégias
3. **Fase 7**: Dashboard de visualização
4. **Fase 8**: Automação e deploy

---

> **"Machine Learning não é mágica. É matemática bem aplicada."**  
> — LEV Quant Research Lab
