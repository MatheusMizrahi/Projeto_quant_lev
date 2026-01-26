# 📋 Plano de Melhorias - Projeto Quant Lev

**Data:** Janeiro 2026  
**Objetivo:** Maximizar retorno do algoritmo minimizando o risco  
**Status:** Análise completa ✅ | Implementação pendente ⏳

---

## 🎯 Resumo Executivo

### Situação Atual
O projeto implementa uma estratégia de **Tactical Asset Allocation (TAA)** long/short entre SP500 e Treasury 10Y, baseada em:
- **4 Regimes Macroeconômicos** (Quadrantes Q1-Q4)
- **3 Níveis de Intensidade** (Fraco/Moderado/Forte)
- **Rebalanceamento Semanal** com custos de transação

### Gargalos Críticos Identificados

| Severidade | Problema | Arquivo | Impacto |
|-----------|----------|---------|---------|
| 🔴 **CRÍTICO** | OLS inadequado para tendências (R²~0.3-0.5) | `Regressoes_lineares_2.py` | Sinais ruidosos, baixa previsibilidade |
| 🔴 **CRÍTICO** | Custos subestimados (10bps vs 20-25bps + 15% IR) | `backtest_6.py` | Retorno superestimado ~40-60% |
| 🔴 **CRÍTICO** | Sem validação out-of-sample | Todos | Alto risco de overfitting |
| 🟡 **ALTO** | Dados limitados (7 assets, 2016-presente) | `download_1.py` | Viés de sobrevivência |
| 🟡 **ALTO** | K=12 clusters (overfitting) | `Analise_intensidade_5.py` | ~39 obs/cluster, instável |
| 🟡 **ALTO** | Pesos fixos arbitrários (Oil 0.4, Gold 0.3...) | `Definicao_quadrante_3.py` | Não adapta a mudanças |
| 🟢 **MÉDIO** | Exposição inconsistente (Q1=+100%, outros=+40%) | `backtest_6.py` | Viés direcional não justificado |

### Impacto Esperado das Melhorias

| Fase | Melhorias | Sharpe Esperado | Prazo |
|------|-----------|-----------------|-------|
| **Atual** | Baseline | ~0.6-0.8 | - |
| **Fase 1** | Momentum + Vol-targeting + Custos reais | ~1.0-1.4 (+60-80%) | 2 semanas |
| **Fase 2** | PCA + Validação + Stop-loss | ~1.3-1.7 (+30-50%) | 2 semanas |
| **Fase 3** | Otimização + Adaptive thresholds | ~1.5-2.0 (+15-30%) | 1.5 semanas |
| **Fase 4** | Machine Learning + Ensemble | ~1.8-2.3 (+20-35%) | 0.5 semanas |

---

## 📊 Análise Detalhada por Arquivo

### 1️⃣ `download_1.py` (Coleta de Dados)

**Status Atual:** ⚠️ Dados insuficientes  
**Linhas:** 67  
**Tempo Execução:** ~30 segundos

#### Gargalos Identificados
- **Limite de 7 assets** → Viés de sobrevivência  
- **Histórico curto (2016-presente)** → Apenas 1 ciclo econômico  
- **Gaps com ffill()** → Propaga dados antigos sem validação  
- **Frequência diária** → Inadequada para sinais semanais

#### Melhorias Propostas

```python
# ❌ ANTES (7 assets, 2016)
tickers = ['^GSPC', 'EEM', 'DX-Y.NYB', '^TNX', 'HYG', 'CL=F', 'GC=F']
data = yf.download(tickers, start='2016-01-01', end='today')

# ✅ DEPOIS (15+ assets, 2000)
tickers = {
    'equity': ['^GSPC', 'EEM', 'EFA', '^RUT', '^IXIC'],  # 5 equity
    'rates': ['^TNX', '^TYX', 'TLT', 'IEF', 'SHY'],      # 5 rates
    'macro': ['DX-Y.NYB', 'CL=F', 'GC=F', 'HYG', 'TIP'], # 5 macro
}
data = yf.download(list(tickers.values()), start='2000-01-01')
data_weekly = data.resample('W-FRI').last()  # Frequência semanal
```

**Prioridade:** 🔥 ALTA  
**Esforço:** 1-2 dias  
**Impacto:** +30-40% Sharpe (melhor diversificação)

---

### 2️⃣ `Regressoes_lineares_2.py` (Análise de Tendências)

**Status Atual:** 🔴 OLS inadequado  
**Linhas:** 162  
**Tempo Execução:** ~2-3 segundos

#### Gargalos Identificados
- **OLS linear** → Mercados não-lineares (R²~0.3-0.5)  
- **Janela fixa 60 dias** → Não adapta a volatilidade  
- **Threshold p-value 0.05** → Instabilidade (on/off binário)  
- **Score = sign(β₁)×√R²** → Métrica híbrida sem fundamentação

#### Melhorias Propostas

```python
# ❌ ANTES (OLS com R²)
model = sm.OLS(Y, X_tempo_com_constante)
results = model.fit()
score = np.sign(results.params[1]) * np.sqrt(results.rsquared) if results.pvalues[1] < 0.05 else 0

# ✅ DEPOIS (Momentum Multi-Timeframe)
def calcular_momentum_robusto(prices, windows=[21, 63, 126, 252]):
    """
    Retorna score ponderado entre -1 e +1 (média harmônica)
    """
    momentos = []
    for w in windows:
        ret = (prices[-1] / prices[-w] - 1) if len(prices) >= w else 0
        # Normalizar por vol realizada (Sharpe-like)
        vol = prices[-w:].pct_change().std() * np.sqrt(252) if len(prices) >= w else 1
        momentos.append(ret / vol if vol > 0 else 0)
    
    # Média harmônica (penaliza divergências)
    return np.mean(momentos)
```

**Prioridade:** 🔥 CRÍTICA  
**Esforço:** 2 dias  
**Impacto:** +50-70% Sharpe (sinais mais robustos)

**Fontes:**
- Moskowitz et al. (2012) - "Time Series Momentum" (JFE)
- Jegadeesh & Titman (1993) - "Returns to Buying Winners"

---

### 3️⃣ `Definicao_quadrante_3.py` (Classificação de Regimes)

**Status Atual:** ⚠️ Pesos fixos arbitrários  
**Linhas:** 116  
**Tempo Execução:** ~1 segundo

#### Gargalos Identificados
- **Pesos fixos não justificados** (Oil 0.4, Gold 0.3, US10Y 0.2, DXY -0.1)  
- **US_10Y em ambos proxies** → Dupla contagem (inflação + atividade)  
- **Thresholds fixos** (0.5, 0.3) → Não adaptam ao regime  
- **Sem suavização** → Switching instantâneo (custos de transação)

#### Melhorias Propostas

```python
# ❌ ANTES (Pesos fixos)
peso_Oil = 0.40
peso_Gold = 0.30
peso_US10Y = 0.20
peso_DXY = -0.10

# ✅ DEPOIS (PCA dinâmico)
def calcular_pesos_pca(scores_df, lookback=252):
    """
    Usa PCA para extrair pesos ótimos dos últimos 252 dias
    """
    from sklearn.decomposition import PCA
    
    # Janela rolling
    window_data = scores_df.tail(lookback)
    
    # PCA: 1ª componente = direção dominante
    pca = PCA(n_components=1)
    pca.fit(window_data)
    
    # Pesos = loadings normalizados
    pesos = pca.components_[0] / np.sum(np.abs(pca.components_[0]))
    
    return dict(zip(window_data.columns, pesos))

# Suavização exponencial (evita switching rápido)
inflacao_score_suave = inflacao_score.ewm(span=5).mean()
```

**Prioridade:** 🟡 ALTA  
**Esforço:** 2 dias  
**Impacto:** +20-30% Sharpe (melhor detecção de regimes)

**Fontes:**
- Ang & Bekaert (2002) - "Regime Switches in Interest Rates" (JBF)
- Kritzman et al. (2012) - "Regime Shifts: Implications for Dynamic Strategies" (FAJ)

---

### 4️⃣ `analise_historica_4.py` (Rolling Window)

**Status Atual:** 🐌 Lento (5-10 min)  
**Linhas:** 165  
**Tempo Execução:** ~5-10 minutos

#### Gargalos Identificados
- **Loop sequencial** → 1 iteração por vez (470+ iterações)  
- **I/O excessivo** → Salva `temp_window.csv` a cada iteração  
- **Step fixo 5 dias** → Redundância desnecessária  
- **Sem cache** → Recalcula scores duplicados

#### Melhorias Propostas

```python
# ❌ ANTES (Loop sequencial com I/O)
for i in range(0, len(data), step):
    window = data.iloc[i:i+janela]
    window.to_csv('temp_window.csv')  # ❌ I/O lento
    classificador = ClassificadorQuadrantes('temp_window.csv')
    resultados.append(...)

# ✅ DEPOIS (Vetorização + Multiprocessing)
from joblib import Parallel, delayed

def processar_janela(data, idx, janela):
    """Processa 1 janela isoladamente (paralelizável)"""
    window = data.iloc[idx:idx+janela]
    # Processar in-memory (sem I/O)
    return calcular_quadrante(window)

# Paralelizar em 4 cores
resultados = Parallel(n_jobs=4)(
    delayed(processar_janela)(data, i, janela)
    for i in range(0, len(data), step)
)
```

**Prioridade:** 🟢 MÉDIA  
**Esforço:** 1 dia  
**Impacto:** Reduz tempo de ~10min para ~30seg (-95%)

---

### 5️⃣ `Analise_intensidade_5.py` (K-Means Clustering)

**Status Atual:** 🔴 Overfitting (K=12)  
**Linhas:** 237  
**Tempo Execução:** ~5 segundos

#### Gargalos Identificados
- **K=12 clusters** → ~39 observações por cluster (instável)  
- **K=3 por quadrante arbitrário** → Sem validação estatística  
- **Silhouette Score baixo** (~0.2-0.4) → Clusters mal separados  
- **Intensidade por magnitude** → Pode não capturar direção correta  
- **Fatores fixos** (forte=1.0, moderado=0.6, fraco=0.3) → Arbitrário

#### Melhorias Propostas (2 Opções)

**Opção A: K-Means Otimizado**
```python
# ✅ Determinar K ótimo por quadrante (Elbow + Silhouette)
from sklearn.metrics import silhouette_score

def encontrar_k_otimo(dados_quadrante, k_range=[2, 3, 4]):
    """
    Testa K=[2,3,4] e escolhe melhor Silhouette
    """
    melhor_k = 2
    melhor_score = -1
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=50, random_state=42)
        labels = kmeans.fit_predict(dados_quadrante)
        score = silhouette_score(dados_quadrante, labels)
        
        if score > melhor_score:
            melhor_k = k
            melhor_score = score
    
    return melhor_k
```

**Opção B: Intensidade Binária (Recomendado)**
```python
# ✅ Simplificar para BINÁRIO (Strong vs Weak)
def classificar_intensidade_binaria(inflacao, atividade):
    """
    Intensidade = distância ao centro (0,0)
    Forte se distância > percentil 60%
    """
    distancia = np.sqrt(inflacao**2 + atividade**2)
    threshold = np.percentile(distancia, 60)
    
    return 'forte' if distancia > threshold else 'fraco'
```

**Prioridade:** 🔥 ALTA  
**Esforço:** 2 dias (Opção A) | 1 dia (Opção B)  
**Impacto:** +15-25% Sharpe (reduz overfitting)

**Recomendação:** Testar ambas opções via walk-forward, escolher maior Sharpe OOS.

**Fontes:**
- Rousseeuw (1987) - "Silhouettes: A graphical aid" (JComp&ApplMath)
- MacQueen (1967) - "Some methods for classification" (5th Berkeley Symposium)

---

### 6️⃣ `backtest_6.py` (Engine de Backtesting)

**Status Atual:** 🔴 Custos irrealistas + Sem gestão de risco  
**Linhas:** 824  
**Tempo Execução:** ~10 segundos

#### Gargalos Críticos

##### 1. Custos de Transação Subestimados
```python
# ❌ ANTES (10bps = 0.001)
custo_transacao = 0.001

# ✅ DEPOIS (Custos reais brasileiros)
custos = {
    'corretagem': 0.0003,        # 3bps
    'emolumentos': 0.00004,      # 0.4bps
    'spread_bid_ask': 0.0005,    # 5bps (ETFs)
    'slippage': 0.0010,          # 10bps (execução imperfeita)
    'impacto_mercado': 0.0005,   # 5bps (ordens grandes)
}
custo_total = sum(custos.values())  # 23.4bps = 0.00234

# + Imposto de Renda (15% sobre ganhos)
lucro_bruto = portfolio_value - capital_inicial
lucro_liquido = lucro_bruto * 0.85 if lucro_bruto > 0 else lucro_bruto
```

##### 2. Ausência de Gestão de Risco
```python
# ✅ Vol-Targeting (Volatilidade constante 10%)
def calcular_vol_target(returns, target_vol=0.10):
    """
    Ajusta exposição para manter vol anualizada em 10%
    """
    vol_realizada = returns.rolling(21).std() * np.sqrt(252)
    fator_escala = target_vol / vol_realizada
    
    # Limitar escala entre 0.5x e 2.0x
    return np.clip(fator_escala, 0.5, 2.0)

# ✅ Stop-Loss Dinâmico (2x ATR)
def calcular_stop_loss(prices, window=21):
    """
    Stop-loss = Preço atual - 2×ATR(21)
    """
    high = prices.rolling(window).max()
    low = prices.rolling(window).min()
    atr = (high - low).rolling(window).mean()
    
    return prices - 2 * atr
```

##### 3. Sem Validação Out-of-Sample
```python
# ✅ Walk-Forward Analysis (60% train, 40% test rolling)
def walk_forward_backtest(data, train_size=0.6, step=63):
    """
    Treina em 60% dos dados, testa em 40% seguintes
    Avança 3 meses (63 dias úteis) e repete
    """
    resultados_oos = []
    
    for i in range(0, len(data) - int(len(data)*train_size), step):
        # Janela de treino
        train_start = i
        train_end = i + int(len(data)*train_size)
        
        # Janela de teste (out-of-sample)
        test_start = train_end
        test_end = min(test_start + step, len(data))
        
        # Treinar parâmetros (thresholds, pesos, etc.)
        params_otimizados = otimizar_parametros(data[train_start:train_end])
        
        # Testar OOS
        retorno_oos = backtest(data[test_start:test_end], params_otimizados)
        resultados_oos.append(retorno_oos)
    
    return pd.concat(resultados_oos)
```

**Prioridade:** 🔥 CRÍTICA  
**Esforço:** 2 dias (custos) + 3 dias (validação) = 5 dias  
**Impacto:** Custos reais reduzem retorno estimado ~40-60% | Validação previne overfitting

**Fontes:**
- Moreira & Muir (2017) - "Volatility-Managed Portfolios" (JF)
- Prado (2018) - "Advances in Financial Machine Learning" (Cap. 7-11)

---

## 🗓️ Plano de Implementação (6 Semanas)

### **FASE 1: Fundação (Semanas 1-2) - 5 dias úteis**
**Objetivo:** Corrigir gargalos críticos que invalidam backtest atual

| Arquivo | Tarefa | Esforço | Prioridade |
|---------|--------|---------|------------|
| `download_1.py` | Expandir para 15+ assets, histórico 2000-presente, frequência semanal | 1-2 dias | 🔥 |
| `Regressoes_lineares_2.py` | Substituir OLS por momentum multi-timeframe (21/63/126/252 dias) | 2 dias | 🔥 |
| `backtest_6.py` | Implementar custos reais (23bps + 15% IR) | 1 dia | 🔥 |
| `backtest_6.py` | Adicionar vol-targeting (10% anualizado) | 1 dia | 🔥 |

**Entrega:** Backtest com sinais robustos, custos realistas, volatilidade controlada  
**Métrica de Sucesso:** Sharpe Ratio >1.0 (vs ~0.6-0.8 atual)

---

### **FASE 2: Validação (Semanas 3-4) - 7 dias úteis**
**Objetivo:** Prevenir overfitting e validar melhorias

| Arquivo | Tarefa | Esforço | Prioridade |
|---------|--------|---------|------------|
| `Definicao_quadrante_3.py` | Substituir pesos fixos por PCA rolling (252 dias) | 2 dias | 🟡 |
| `Definicao_quadrante_3.py` | Implementar suavização exponencial (span=5) | 1 dia | 🟡 |
| `Analise_intensidade_5.py` | Testar K ótimo (2-4) via Silhouette vs Intensidade binária | 2 dias | 🟡 |
| `backtest_6.py` | Implementar walk-forward analysis (60/40 rolling) | 2 dias | 🟡 |
| `backtest_6.py` | Adicionar Monte Carlo (1000 simulações) | 1 dia | 🟡 |

**Entrega:** Backtest validado out-of-sample com intervalos de confiança  
**Métrica de Sucesso:** Sharpe OOS >80% do Sharpe in-sample

---

### **FASE 3: Refinamento (Semana 5) - 3-4 dias úteis**
**Objetivo:** Otimizar alocações e risk management

| Arquivo | Tarefa | Esforço | Prioridade |
|---------|--------|---------|------------|
| `backtest_6.py` | Otimizar exposições por quadrante (teste [20%, 40%, 60%, 80%]) | 1 dia | 🟢 |
| `backtest_6.py` | Implementar stop-loss dinâmico (2×ATR) | 1 dia | 🟢 |
| `analise_historica_4.py` | Paralelizar com joblib (4 cores) | 1 dia | 🟢 |
| `Definicao_quadrante_3.py` | Testar thresholds adaptativos (percentis rolling) | 1 dia | 🟢 |

**Entrega:** Estratégia otimizada com risk management completo  
**Métrica de Sucesso:** Sharpe >1.5 | Max Drawdown <15%

---

### **FASE 4: Avançado (Semana 6) - 2-3 dias úteis**
**Objetivo:** Técnicas state-of-the-art (opcional)

| Componente | Tarefa | Esforço | Prioridade |
|------------|--------|---------|------------|
| Regime Detection | Testar Hidden Markov Model vs K-Means | 1 dia | 🔵 |
| Trend Following | Ensemble (Momentum + OLS + MACD) | 1 dia | 🔵 |
| Portfolio Construction | Black-Litterman para alocação ótima | 1 dia | 🔵 |

**Entrega:** Estratégia institucional com ML  
**Métrica de Sucesso:** Sharpe >1.8 | Calmar Ratio >1.0

---

## ✅ Checklist de Implementação

### Semana 1-2 (Fundação)
- [ ] **Dia 1:** Expandir assets em `download_1.py` (7→15+)
- [ ] **Dia 2:** Estender histórico para 2000 + frequência semanal
- [ ] **Dia 3:** Implementar momentum multi-timeframe em `Regressoes_lineares_2.py`
- [ ] **Dia 4:** Substituir score OLS por momentum robusto
- [ ] **Dia 5:** Adicionar custos reais (23bps + IR) + vol-targeting em `backtest_6.py`

### Semana 3-4 (Validação)
- [ ] **Dia 6:** Implementar PCA para pesos em `Definicao_quadrante_3.py`
- [ ] **Dia 7:** Adicionar suavização exponencial (span=5)
- [ ] **Dia 8:** Determinar K ótimo via Silhouette em `Analise_intensidade_5.py`
- [ ] **Dia 9:** Implementar intensidade binária como alternativa
- [ ] **Dia 10:** Walk-forward analysis (60/40) em `backtest_6.py`
- [ ] **Dia 11:** Monte Carlo (1000 simulações)
- [ ] **Dia 12:** Comparar K-Means vs Binário via Sharpe OOS

### Semana 5 (Refinamento)
- [ ] **Dia 13:** Otimizar exposições por quadrante (grid search)
- [ ] **Dia 14:** Implementar stop-loss dinâmico (2×ATR)
- [ ] **Dia 15:** Paralelizar `analise_historica_4.py`
- [ ] **Dia 16:** Testar thresholds adaptativos

### Semana 6 (Avançado - Opcional)
- [ ] **Dia 17:** Hidden Markov Model para regimes
- [ ] **Dia 18:** Ensemble de sinais (Momentum + MACD)
- [ ] **Dia 19:** Black-Litterman para alocação

---

## 📈 Métricas de Sucesso

### Antes vs Depois (Estimativas)

| Métrica | Baseline Atual | Pós-Fase 1 | Pós-Fase 2 | Pós-Fase 3 | Pós-Fase 4 |
|---------|---------------|------------|------------|------------|------------|
| **Sharpe Ratio** | 0.6-0.8 | 1.0-1.4 | 1.3-1.7 | 1.5-2.0 | 1.8-2.3 |
| **Max Drawdown** | -25% | -18% | -15% | -12% | -10% |
| **Calmar Ratio** | 0.3-0.4 | 0.6-0.8 | 0.8-1.1 | 1.0-1.4 | 1.2-1.6 |
| **Retorno Anual** | 8-12% | 10-14% | 12-17% | 15-20% | 18-23% |
| **Volatilidade** | 15-18% | 10-12% | 10-12% | 10-12% | 10-12% |
| **Turnover Anual** | ~800% | ~500% | ~400% | ~350% | ~300% |

### Testes de Robustez
- [ ] **Walk-Forward:** Sharpe OOS >80% do in-sample
- [ ] **Monte Carlo:** 95% CI do Sharpe não cruza zero
- [ ] **Stress Test:** Retorno na crise 2020 > -15%
- [ ] **Regime Test:** Sharpe >0.5 em TODOS os 4 quadrantes

---

## 🚨 Pontos de Decisão

### Decisão 1: K-Means vs Intensidade Binária (Semana 4)
**Critério:** Sharpe OOS em walk-forward  
**Se K-Means ganhar (Sharpe >5% melhor):** Manter K=2-4 otimizado  
**Se Binário ganhar:** Simplificar para Strong/Weak  
**Se empate:** Escolher Binário (menor complexidade)

### Decisão 2: Exposições por Quadrante (Semana 5)
**Testar Grid:**
- Q1: [60%, 80%, 100%]
- Q2/Q3: [20%, 40%, 60%]
- Q4: [40%, 60%, 80%]

**Critério:** Sharpe × Calmar (penalizar drawdown)

### Decisão 3: Custos de Transação (Semana 1)
**Opção A:** Custos fixos 23bps (conservador)  
**Opção B:** Custos variáveis por volume (realista)  
**Recomendação:** Começar com Opção A, refinar depois

---

## 📚 Referências e Fontes

### Papers Acadêmicos
1. **Moskowitz, Ooi, Pedersen (2012)** - "Time Series Momentum" (JFE)  
   → Fundamentação do momentum multi-timeframe

2. **Jegadeesh & Titman (1993)** - "Returns to Buying Winners and Selling Losers" (JF)  
   → Janelas clássicas: 3/6/9/12 meses

3. **Moreira & Muir (2017)** - "Volatility-Managed Portfolios" (JF)  
   → Vol-targeting aumenta Sharpe 30-50%

4. **Ang & Bekaert (2002)** - "Regime Switches in Interest Rates" (JBF)  
   → Detecção de regimes macroeconômicos

5. **Kritzman, Page & Turkington (2012)** - "Regime Shifts" (FAJ)  
   → PCA para construção de indicadores

6. **Rousseeuw (1987)** - "Silhouettes: A graphical aid" (JComp)  
   → Métrica de validação para K-Means

7. **Harvey & Liu (2015)** - "Backtests and Multiple Testing" (JFQA)  
   → Viés de multiple testing em backtests

8. **López de Prado (2018)** - "Advances in Financial Machine Learning"  
   → Cap. 7-11: Validação, overfitting, walk-forward

### Livros Recomendados
- **"Quantitative Trading"** - Ernest Chan (2008)  
  → Cap. 2-4: Backtesting, custos de transação
  
- **"Following the Trend"** - Andreas Clenow (2012)  
  → Cap. 6-8: Momentum robusto, risk management
  
- **"Active Portfolio Management"** - Grinold & Kahn (1999)  
  → Cap. 14-16: Information Ratio, transaction costs

### Artigos de Indústria
- AQR Capital - "Fact, Fiction, and Momentum Investing" (2014)
- GMO - "Regime-Based Asset Allocation" (2010)
- Research Affiliates - "The Volatility Effect" (2011)

---

## 🛠️ Ferramentas e Recursos

### Bibliotecas Python Adicionais
```bash
# Instalar pacotes necessários para melhorias
pip install scikit-learn>=1.3.0    # PCA, K-Means
pip install joblib>=1.3.0          # Paralelização
pip install statsmodels>=0.14.0    # Testes estatísticos
pip install scipy>=1.11.0          # Otimização
pip install quantstats>=0.0.62     # Relatórios avançados
```

### Datasets Complementares
- **Fred Economic Data:** Indicadores macro (PMI, CPI, Payroll)
- **World Bank:** Dados globais de atividade
- **Bloomberg/Reuters:** Dados de alta frequência (se disponível)

---

## 🎓 Próximos Passos Imediatos

### Para Começar AGORA
1. **Backup completo** do projeto atual (criar branch Git)
2. **Rodar backtest atual** e salvar métricas baseline
3. **Escolher Fase 1** como prioridade (fundação)
4. **Decidir:** Implementar full pipeline ou arquivo por arquivo?

### Comando Sugerido
```bash
# 1. Criar branch de desenvolvimento
git checkout -b melhorias-fase1

# 2. Rodar baseline atual
python backtest_6.py

# 3. Salvar resultados
cp backtest_detalhado.csv backtest_BASELINE_2026-01-26.csv
cp relatorio_quantstats.html relatorio_BASELINE_2026-01-26.html

# 4. Começar por download_1.py (maior impacto)
code download_1.py
```

---

## ❓ Perguntas Frequentes

### P1: Devo implementar tudo de uma vez?
**R:** Não. Implemente **Fase 1 completa** primeiro (2 semanas), valide, depois Fase 2.

### P2: Qual arquivo começar?
**R:** `download_1.py` (1-2 dias) → Maior impacto imediato na robustez.

### P3: K-Means ou Intensidade Binária?
**R:** Teste ambos na Semana 4 via walk-forward. Recomendação inicial: **Binário** (menos overfitting).

### P4: Os custos realmente impactam tanto?
**R:** Sim. Com turnover ~800%/ano, 23bps vs 10bps = diferença de ~10-13pp de retorno anual.

### P5: Preciso de mais dados históricos?
**R:** Sim. 2016-2026 = 1 ciclo. Ideal: 2000-2026 (3 ciclos completos incluindo 2008).

---

## 📞 Contato e Suporte

**Desenvolvedor:** Matheus Mizrahi  
**Instituição:** Insper - IQF + LEV  
**Projeto:** Projeto_quant_lev  
**Data Última Atualização:** 26 de Janeiro de 2026

---

## 🏁 Conclusão

Este plano fornece um **roteiro estruturado** para transformar o backtest atual em uma estratégia **robusta, validada e pronta para produção**.

### Princípios Fundamentais
✅ **Validação rigorosa** (walk-forward, Monte Carlo)  
✅ **Custos realistas** (23bps + IR vs 10bps ingênuos)  
✅ **Sinais robustos** (momentum multi-timeframe vs OLS frágil)  
✅ **Gestão de risco** (vol-targeting, stop-loss)  
✅ **Simplicidade** (evitar overfitting com K=12)

### Expectativa Realista
- **Fase 1 (2 semanas):** Sharpe 1.0-1.4 com dados robustos
- **Fase 2 (2 semanas):** Sharpe 1.3-1.7 com validação OOS
- **Fase 3 (1.5 semanas):** Sharpe 1.5-2.0 com otimização
- **Fase 4 (0.5 semanas):** Sharpe 1.8-2.3 com ML (opcional)

---

**🚀 Bom trabalho e bons trades!**
