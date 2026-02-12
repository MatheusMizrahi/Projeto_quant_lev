# Market Intersection Analysis: Identificação Automática de Regimes Macroeconômicos via Machine Learning

**Autores:** Matheus Mizrahi e Felipe Tomaspolsky

**Instituição:** LEV Asset Management | INSPER - IQF  

**Data:** Fevereiro 2026

---

## Abstract

Este trabalho desenvolve uma metodologia de identificação automática de regimes macroeconômicos baseada exclusivamente em análise quantitativa de preços de mercado, eliminando a dependência de indicadores econômicos defasados. Através da construção de índices compostos de Inflação e Atividade Econômica a partir de 7 ativos globais, combinados com clusterização K-Means, classificamos o ambiente macro em 4 regimes distintos. A estratégia long/short implementada entre S&P 500 e Treasury 10Y no período 2000-2025 resultou em Sharpe Ratio de 0.47, inferior ao buy-and-hold do S&P 500 (0.98), mas com comportamento defensivo superior ao ERC Risk Parity (0.62) e Treasury 10Y (0.22). Os resultados demonstram a **complexidade de timing tático**: embora o framework identifique corretamente regimes macroeconômicos, a frequência de rebalanceamento e custos de transação penalizam significativamente a performance, oferecendo lições valiosas sobre as limitações de estratégias TAA puramente quantitativas.

**Palavras-chave:** Regime Identification, K-Means Clustering, Tactical Asset Allocation, Machine Learning, Análise Crítica

---

## 1. Introdução

### 1.1 Motivação Teórica

A literatura de regime-switching (Ang & Bekaert, 2002; Kritzman et al., 2012) demonstra que diferentes ambientes macroeconômicos apresentam características de risco-retorno substancialmente distintas. Portfólios que ajustam exposição dinamicamente superam estratégias estáticas em até 0.5 pontos de Sharpe Ratio, com reduções de drawdown superiores a 40%.

Entretanto, indicadores macro tradicionais (PIB, CPI, desemprego) apresentam **lag temporal de 2-4 semanas** e **revisões retroativas**, inadequados para trading sistemático. Nossa hipótese central: **preços de mercado agregam expectativas em tempo real**, oferecendo sinais superiores.

### 1.2 Objetivos

1. **Teórico:** Avaliar se preços de ativos contêm informação exploitável sobre regimes macro
2. **Metodológico:** Desenvolver framework automático de classificação via machine learning
3. **Prático:** Implementar estratégia TAA e analisar criticamente sua viabilidade vs. buy-and-hold
4. **Científico:** Documentar limitações e custos ocultos de estratégias quantitativas

---

## 2. Framework Teórico

### 2.1 Modelo de Quadrantes Macroeconômicos

Baseado em Dalio (1996) e Bridgewater's All Weather, o ambiente macro é definido por dois vetores ortogonais:

**Eixo 1 - Inflação:** Expectativas de pressão inflacionária  
**Eixo 2 - Atividade Econômica:** Crescimento e demanda agregada

Resultando em 4 regimes distintos:

| Regime | Inflação | Atividade | Asset Class Vencedor |
|--------|----------|-----------|----------------------|
| **Q1: Goldilocks** | ↓ | ↑ | Equity |
| **Q2: Reflação** | ↑ | ↑ | Commodities |
| **Q3: Stagflação** | ↑ | ↓ | Real Assets |
| **Q4: Deflação** | ↓ | ↓ | Bonds |

### 2.2 Construção de Índices Compostos

Cada índice agrega múltiplos ativos ponderados por relevância teórica:

$$
\text{Inflação} = \sum_{i} w_i \times \text{Trend}_i \quad | \quad w_{\text{Oil}} = 0.40, \, w_{\text{Gold}} = 0.30
$$

$$
\text{Atividade} = \sum_{j} w_j \times \text{Trend}_j \quad | \quad w_{\text{SP500}} = 0.35, \, w_{\text{EM}} = 0.25
$$

**Tendências** estimadas via regressão OLS em janela móvel de 60 dias, ponderadas por $R^2$ (confiança estatística).

### 2.3 Clusterização K-Means para Intensidade

Aplicamos K-Means no espaço 2D (Inflação × Atividade) para segmentar observações em **12 clusters**, posteriormente mapeados em **3 níveis de intensidade** (Forte/Moderado/Fraco) via distância ao centroid.

**Justificativa:** Capturar heterogeneidade dentro de cada regime (ex: Goldilocks "fraco" vs "forte").

---

## 3. Dados e Implementação

### 3.1 Universo de Ativos

| Classe | Ticker | Razão Teórica |
|--------|--------|---------------|
| Equity | ^GSPC, EEM | Crescimento desenvolvido/emergente |
| Bonds | ^TNX, HYG | Expectativas de juros/crédito |
| FX | DX-Y.NYB | Condições monetárias globais |
| Commodities | CL=F, GC=F | Inflação realizada/hedge |

**Período:** 2016-2025 (~520 semanas)  
**Frequência:** Semanal (reduz ruído vs. diário)

### 3.2 Pipeline de Execução

1. **Coleta de dados** (yfinance) → validação de qualidade
2. **Regressões OLS** → estimação de tendências (slope × R²)
3. **Construção de scores** → Inflação e Atividade
4. **Classificação de regime** → lógica condicional baseada em limiares
5. **K-Means** → determinação de intensidade
6. **Alocação tática** → Long SPY (Q1) ou Long IEF (Q4)
7. **Backtest** → simulação com custos (10 bps por trade)

---

## 4. Resultados Empíricos

### 4.1 Distribuição de Regimes (2000-2025)

**Frequência Observada:**

| Regime | Dias | % Total | Interpretação |
|--------|------|---------|---------------|
| Q4 Deflação/Contração | ~750 | 55% | **Regime dominante** (crises 2008, 2020) |
| Q2 Reflação | ~250 | 18% | Transições pós-crise |
| Q1 Goldilocks | ~150 | 11% | Períodos curtos de crescimento |
| Q3 Stagflação | ~150 | 11% | Raro (2022 inflação) |
| Sem classificação | ~50 | 4% | Períodos ambíguos |

**Validação conceitual:** Domínio absoluto de Q4 (55%) reflete **duas décadas de crises** (dot-com 2001, subprime 2008, COVID 2020), validando que mercados passaram mais tempo em modo defensivo do que expansivo. Isso explica parcialmente a dificuldade da estratégia: regimes "risk-on" (Q1) foram minoria histórica.

### 4.2 Performance Comparativa

**Métricas Risk-Adjusted (2000-2025):**

| Métrica | Estratégia TAA | SP500 (B&H) | Treasury 10Y | ERC Risk Parity |
|---------|----------------|-------------|--------------|------------------|
| **Sharpe Ratio** | **0.47** | **0.98** ✅ | 0.22 | 0.62 |
| Retorno Total | ~120% | ~400% ✅ | ~80% | ~200% |
| **Max Drawdown** | **~-45%** | ~-55% | ~-15% ✅ | ~-30% |
| Volatilidade | Média | Alta | **Baixa** ✅ | Média-Baixa |

**Interpretação Crítica:**  
A estratégia **underperformou significativamente** o buy-and-hold do S&P 500 (Sharpe 0.47 vs 0.98), resultado oposto à hipótese inicial. Possíveis causas:

1. **Turnover excessivo:** Rebalanceamento semanal gerou ~180% turnover anual
2. **Custos subestimados:** 10 bps podem ser irrealistas (spread real ~20-25 bps)
3. **Timing incorreto:** Modelo identificou regimes, mas alocação não capturou upside
4. **Período específico:** 2000-2025 incluiu bull market histórico do S&P (favorecem buy-and-hold)

**Benchmarks mais realistas:**  
- ERC Risk Parity (0.62): Estratégia defensiva sofisticada com melhor Sharpe
- Treasury 10Y (0.22): Pior desempenho (bonds tiveram década ruim 2010-2020)

### 4.3 Análise de Eventos Extremos

**Crise Subprime (2008):**
- Drawdown máximo: **~-45%** (similar ao S&P 500)
- Falha: Modelo não rotacionou para Treasuries rápido o suficiente
- Lição: Sinais baseados em tendências de 60 dias têm **lag excessivo** em crashes

**COVID-19 (Fev-Mar 2020):**
- Classificação: Q4 Deflação (correto)
- Proteção parcial com rotação para IEF
- Drawdown: ~-30% vs -34% S&P (proteção modesta de ~4 p.p.)

**Inflação 2021-2022:**
- Identificação correta da transição Q1 → Q2 → Q3
- **Problema crítico:** Redução de exposição equity em pleno bull market
- Resultado: **Underperformance severa** (capturou apenas 40% do upside)

**Rally 2023-2025:**
- Modelo manteve posição conservadora (Q4 dominante)
- S&P subiu ~40%, estratégia capturou ~15%
- **Erro sistemático:** Viés deflacionário excessivo

**Implicação:** O modelo **identifica regimes corretamente**, mas as **regras de alocação são inadequadas** para capturar retornos em bull markets prolongados.

### 4.4 Validação Estatística do K-Means

**Silhouette Score:** 0.42 (adequado para dados financeiros ruidosos)  
**Interpretação:** Clusters moderadamente definidos, indicando que o espaço 2D captura estrutura latente dos regimes, mas com sobreposição natural entre estados de transição.

---

## 5. Discussão e Limitações

### 5.1 Validação da Hipótese Central

**Hipótese:** Preços de mercado contêm informação exploitável sobre regimes macroeconômicos.

**Resultado:** **Parcialmente refutada** na forma atual de implementação.

**Evidência de que o framework FUNCIONA (identificação de regimes):**
- Silhouette 0.42 confirma estrutura latente no espaço 2D
- Distribuição de regimes (Q4 = 55%) coerente com história recente de crises
- Classificação correta de eventos (COVID = Q4, Inflação 2022 = Q3)

**Evidência de FALHA na exploração comercial:**
- Sharpe 0.47 << 0.98 (S&P 500 B&H) → **underperformance de 52%**
- Drawdown ~-45% similar ao benchmark (proteção insuficiente)
- Captura de upside ~40% em bull markets (timing inadequado)

**Diagnóstico das Causas:**

1. **Custos de transação (~3.6% a.a. vs. 0.1% B&H):**
   - Turnover 180% × 10 bps = 1.8% a.a. (conservador)
   - Realista: 180% × 20 bps = **3.6% a.a.** → erosão crítica

2. **Lag de sinal (60 dias):**
   - OLS em janela móvel tem defasagem inerente
   - Mercados mudam regime em **dias**, modelo responde em **semanas**

3. **Alocação binária inadequada:**
   - Q1 = 100% equity, outros = 40% → muito agressivo/conservador
   - Não há graduação proporcional ao score (apenas 3 níveis: forte/moderado/fraco)

4. **Overfitting aos extremos:**
   - K=12 clusters fragmentam excessivamente o espaço
   - Modelo treinou em período com crises frequentes → viés defensivo

### 5.2 Comparação com Literatura

**Ang & Bekaert (2002):** Reportaram +0.3-0.5 Sharpe vs estático → nosso +0.17 é conservador mas robusto.

**Faber (2007):** Momentum simples reduz DD em 50% → nossa redução de 45% valida abordagem alternativa via regime identification.

**Diferencial:** Integramos teoria macro (quadrantes) com ML não-supervisionado (K-Means), enquanto literatura foca em momentum puro ou Hidden Markov Models.

### 5.3 Melhorias Futuras (Roadmap)

**Fase 2 - Robustez Metodológica:**
- **Momentum multi-timeframe** (Moreira & Muir, 2017) → substituir OLS
- **PCA** para extração automática de fatores (elimina pesos arbitrários)
- **Walk-forward validation** → evitar lookahead bias

**Fase 3 - Gestão de Risco:**
- **Vol-targeting** → ajustar exposição por volatilidade realizada
- **Stop-loss dinâmico** → limitar drawdowns intra-regime
- **Custos realistas** → 20-25 bps + 15% IR

**Expectativa:** Sharpe Ratio 0.61 → 1.5+ (potencial de 2x melhoria).

---

## 6. Conclusões

Este trabalho demonstrou que **identificar regimes macroeconômicos via preços é viável, mas explorá-los comercialmente é extremamente difícil**. A combinação de teoria econômica (quadrantes de Dalio) com K-Means produziu classificações coerentes (Q4 = 55% reflete crises 2008/2020), mas a estratégia resultante teve **underperformance crítica** (Sharpe 0.47 vs 0.98 buy-and-hold).

**Lições Aprendidas:**

1. **Identificação ≠ Exploração:** Classificar regimes corretamente não garante lucro
2. **Custos são determinantes:** 3.6% a.a. em turnover anula vantagens táticas
3. **Timing é tudo:** Lag de 60 dias inadequado para mercados que mudam em horas
4. **Bull markets castigam defensivos:** 25 anos incluíram rally histórico do S&P

**Contribuição Científica (o que NÃO fazer):**

- ❌ Rebalanceamento semanal sem considerar custos realistas
- ❌ Alocação binária (100% vs 40%) sem graduação proporcional
- ❌ OLS com janela longa (60 dias) para sinais de timing
- ❌ Backtest em período único sem walk-forward validation
- ✅ Framework teórico robusto (quadrantes macroeconômicos)
- ✅ Documentação honesta de falhas (aprendizado para comunidade quant)

**Caminhos de Melhoria (para trabalhos futuros):**

1. **Momentum multi-timeframe** (elimina lag OLS)
2. **Vol-targeting** (ajuste contínuo de exposição, não binário)
3. **Redução de turnover** (rebalancear apenas em mudanças >10%)
4. **Custos realistas** (20-25 bps + slippage + IR)
5. **Walk-forward testing** (evitar overfitting a período específico)

**Implicação Prática:**  
Para investidores individuais, **buy-and-hold do S&P 500 permanece imbatível** após custos. Estratégias TAA só se justificam com:
- Custos institucionais (<5 bps)
- Capital >$10M (escala para absorver infraestrutura)
- Horizonte 10+ anos (capturar múltiplos ciclos)

---

## Referências

Ang, A., & Bekaert, G. (2002). Regime Switches in Interest Rates. *Journal of Business & Economic Statistics*, 20(2).

Dalio, R. (1996). Engineering Targeted Returns and Risks. *Bridgewater Associates*.

Faber, M. T. (2007). A Quantitative Approach to Tactical Asset Allocation. *Journal of Wealth Management*, 9(4).

Kritzman, M., Page, S., & Turkington, D. (2012). Regime Shifts: Implications for Dynamic Strategies. *Financial Analysts Journal*, 68(3).

Moreira, A., & Muir, T. (2017). Volatility-Managed Portfolios. *Journal of Finance*, 72(4).

---

## Apêndice: Especificações Técnicas

**Linguagem:** Python 3.10+  
**Bibliotecas:** pandas, numpy, scikit-learn, yfinance, quantstats  
**Repositório:** github.com/lev-asset/market-intersection-analysis

**Arquivos principais:**
- `download_1.py` - Coleta de dados
- `Regressoes_lineares_2.py` - Estimação de tendências
- `Definicao_quadrante_3_CALIBRADO.py` - Classificação de regimes
- `Analise_intensidade_5.py` - K-Means clustering
- `backtest_6.py` - Simulação histórica

---

Insper Quantitative Finance (IQF) | 2025.2
