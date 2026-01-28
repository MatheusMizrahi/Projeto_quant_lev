# Projetos 2025.1 – LEV Quant Strategy

---

## 🎯 LEV Quant Strategy
**Market Intersection Analysis**

### Membros
- Matheus Mizrahi
- LEV Asset Management | Quantitative Research Lab

---

## 📝 Descrição

O **Projeto LEV Quant Strategy** desenvolve um sistema de **alocação tática automatizada (TAA)** baseado em análise quantitativa de múltiplos mercados globais para identificar **regimes macroeconômicos** e realizar **trading sistemático long/short** entre classes de ativos.

Utilizando dados históricos de preço de 7 ativos globais (S&P 500, Emerging Markets, DXY, Treasury 10Y, High Yield, Petróleo WTI e Ouro), o modelo estima **tendências via regressão linear**, constrói **índices compostos de Inflação e Atividade Econômica**, identifica automaticamente o **regime macro vigente** (Goldilocks, Reflação, Stagflação ou Desinflação) através de lógica condicional, e aplica **clusterização K-Means** para determinar a **intensidade do sinal** (forte, moderado ou fraco), ajustando dinamicamente as posições do portfólio.

---

## 💡 Conceitos Envolvidos

- **Análise Quantitativa de Múltiplos Mercados**
- **Regressão Linear e Séries Temporais**
- **Machine Learning (K-Means Clustering)**
- **Tactical Asset Allocation (TAA)**
- **Long/Short Equity Strategies**
- **Regime Macroeconômico Quantitativo**
- **Backtesting e Análise de Performance**
- **Gestão de Risco e Custos de Transação**

---

## 🏗️ Arquitetura do Sistema

### **Fase 1 - Market Intersection Analysis** ✅ IMPLEMENTADO
1. **Coleta de Dados:** Download de 7 ativos via Yahoo Finance (2016-presente)
2. **Análise de Tendências:** Regressão linear OLS em janela móvel de 60 dias
3. **Construção de Índices Compostos:**
   - Índice de Inflação = Oil (40%) + Gold (30%) + US10Y (20%) - DXY (10%)
   - Índice de Atividade = SP500 (35%) + EM (25%) + HYG (25%) + US10Y (10%) - DXY (5%)
4. **Classificação de Regimes:** Identificação automática via lógica condicional
   - Goldilocks (crescimento + inflação baixa)
   - Reflação (crescimento + inflação alta)
   - Stagflação (contração + inflação alta)
   - Desinflação (contração + inflação baixa)
5. **Clusterização K-Means:** Determinação de intensidade do sinal (12 clusters)
6. **Trading Rules:** Alocação long/short SP500 vs Treasury 10Y
7. **Backtest:** Simulação histórica com custos de transação (10 bps)

### **Fase 2 - Otimização e Robustez** ⏳ EM PLANEJAMENTO
- Substituição de OLS por **Momentum Multi-Timeframe**
- **PCA (Principal Component Analysis)** para redução dimensional
- **Validação Out-of-Sample** (walk-forward, cross-validation)
- **Vol-Targeting** para ajuste dinâmico de exposição
- Custos realistas (20-25 bps + 15% IR)
- **Stop-loss** dinâmico e trailing stop
- Redução de clusters (K=4 ou K=6)

### **Fase 3 - Machine Learning Avançado** 🔮 FUTURO
- **Ensemble Methods** (Random Forest, Gradient Boosting)
- **LSTM/GRU** para previsão de tendências
- **Adaptive Thresholds** via otimização bayesiana
- Expansão para 15+ ativos
- Histórico desde 2000 (3+ ciclos econômicos)

---

## 📊 Resultados Preliminares

### Performance (Backtest 2016-2025)
- **Sharpe Ratio:** ~0.6-0.8 (baseline)
- **Max Drawdown:** ~15-20%
- **Win Rate:** ~55-60%
- **Frequência:** Rebalanceamento semanal

### Próximas Metas
- **Sharpe Target:** >1.5 (após otimizações Fase 2)
- **Drawdown Target:** <12%
- **Redução de custos:** Implementação de custos realistas
- **Robustez:** Validação em múltiplos períodos

---

## 🛠️ Stack Tecnológico

- **Python 3.13**
- **Pandas / NumPy** (manipulação de dados)
- **Statsmodels** (regressões OLS)
- **Scikit-learn** (K-Means clustering)
- **Matplotlib / Seaborn** (visualizações)
- **yFinance** (download de dados)
- **QuantStats** (métricas de performance)

---

## 📚 Documentação Gerada

- **README.md:** Visão geral e metodologia completa
- **GLOSSARIO_FINANCEIRO.md:** 100+ termos técnicos
- **PLANO_MELHORIAS.md:** Análise crítica e roadmap de otimizações
- **GUIA_KMEANS_PASSO_A_PASSO.md:** Tutorial de clusterização
- **TIMING_ESTRATEGIA.md:** Análise de entry/exit timing

---

## 🎯 Diferenciais do Projeto

✅ **Abordagem 100% Quantitativa:** Sem julgamentos subjetivos ou análise fundamental  
✅ **Market-Based:** Decisões baseadas apenas em preços de mercado  
✅ **Regime Detection:** Identificação automática de ambientes macro  
✅ **Clusterização:** Intensidade de sinal via ML não-supervisionado  
✅ **Long/Short:** Estratégia market-neutral com hedge dinâmico  
✅ **Backtesting Rigoroso:** Custos de transação e validação histórica  
✅ **Código Modular:** Separação clara entre dados, análise e execução  
✅ **Documentação Extensiva:** 600+ linhas de documentação técnica

---

## 🔍 Próximos Passos

### Curto Prazo (2 semanas)
1. Implementar Momentum multi-timeframe
2. Corrigir custos de transação (20-25 bps)
3. Vol-targeting dinâmico
4. Expandir dataset para 15 ativos

### Médio Prazo (1 mês)
5. PCA para construção de índices
6. Walk-forward validation
7. Stop-loss adaptativo
8. Reduzir clusters para K=4-6

### Longo Prazo (2 meses)
9. Ensemble ML (Random Forest)
10. Otimização bayesiana de hiperparâmetros
11. Dashboard interativo
12. Paper acadêmico

---

**Deep Market Analysis: Quantitative Approach to Macro Regime Detection**  
*Matheus Mizrahi & LEV Asset Management*

---
