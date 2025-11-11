# Projeto LEV


# 📈 Market Intersection Analysis – Fase 1  
**LEV Asset Management | Quantitative Research Lab**

---

## 🧩 Visão Geral do Projeto

O objetivo central é identificar, através de uma análise **quantitativa e objetiva da precificação dos mercados globais**, o **regime macroeconômico vigente** — sem recorrer a indicadores econômicos tradicionais.  
A metodologia parte de **tendências de preço em quatro classes de ativos** e evolui até a **definição automática do regime e intensidade via clusterização (K-Means)**.

---

## 🚀 Estrutura Geral da Fase 1

### 1. 🧮 **Seleção de Ativos (Passo 1)**
Definimos os proxies que representarão o comportamento de cada mercado global:

#### **Portfolio de Ativos Globais:**
- **Ações Desenvolvidas:** `S&P 500 (^GSPC)`
- **Ações Emergentes:** `MSCI Emerging Markets (EEM)`
- **Renda Fixa Governamental:** `US Treasury 10Y (^TNX)`
- **Renda Fixa Corporativa:** `High Yield ETF (HYG)`
- **Moedas:** `DXY - Dollar Index (DX-Y.NYB)`
- **Commodities Energéticas:** `Petróleo WTI (CL=F)`
- **Commodities Metálicas:** `Ouro (GC=F)`

> Esses ativos servem como termômetros dos principais vetores de risco: crescimento, inflação, política monetária e liquidez global.

#### **Construção dos Índices Compostos:**

```
📊 INFLAÇÃO = 🛢️ Oil (40%) + 🟡 Gold (30%) + 📈 US10Y (20%) - 💵 DXY (10%)
              ├─ Realizada (commodities)
              ├─ Esperada (bonds)
              └─ Contexto monetário (dólar)

📈 ATIVIDADE = 📊 SP500 (35%) + 🌏 EM (25%) + 💳 HYG (25%) + 📈 US10Y (10%) - 💵 DXY (5%)
               ├─ Crescimento desenvolvidos
               ├─ Crescimento emergentes
               ├─ Condições de crédito
               └─ Ambiente monetário
```

**Lógica dos Pesos:**
- **Inflação**: Petróleo domina (40%) por ser o driver principal de custos, seguido por Ouro (30%) como hedge tradicional
- **Atividade**: SP500 lidera (35%) como proxy de crescimento desenvolvido, complementado por Emergentes (25%) e crédito corporativo (25%)
- **Pesos Negativos**: DXY tem relação inversa (dólar forte → commodities caem → inflação baixa / crescimento EM fraco)

---

### 2. 📊 **Cálculo das Regressões (Passos 2 e 3)**
Para cada ativo:
- Rodamos uma **regressão linear simples** no tempo para estimar a **tendência (inclinação da reta)**.  
- Duas métricas são testadas:
  1. Apenas a **inclinação (slope)** da regressão.  
  2. A **inclinação ponderada pelo R²**, para capturar confiabilidade do ajuste.

> 🔍 A ideia é medir se cada ativo está em “tendência de alta ou baixa” e com que força estatística.

---

### 3. 🧭 **Definição do Quadrante (Passo 4)**
Com base nos sinais das regressões, o modelo define o **regime macro** via lógica condicional (`if/else`):

| Cenário | S&P 500 | TLT | DXY | Petróleo |
|----------|----------|-----|-----|-----------|
| **Goldilocks (Perfeito)** | ↑ | ↑ | ↓ | ↓ |
| **Reflação** | ↑ | ↓ | — | ↑ |
| **Stagflação** | ↓ | ↓ | ↑ | ↑ |
| **Desinflação / Contração** | ↓ | ↑ | ↑ | ↓ |

> Essa classificação é puramente quantitativa, baseada em preço, e independe de julgamentos econômicos.

---

### 4. 🧠 **Clusterização via K-Means (Passo 9)**
Após definir o quadrante, aplicamos o algoritmo **K-Means** para identificar **a intensidade do regime** (forte, moderado ou fraco).  
Isso remove a necessidade de julgamento humano, criando uma **escala objetiva de força de sinal**.

---

### 5. ⚙️ **Definição da Trading Rule (Passo 6)**
Cada quadrante possui uma **regra de alocação automática**:
- O modelo monta um **portfólio ótimo para o regime identificado**, definindo posições **long/short** e pesos percentuais.  
- A **intensidade do sinal** (vinda do K-Means) ajusta o tamanho das posições.

> 💡 Exemplo de hedge: comprar o ativo com melhor tendência (“melhor regressor”) e vender o pior — reduzindo exposição direcional.

---

### 6. 🧩 **Hedge Dinâmico**
Parte da trading rule é o **hedge adaptativo**, que alterna exposição conforme o quadrante:
- Em “Goldilocks”: favorece ativos de risco (ações, high yield).
- Em “Stagflação”: prioriza commodities e reduz risco direcional.
- Em “Desinflação”: privilegia bonds e dólar.

---

### 7. 🔁 **Backtesting e Ajuste (Passos 7 e 8)**
Por fim, as regras são testadas historicamente:
- Simulações de janelas **(5 a 20 dias após o sinal)**.
- Avaliação de métricas como **retorno médio, drawdown, hit ratio e Sharpe ratio**.
- Ajuste iterativo dos parâmetros de regressão e clusterização.

---

## 🎯 Objetivo Final
> Criar um **“Book Macro Sistemático”**, capaz de identificar regimes e gerar sinais de alocação automaticamente — base para um portfólio macro estilo hedge fund.

---

## 📘 Referências Conceituais
- **John J. Murphy – _Trading with Intermarket Analysis_ (2013)**  
- **Leitura complementar:** _Market Intersection Analysis Framework (LEV, 2025)_  
- **Temas-base:** inter-relação entre ativos, regime de inflação e crescimento, correlações dinâmicas e ciclo de alocação global.

---

## 🧭 Próximos Passos
- [ ] Implementar regressões e cálculo de slope × R².  
- [ ] Criar o mapeamento condicional dos quadrantes.  
- [ ] Iniciar o pipeline de clusterização K-Means.  
- [ ] Validar regimes históricos e intensidade dos sinais.  

---

## 📍 STATUS DO PROJETO
**Última atualização: 04 de Novembro de 2025**

### ✅ **Implementado até o momento:**

#### **Fase 1: Coleta e Preparação de Dados**
- ✅ **`dowload.py`**: Script de download automático de dados via yfinance
  - Ativos Globais: SP500, MSCI EM, DXY, US 10Y, High Yield ETF, Oil WTI, Gold
  - Período: Out/2020 a Out/2025
  - Salvamento em CSV (`data_prices.csv`)

#### **Fase 2: Análise de Regressões Lineares**
- ✅ **`Regressoes_lineares.py`**: Classe `AnalisadorRegressao`
  - Regressão linear de cada ativo vs. tempo
  - Cálculo de Beta (tendência), R², p-value e score ponderado
  - Score = sinal(β₁) × √R² (apenas se significativo)
  - Estrutura modular e reutilizável
  - Modo verbose/silencioso para importação

#### **Fase 3: Classificação de Quadrantes**
- ✅ **`Definicao_quadrante.py`**: Classe `ClassificadorQuadrantes`
  - Cálculo de proxies compostas globais:
    - **Inflação** = Oil_WTI (40%) + Gold (30%) + US_10Y (20%) - DXY (10%)
    - **Atividade** = SP500 (35%) + MSCI_EM (25%) + HYG (25%) + US_10Y (10%) - DXY (5%)
  - Mapeamento em sistema de coordenadas (Inflação × Atividade)
  - Identificação automática dos 4 quadrantes macroeconômicos
  - Limiares ajustáveis (fixos ou percentis históricos)

### 🚧 **Em desenvolvimento / Próximas etapas:**

#### **Fase 4: Clusterização e Intensidade (K-Means)**
- [ ] Implementar K-Means para classificar intensidade do regime (forte/moderado/fraco)
- [ ] Definir features para clusterização (scores, volatilidades, correlações)
- [ ] Validar número ótimo de clusters (Elbow Method / Silhouette Score)

#### **Fase 5: Trading Rules e Alocação**
- [ ] Criar regras de alocação para cada quadrante
- [ ] Implementar sistema de hedge dinâmico
- [ ] Definir pesos e posições long/short por regime
- [ ] Ajustar tamanho de posição baseado na intensidade (K-Means)

#### **Fase 6: Backtesting e Validação**
- [ ] Implementar engine de backtesting
- [ ] Testar janelas de rebalanceamento (5, 10, 20 dias)
- [ ] Calcular métricas de performance:
  - Retorno acumulado
  - Sharpe Ratio
  - Drawdown máximo
  - Hit ratio
  - Turnover
- [ ] Otimização de hiperparâmetros (limiares, pesos, janelas)

#### **Fase 7: Visualização e Reporting**
- [ ] Criar dashboards interativos com plotly/dash
- [ ] Gráficos de regime ao longo do tempo
- [ ] Heatmaps de correlação entre ativos
- [ ] Relatórios automatizados de performance

#### **Fase 8: Deploy e Automação**
- [ ] Automatizar atualização diária de dados
- [ ] Sistema de alertas para mudanças de regime
- [ ] API para consulta de regime atual
- [ ] Integração com sistemas de execução (futuro)

---

### 🎯 **Marco Atual:**
> Estamos na **transição entre Fase 3 e Fase 4**. A base de análise quantitativa está completa — conseguimos identificar regimes macroeconômicos a partir dos preços. O próximo passo crítico é adicionar inteligência sobre a **força/convicção** de cada sinal via clusterização.

---

> _"Os mercados são como organismos interligados — compreender seus fluxos cruzados é compreender o próprio ciclo macroeconômico."_  
> — **LEV Quant Research Lab**
