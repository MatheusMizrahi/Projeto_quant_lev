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
- **Ações:** `S&P 500`
- **Renda Fixa:** `T-Bill`, `TLT` ou `HYG`  
- **Commodities:** `Petróleo (WTI)`
- **Moedas:** `DXY (Dollar Index)`

> Esses ativos servem como termômetros dos principais vetores de risco: crescimento, inflação, política monetária e liquidez global.

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

> _“Os mercados são como organismos interligados — compreender seus fluxos cruzados é compreender o próprio ciclo macroeconômico.”_  
> — **LEV Quant Research Lab**
