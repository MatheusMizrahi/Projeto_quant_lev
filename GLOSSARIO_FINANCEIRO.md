# 📖 Glossário de Termos do Mercado Financeiro

**Projeto:** Estratégia Quantitativa Long/Short - SP500 vs Treasury 10Y  
**Propósito:** Referência rápida para termos técnicos utilizados no projeto  
**Última atualização:** Janeiro 2026

---

## 📂 Índice por Categoria

1. [Instrumentos Financeiros](#1-instrumentos-financeiros)
2. [Custos de Transação](#2-custos-de-transação)
3. [Métricas de Performance](#3-métricas-de-performance)
4. [Conceitos de Portfolio](#4-conceitos-de-portfolio)
5. [Regimes Macroeconômicos](#5-regimes-macroeconômicos)
6. [Risco e Volatilidade](#6-risco-e-volatilidade)
7. [Backtesting](#7-backtesting)
8. [Execução e Trading](#8-execução-e-trading)
9. [Indicadores e Análise](#9-indicadores-e-análise)

---

## 1. Instrumentos Financeiros

### **ETF (Exchange Traded Fund)**
Fundo de investimento negociado em bolsa como se fosse uma ação. Replica índices, setores ou classes de ativos com alta liquidez e baixo custo.

### **SPY (SPDR S&P 500 ETF Trust)**
ETF mais líquido do mundo que replica o índice S&P 500. Permite exposição às 500 maiores empresas americanas através de um único ativo.

### **IEF (iShares 7-10 Year Treasury Bond ETF)**
ETF que replica títulos do Tesouro americano com vencimento entre 7-10 anos. Usado como proxy para Treasury 10Y.

### **TLT (iShares 20+ Year Treasury Bond ETF)**
ETF de títulos do Tesouro americano de longo prazo (20+ anos). Maior sensibilidade a mudanças nas taxas de juros.

### **S&P 500**
Índice das 500 maiores empresas americanas por capitalização de mercado. Principal benchmark de ações dos EUA.

### **Treasury (Tesouro Americano)**
Títulos de dívida emitidos pelo governo dos EUA. Considerados os ativos mais seguros do mundo. O "10Y" refere-se ao vencimento de 10 anos.

### **DXY (US Dollar Index)**
Índice que mede a força do dólar americano contra uma cesta de moedas globais (euro, iene, libra, etc).

### **VIX (Volatility Index)**
Índice de volatilidade do S&P 500, também chamado de "índice do medo". Valores altos indicam maior incerteza no mercado.

### **HYG (High Yield Corporate Bond ETF)**
ETF de títulos corporativos de alto rendimento (high yield = "junk bonds"). Indicador de apetite por risco.

### **Futures**
Contratos padronizados para comprar/vender um ativo em data futura a preço determinado. Exemplo: ES (S&P 500 Futures).

---

## 2. Custos de Transação

### **bps (Basis Points / Pontos-Base)**
Unidade de medida para variações percentuais pequenas. **1 bp = 0.01% = 0.0001**.  
Exemplo: 10 bps = 0.10% = R$100 em R$100.000.

### **Spread (Bid-Ask Spread)**
Diferença entre o maior preço de compra (bid) e o menor preço de venda (ask) no livro de ofertas. Representa o custo de liquidez.  
**Fórmula:** Spread = Ask - Bid

### **Bid (Lance de Compra)**
Maior preço que um comprador está disposto a pagar por um ativo. É o preço que você **recebe** ao vender.

### **Ask (Lance de Venda)**
Menor preço que um vendedor está disposto a aceitar. É o preço que você **paga** ao comprar.

### **Corretagem (Brokerage Fee)**
Taxa cobrada pela corretora para executar ordens de compra/venda. Pode ser fixa (por ordem) ou variável (% do valor).

### **Slippage (Derrapagem)**
Diferença entre o preço esperado e o preço realmente executado. Causado por volatilidade, velocidade de execução e tamanho da ordem.

### **Transaction Costs (Custos de Transação)**
Soma de todos os custos para executar um trade: spread + corretagem + slippage + impostos.

### **Turnover**
Frequência de rebalanceamento do portfolio. Alto turnover = mais custos de transação.  
**Fórmula:** Turnover anual = Σ|mudanças de posição| / 2

### **Emolumentos**
Taxas cobradas pela bolsa de valores para processar operações. No Brasil (B3): ~3.25 bps.

---

## 3. Métricas de Performance

### **CAGR (Compound Annual Growth Rate)**
Taxa de crescimento anual composta. Retorno médio anualizado ao longo do período.  
**Fórmula:** CAGR = (Valor Final / Valor Inicial)^(1/anos) - 1

### **Sharpe Ratio**
Medida de retorno ajustado ao risco. Retorno excedente por unidade de volatilidade.  
**Fórmula:** (Retorno - Taxa Livre de Risco) / Volatilidade  
**Interpretação:** >1.0 = bom, >2.0 = excelente

### **Sortino Ratio**
Similar ao Sharpe, mas penaliza apenas volatilidade negativa (downside). Ignora volatilidade positiva.  
**Fórmula:** (Retorno - Taxa Livre de Risco) / Downside Deviation

### **Calmar Ratio**
Relação entre retorno anualizado e máximo drawdown.  
**Fórmula:** CAGR / |Max Drawdown|  
**Interpretação:** >1.0 = bom, >3.0 = excelente

### **Max Drawdown (MDD)**
Maior queda percentual do pico ao vale durante o período. Mede a pior perda acumulada.  
**Fórmula:** MDD = (Vale - Pico) / Pico

### **Volatilidade (Volatility)**
Medida de dispersão dos retornos. Tipicamente desvio padrão dos retornos anualizados.  
**Fórmula:** σ_anual = σ_diária × √252

### **Information Ratio**
Retorno ativo (vs benchmark) dividido pelo tracking error. Mede habilidade do gestor.  
**Fórmula:** (Retorno Portfolio - Retorno Benchmark) / Tracking Error

### **Alpha (α)**
Retorno excedente gerado além do esperado pelo risco sistemático (beta). Mede "skill" do gestor.

### **Beta (β)**
Sensibilidade do portfolio ao mercado. Beta = 1 significa movimento igual ao mercado.  
**Interpretação:** β > 1 = mais volátil que mercado, β < 1 = menos volátil

### **Win Rate (Taxa de Acerto)**
Percentual de trades lucrativos sobre o total de trades.  
**Fórmula:** Win Rate = (Trades Positivos / Total Trades) × 100

---

## 4. Conceitos de Portfolio

### **Long Position (Posição Comprada)**
Comprar um ativo esperando valorização. Lucro quando o preço sobe.  
**Exposição:** Positiva (+)

### **Short Position (Posição Vendida)**
Vender um ativo sem possuí-lo (emprestado), esperando desvalorização. Lucro quando o preço cai.  
**Exposição:** Negativa (-)

### **Long/Short Strategy**
Estratégia que combina posições compradas (long) e vendidas (short) simultaneamente. Pode ser market-neutral ou direcional.

### **Market Neutral**
Estratégia onde exposição líquida (long - short) é zero ou próxima de zero. Busca retorno descorrelacionado do mercado.

### **Exposure (Exposição)**
Percentual do capital alocado em uma posição ou classe de ativo.  
**Exposição Líquida:** Soma algébrica de long e short (pode ser positiva, negativa ou zero)

### **Leverage (Alavancagem)**
Uso de capital emprestado para aumentar exposição. Amplifica ganhos e perdas.  
**Fórmula:** Leverage = Exposição Total / Capital Próprio

### **Rebalancing (Rebalanceamento)**
Ajuste periódico das posições do portfolio para retornar aos pesos desejados.

### **Tactical Asset Allocation (TAA)**
Estratégia ativa que ajusta alocações baseada em previsões de curto/médio prazo de mercado.

### **Risk Parity (Paridade de Risco)**
Abordagem que aloca capital para equalizar a contribuição de risco de cada ativo. Também chamada ERC (Equal Risk Contribution).

### **60/40 Portfolio**
Portfolio clássico com 60% em ações e 40% em bonds. Benchmark tradicional para investidores moderados.

---

## 5. Regimes Macroeconômicos

### **Regime Macroeconômico**
Estado da economia caracterizado por combinação de inflação e crescimento. Usado para orientar alocações táticas.

### **Q1 - Goldilocks (Crescimento + Inflação Baixa)**
Cenário ideal: economia crescendo com inflação controlada. Favorece ações e ativos de risco.

### **Q2 - Reflação (Crescimento + Inflação Alta)**
Economia aquecendo com inflação acelerando. Favorece commodities e ativos reais.

### **Q3 - Estagflação (Recessão + Inflação Alta)**
Pior cenário: economia contraindo com inflação alta. Dificulta política monetária. Favorece ouro e liquidez.

### **Q4 - Deflação (Recessão + Inflação Baixa)**
Economia contraindo com inflação caindo. Favorece bonds governamentais (treasuries) e defensivos.

### **Intensidade do Sinal**
Força/confiança na classificação do regime. Dividida em: **Forte**, **Moderada** e **Fraca**.

### **Proxy de Inflação**
Combinação de ativos que representa expectativas de inflação: Oil, Gold, US10Y, DXY.

### **Proxy de Atividade Econômica**
Combinação de ativos que representa crescimento econômico: SP500, MSCI_EM, High Yield.

---

## 6. Risco e Volatilidade

### **Standard Deviation (Desvio Padrão)**
Medida estatística de dispersão dos retornos. Mais alta = maior variabilidade = maior risco.

### **Downside Deviation**
Desvio padrão calculado apenas com retornos negativos. Usado no Sortino Ratio.

### **Drawdown**
Queda percentual do patrimônio desde o último pico até o vale atual.  
**Drawdown Corrente:** (Equity Atual - Pico Anterior) / Pico Anterior

### **Rolling Volatility (Volatilidade Móvel)**
Volatilidade calculada em janelas deslizantes (ex: últimos 60 dias). Captura mudanças dinâmicas no risco.

### **Value at Risk (VaR)**
Perda máxima esperada com determinado nível de confiança (ex: 95%) em determinado período.

### **Stop-Loss**
Ordem automática para vender quando o preço atinge nível pré-determinado. Limita perdas.

### **ATR (Average True Range)**
Medida de volatilidade baseada na amplitude média de variação de preços. Usada para definir stop-loss dinâmicos.

### **Vol-Targeting (Volatilidade-Alvo)**
Técnica de ajustar exposição para manter volatilidade constante. Reduz exposição quando vol sobe, aumenta quando vol cai.

---

## 7. Backtesting

### **Backtest**
Simulação histórica de uma estratégia usando dados passados. Testa viabilidade antes de investir capital real.

### **In-Sample (IS)**
Período de dados usado para desenvolver/otimizar a estratégia. Também chamado "treino".

### **Out-of-Sample (OOS)**
Período de dados NÃO usado no desenvolvimento. Testa se a estratégia generaliza. Também chamado "teste".

### **Walk-Forward Analysis**
Método de validação que simula produção: treina em período passado, testa no período seguinte, avança e repete.

### **Overfitting**
Estratégia otimizada demais para dados históricos, capturando ruído ao invés de padrões reais. Falha no futuro.

### **Look-Ahead Bias (Viés de Antecipação)**
Erro de usar informação futura no momento da decisão. Invalida o backtest.  
**Correção:** Usar `.shift(1)` para garantir sinal disponível apenas após o fechamento.

### **Survivorship Bias (Viés de Sobrevivência)**
Usar apenas ativos que sobreviveram até hoje, ignorando os que faliram. Superestima retornos históricos.

### **Monte Carlo Simulation**
Técnica de simulação aleatória (1000+ cenários) para estimar distribuição de resultados e intervalos de confiança.

### **Equity Curve**
Gráfico da evolução do patrimônio ao longo do tempo. Visualiza performance acumulada.

---

## 8. Execução e Trading

### **Market Order (Ordem a Mercado)**
Ordem executada imediatamente ao melhor preço disponível. Garante execução, mas não o preço.

### **Limit Order (Ordem Limitada)**
Ordem que só executa a preço especificado ou melhor. Controla preço, mas não garante execução.

### **Fill (Execução)**
Confirmação de que uma ordem foi executada. "Filled at $500" = ordem executada a $500.

### **Liquidity (Liquidez)**
Facilidade de comprar/vender um ativo sem impactar significativamente o preço. Alta liquidez = spreads pequenos.

### **Market Impact (Impacto de Mercado)**
Movimento de preço causado pela própria ordem. Ordens grandes em ativos ilíquidos têm maior impacto.

### **Order Book (Livro de Ofertas)**
Lista de ordens de compra (bids) e venda (asks) aguardando execução, ordenadas por preço.

### **Volume**
Quantidade de ativos negociados em determinado período. Volume alto = maior liquidez.

### **DMA (Direct Market Access)**
Acesso direto ao mercado sem intermediários. Permite execução mais rápida e controle total.

### **HFT (High-Frequency Trading)**
Estratégias que executam milhares de trades por segundo usando algoritmos. Requer infraestrutura especializada.

---

## 9. Indicadores e Análise

### **Momentum**
Tendência de ativos que subiram (caíram) continuarem subindo (caindo). Base de estratégias de "trend following".

### **Trend Following**
Estratégia que busca capturar tendências de médio/longo prazo. Compra ativos em alta, vende em baixa.

### **Moving Average (Média Móvel)**
Média dos preços em janela deslizante. Suaviza ruído e identifica tendências.  
**Tipos:** SMA (simples), EMA (exponencial), WMA (ponderada)

### **MACD (Moving Average Convergence Divergence)**
Indicador de momentum baseado em diferença entre médias móveis rápida e lenta.

### **RSI (Relative Strength Index)**
Oscilador de momentum (0-100) que identifica sobrecompra (>70) ou sobrevenda (<30).

### **Bollinger Bands**
Bandas de volatilidade ao redor de média móvel. Preço tocando banda superior/inferior indica possível reversão.

### **OLS (Ordinary Least Squares)**
Método de regressão linear para estimar relação entre variáveis. Minimiza soma dos erros ao quadrado.

### **R² (R-Squared / Coeficiente de Determinação)**
Medida de ajuste da regressão (0-1). R²=0.5 significa que o modelo explica 50% da variação.

### **p-value (Valor-p)**
Probabilidade de observar resultado por acaso. p<0.05 indica significância estatística (95% de confiança).

### **PCA (Principal Component Analysis)**
Técnica de redução de dimensionalidade que extrai componentes principais (direções de maior variação) dos dados.

### **K-Means Clustering**
Algoritmo de agrupamento que particiona dados em K clusters baseado em similaridade (distância).

### **Silhouette Score**
Métrica de qualidade de clustering (-1 a +1). Valores altos indicam clusters bem separados.  
**Interpretação:** >0.5 = boa separação, <0.2 = clusters fracos

### **Rolling Window (Janela Deslizante)**
Análise em janelas de tempo fixas que se movem pelo histórico. Permite capturar dinâmicas temporais.

---

## 📊 Conversões Rápidas

### **Basis Points ↔ Percentual ↔ Decimal**

| bps | Percentual | Decimal | R$100k |
|-----|-----------|---------|---------|
| 1 | 0.01% | 0.0001 | R$ 10 |
| 5 | 0.05% | 0.0005 | R$ 50 |
| 10 | 0.10% | 0.0010 | R$ 100 |
| 50 | 0.50% | 0.0050 | R$ 500 |
| 100 | 1.00% | 0.0100 | R$ 1.000 |

### **Anualização de Métricas**

```python
# Retornos
retorno_anual = retorno_diario * 252  # 252 dias úteis

# Volatilidade
vol_anual = vol_diaria * np.sqrt(252)

# Sharpe Ratio
sharpe_anual = sharpe_diario * np.sqrt(252)
```

---

## 🔗 Recursos Adicionais

### **Livros Recomendados**
- **"Quantitative Trading"** - Ernest Chan
- **"Following the Trend"** - Andreas Clenow
- **"Advances in Financial Machine Learning"** - Marcos López de Prado

### **Fontes de Dados**
- **yfinance:** Dados históricos gratuitos (Yahoo Finance)
- **FRED:** Indicadores macroeconômicos (Federal Reserve)
- **Quandl/NASDAQ Data Link:** Dados alternativos

### **Frameworks e Bibliotecas**
- **QuantStats:** Análise de performance e relatórios
- **Backtrader:** Framework de backtesting
- **Zipline:** Backtesting (usado pelo Quantopian)

---

## 📝 Notas de Uso

**Convenções neste Projeto:**
- Retornos sempre em **base decimal** (0.01 = 1%)
- Custos sempre em **basis points** (10 bps = 0.10%)
- Frequência semanal: dados de **sexta-feira** (fechamento)
- Sinal calculado no **fim da semana N**, executado no **início da semana N+1**

**Siglas Comuns:**
- **B&H:** Buy and Hold (comprar e manter)
- **TAA:** Tactical Asset Allocation
- **PM:** Portfolio Management
- **RM:** Risk Management
- **PnL:** Profit and Loss (lucro/prejuízo)

---

**Última atualização:** 27/01/2026  
**Termos totais:** 100+  
**Categorias:** 9

Para dúvidas ou sugestões de novos termos, consulte a documentação do projeto ou os arquivos de análise.
