import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import numpy as np  # Importar numpy
import warnings

# Ignorar avisos
warnings.simplefilter(action='ignore', category=FutureWarning)

print("Iniciando o script de análise de tendência...")

# --- 1. Definição de Ativos e Coleta de Dados ---

tickers = {
    'SP500': '^GSPC',
    'DXY': 'DX-Y.NYB',
    'HighYield_ETF': 'HYG',
    'USD_BRL': 'BRL=X',
    'Oil_WTI': 'CL=F'
}

start_date = '2020-10-01'
end_date = '2025-10-29' 

try:
    data_prices = yf.download(list(tickers.values()), 
                              start=start_date, 
                              end=end_date,
                              auto_adjust=False)['Adj Close']
    
    data_prices.columns = list(tickers.keys())
    data_prices = data_prices.ffill()
    data_prices = data_prices.dropna()

    print(f"\nDados de preços baixados com sucesso de {start_date} a {end_date}.")
    print(data_prices.tail())

except Exception as e:
    print(f"Erro ao baixar dados: {e}")
    exit()

# --- 2. Preparação da Variável Independente (Tempo) ---

# Criar a variável X (Tempo) como um índice numérico (0, 1, 2, ...)
# Usamos np.arange(len(data_prices)) que cria um array [0, 1, 2, ..., N-1]
X_tempo = np.arange(len(data_prices))

# Adicionar a constante (intercepto, ou Beta_0)
X_com_constante = sm.add_constant(X_tempo)

# Renomear a coluna do índice de tempo para clareza no relatório
# A coluna 0 é a constante, a coluna 1 é nosso índice de tempo
X_com_constante = pd.DataFrame(X_com_constante, 
                               columns=['const', 'Time_Index'], 
                               index=data_prices.index)

print("\nVariável X (Tempo) preparada:")
print(X_com_constante.tail())


# --- 3. Execução das Regressões Lineares (Ativo vs. Tempo) ---

print("\n--- Iniciando Análises de Regressão (Ativo vs. Tempo) ---")

# Iterar sobre cada ativo (INCLUINDO o S&P 500 desta vez)
for asset_name in data_prices.columns:

    print(f"\n\n=======================================================")
    print(f" REGRESSÃO: {asset_name} (Y) vs. Time_Index (X)")
    print(f"=======================================================")

    # Definir nossa variável dependente (Y)
    Y = data_prices[asset_name]
    
    # Garantir que Y não tenha NaNs que possam ter sobrado
    Y = Y.dropna()
    X_temp = X_com_constante.loc[Y.index] # Alinha os índices

    # Criar e ajustar o modelo de regressão (Mínimos Quadrados Ordinários)
    model = sm.OLS(Y, X_temp)
    results = model.fit()

    # Imprimir o resumo completo da regressão
    print(results.summary())
    
    # Extrair e explicar os principais valores
    beta_0_preco_inicial = results.params['const']
    beta_1_tendencia = results.params['Time_Index']
    r_squared = results.rsquared

    print("\n--- Interpretação Resumida ---")
    print(f"  Preço Inicial Estimado (const): {beta_0_preco_inicial:.4f}")
    print(f"  Tendência Diária (Time_Index):  {beta_1_tendencia:.4f}")
    print(f"  R-quadrado:                     {r_squared*100:.2f}%")
    print("------------------------------")

print("\n\nScript concluído.")