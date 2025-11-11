import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import numpy as np  # Importar numpy
import warnings
import time  # Adicionar import

# Ignorar avisos
warnings.simplefilter(action='ignore', category=FutureWarning)

print("Iniciando o script de análise de tendência...")

# --- 1. Definição de Ativos e Coleta de Dados ---

# tickers = {
#     'SP500': '^GSPC',
#     'DXY': 'DX-Y.NYB',
#     'HighYield_ETF': 'HYG',
#     'USD_BRL': 'BRL=X',
#     'Oil_WTI': 'CL=F'
# }

tickers = {
    # AÇÕES (Crescimento Econômico)
    'SP500': '^GSPC',           # Ações desenvolvidas - USA
    'MSCI_EM': 'EEM',           # Ações emergentes - Mundo
    
    # MOEDAS (Condições Monetárias)
    'DXY': 'DX-Y.NYB',          # Índice do Dólar
    
    # BONDS (Taxas de Juros / Expectativas)
    'US_10Y': '^TNX',           # Treasury 10 anos
    'HighYield_ETF': 'HYG',     # High Yield Corporate Bonds
    
    # COMMODITIES (Inflação)
    'Oil_WTI': 'CL=F',          # Petróleo
    'Gold': 'GC=F'              # Ouro
}

start_date = '2016-01-01'
end_date = pd.Timestamp.now().normalize()

print(f"\nBaixando dados de {start_date} a {end_date}...\n")

try:
    data_prices = yf.download(list(tickers.values()), 
                              start=start_date, 
                              end=end_date,
                              threads=False,
                              progress=False,
                              auto_adjust=False)['Adj Close']
    
    data_prices.columns = list(tickers.keys())
    data_prices = data_prices.ffill()
    data_prices = data_prices.dropna()

    #Salvar dados em csv
    data_prices.to_csv('data_prices.csv')
    print(f"\n✓ Dados salvos em 'data_prices.csv'")
    print(f"✓ Período: {start_date} a {end_date}")
    print(f"✓ Total de dias: {len(data_prices)}")
    print("\nPrimeiras linhas:")
    print(data_prices.head())
    print("\nÚltimas linhas:")
    print(data_prices.tail())

  
except Exception as e:
    print(f"Erro ao baixar dados: {e}")
    exit()

