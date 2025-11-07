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

