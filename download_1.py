import yfinance as yf
import pandas as pd
import numpy as np
import warnings

#Funções

#Função validar_dados
def validar_dados(df):
    """Valida qualidade dos dados."""
    # 1. Detectar gaps
    gaps = df.isna().sum()
    if gaps.any():
        print(f"⚠️ Gaps detectados: {gaps[gaps > 0]}")
        df = df.ffill().bfill()  # Forward-fill + backward-fill
    
    # 2. Detectar outliers (>5 sigma)
    zscore = (df - df.mean()) / df.std()
    outliers = (zscore.abs() > 5).sum()
    if outliers.any():
        print(f"⚠️ Outliers detectados: {outliers[outliers > 0]}")
        # Winsorize extremos
        df = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
    
    # 3. Verificar correlações anômalas
    corr = df.pct_change().corr()
    if (corr > 0.95).sum().sum() > len(df.columns):
        print("⚠️ Correlações suspeitas (>0.95) detectadas")
    
    return df

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

tickers= {
    # AÇÕES - Crescimento
    'SP500': '^GSPC', 'MSCI_EM': 'EEM', 'Russell_2000': '^RUT',  # Small caps
    'MSCI_EAFE': 'EFA',  # Desenvolvidos ex-US
    
    # BONDS - Taxas/Expectativas
    'US_10Y': '^TNX', 'US_2Y': '^IRX',  # Curva de juros
    'TIP': 'TIP',  # TIPS (breakeven inflation)
    'HighYield_ETF': 'HYG', 'BBB_Corp': 'LQD',  # Crédito
    
    # MOEDAS - Condições Monetárias
    'DXY': 'DX-Y.NYB', 'EUR_USD': 'EURUSD=X', 'JPY_USD': 'JPY=X',
    
    # COMMODITIES - Inflação
    'Oil_WTI': 'CL=F', 'Gold': 'GC=F', 'Copper': 'HG=F',  # Metais industriais
    'CRB_Index': 'DBC',  # Basket de commodities
    
    # VOLATILIDADE - Risco
    'VIX': '^VIX',  # Medo do mercado
    'MOVE': '^MOVE',  # Vol de bonds
}



start_date = '2000-01-01'  # +16 anos = 2.5 ciclos completos
end_date = pd.Timestamp.now().normalize()

print(f"\nBaixando dados de {start_date} a {end_date}...\n")

try:
    data_prices = yf.download(list(tickers.values()), 
                              start=start_date, 
                              end=end_date,
                              threads=False,
                              progress=False,
                              auto_adjust=False)['Adj Close']
    data_prices_weekly = data_prices.resample('W-FRI').last()  # Sexta-feira
    data_prices_weekly.columns = list(tickers.keys())
    data_prices_weekly = validar_dados(data_prices_weekly)

    #Salvar dados em csv
    data_prices_weekly.to_csv('data_prices.csv')
    print(f"\n✓ Dados salvos em 'data_prices.csv'")
    print(f"✓ Período: {start_date} a {end_date}")
    print(f"✓ Total de dias: {len(data_prices_weekly) * 7}")
    print("\nPrimeiras linhas:")
    print(data_prices_weekly.head())
    print("\nÚltimas linhas:")
    print(data_prices_weekly.tail())

  
except Exception as e:
    print(f"Erro ao baixar dados: {e}")
    exit()

