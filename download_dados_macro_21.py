"""
Download de Dados Macroeconômicos (PMI e CPI) via FRED
=======================================================

Este script baixa dados reais de indicadores macroeconômicos:
- PMI Manufacturing (ISM): Indicador de atividade econômica
- CPI: Índice de Preços ao Consumidor (inflação)

Fonte: Federal Reserve Economic Data (FRED)
Data: Fevereiro 2026
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def baixar_pmi_cpi(start_date='2000-01-01', end_date=None):
    """
    Baixa dados de PMI e CPI do FRED via pandas_datareader
    
    Parâmetros:
    -----------
    start_date : str
        Data inicial (formato: 'YYYY-MM-DD')
    end_date : str
        Data final (se None, usa data atual)
    
    Retorna:
    --------
    df : DataFrame
        DataFrame com colunas: PMI, CPI, CPI_YoY
    """
    
    print("="*70)
    print("DOWNLOAD DE DADOS MACROECONÔMICOS")
    print("="*70)
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Tentar usar pandas_datareader
        import pandas_datareader as pdr
        
        print("\n✓ pandas_datareader importado com sucesso!")
        
        print("\n1. Baixando PMI Manufacturing (ISM)...")
        # PMI Manufacturing Index
        pmi = pdr.DataReader('MANEMP', 'fred', start=start_date, end=end_date)
        pmi.columns = ['PMI']
        print(f"   ✓ PMI baixado: {len(pmi)} observações")
        
        print("\n2. Baixando CPI (Consumer Price Index)...")
        # CPI for All Urban Consumers
        cpi = pdr.DataReader('CPIAUCSL', 'fred', start=start_date, end=end_date)
        cpi.columns = ['CPI']
        print(f"   ✓ CPI baixado: {len(cpi)} observações")
        
        # Calcular CPI Year-over-Year (%)
        cpi['CPI_YoY'] = cpi['CPI'].pct_change(12) * 100
        
        # Combinar PMI e CPI
        df = pmi.join(cpi, how='outer')
        
        # Converter para frequência semanal (sexta-feira) usando forward fill
        print("\n3. Convertendo para frequência semanal (W-FRI)...")
        df_semanal = df.resample('W-FRI').last().ffill()
        
        print(f"   ✓ Dados semanais: {len(df_semanal)} observações")
        print(f"   ✓ Período: {df_semanal.index[0].date()} até {df_semanal.index[-1].date()}")
        
        # Estatísticas descritivas
        print("\n" + "="*70)
        print("ESTATÍSTICAS DESCRITIVAS")
        print("="*70)
        print(df_semanal.describe())
        
        # Salvar arquivo
        output_file = 'dados_macro.csv'
        df_semanal.to_csv(output_file)
        print(f"\n✓ Arquivo salvo: {output_file}")
        
        return df_semanal
        
    except ImportError:
        print("\n⚠️  ERRO: pandas_datareader não está instalado!")
        print("\nPara instalar, execute:")
        print("   pip install pandas-datareader")
        print("\nAlternativamente, usando dados sintéticos para demonstração...")
        
        # Criar dados sintéticos para demonstração
        return criar_dados_sinteticos(start_date, end_date)
    
    except Exception as e:
        print(f"\n⚠️  ERRO ao baixar dados do FRED: {type(e).__name__}: {e}")
        print("\nPossíveis causas:")
        print("   1. Sem conexão com internet")
        print("   2. FRED API temporariamente indisponível")
        print("   3. Código do indicador mudou (MANEMP ou CPIAUCSL)")
        print("\nUsando dados sintéticos para demonstração...")
        
        # Criar dados sintéticos para demonstração
        return criar_dados_sinteticos(start_date, end_date)

def criar_dados_sinteticos(start_date, end_date):
    """
    Cria dados sintéticos de PMI e CPI para demonstração
    (Quando pandas_datareader não está disponível)
    """
    
    print("\n" + "="*70)
    print("CRIANDO DADOS SINTÉTICOS (DEMONSTRAÇÃO)")
    print("="*70)
    print("⚠️  ATENÇÃO: Estes dados são SIMULADOS, não reais!")
    print("   Para usar dados reais, instale: pip install pandas-datareader")
    print("="*70)
    
    # Criar datas semanais
    dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
    
    # PMI: Oscila entre 40-60, média 50
    np.random.seed(42)
    pmi_trend = 50 + 5 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    pmi_noise = np.random.normal(0, 3, len(dates))
    pmi = pmi_trend + pmi_noise
    pmi = np.clip(pmi, 35, 65)
    
    # CPI YoY: Oscila entre 0-6%, média 2.5%
    cpi_trend = 2.5 + 1.5 * np.sin(np.linspace(0, 3*np.pi, len(dates)))
    cpi_noise = np.random.normal(0, 0.5, len(dates))
    cpi_yoy = cpi_trend + cpi_noise
    cpi_yoy = np.clip(cpi_yoy, 0, 8)
    
    # Criar CPI sintético (nível, não só YoY)
    # Começar em 200 e crescer com inflação
    cpi_nivel = [200]
    for i in range(1, len(dates)):
        cpi_nivel.append(cpi_nivel[-1] * (1 + cpi_yoy[i-1]/100/52))  # Crescimento semanal
    
    # Criar DataFrame
    df = pd.DataFrame({
        'PMI': pmi,
        'CPI': cpi_nivel,  # Agora tem valores reais
        'CPI_YoY': cpi_yoy
    }, index=dates)
    
    print(f"\n✓ Dados sintéticos criados: {len(df)} observações")
    print(f"✓ Período: {df.index[0].date()} até {df.index[-1].date()}")
    
    # Estatísticas
    print("\n" + "="*70)
    print("ESTATÍSTICAS (DADOS SINTÉTICOS)")
    print("="*70)
    print(df.describe())
    
    # Salvar
    output_file = 'dados_macro_sintetico.csv'
    df.to_csv(output_file)
    print(f"\n✓ Arquivo salvo: {output_file}")
    
    return df

def baixar_precos_ativos(start_date='2000-01-01', end_date=None):
    """
    Baixa preços de SP500 e US Treasury 10Y para backtest
    """
    
    print("\n" + "="*70)
    print("DOWNLOAD DE PREÇOS DE ATIVOS")
    print("="*70)
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    tickers = {
        'SP500': '^GSPC',
        'US_10Y': '^TNX'
    }
    
    dados = {}
    for nome, ticker in tickers.items():
        print(f"\nBaixando {nome} ({ticker})...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Lidar com estrutura de dados do yfinance (pode variar)
        if isinstance(df.columns, pd.MultiIndex):
            # Múltiplos tickers (MultiIndex)
            dados[nome] = df['Adj Close'].iloc[:, 0] if 'Adj Close' in df.columns.levels[0] else df['Close'].iloc[:, 0]
        else:
            # Ticker único
            dados[nome] = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        
        print(f"   ✓ {nome}: {len(df)} observações")
    
    # Combinar
    df_precos = pd.DataFrame(dados)
    
    # Converter para semanal (sexta-feira)
    df_precos = df_precos.resample('W-FRI').last().ffill()
    
    print(f"\n✓ Dados semanais combinados: {len(df_precos)} observações")
    
    # Salvar
    output_file = 'precos_ativos.csv'
    df_precos.to_csv(output_file)
    print(f"✓ Arquivo salvo: {output_file}")
    
    return df_precos

def combinar_dados_completos():
    """
    Combina dados macro + preços em um único arquivo
    """
    
    print("\n" + "="*70)
    print("COMBINANDO DADOS MACRO + PREÇOS")
    print("="*70)
    
    # Baixar tudo
    df_macro = baixar_pmi_cpi()
    df_precos = baixar_precos_ativos()
    
    # Debug: verificar índices
    print("\nDEBUG:")
    print(f"Macro - Período: {df_macro.index[0]} até {df_macro.index[-1]}")
    print(f"Preços - Período: {df_precos.index[0]} até {df_precos.index[-1]}")
    
    # Combinar usando merge com índices
    df_completo = pd.merge(
        df_macro, 
        df_precos, 
        left_index=True, 
        right_index=True, 
        how='inner'
    )
    
    print(f"\n✓ Dados após merge: {len(df_completo)} observações")
    
    # Verificar se há dados
    if len(df_completo) == 0:
        print("\n⚠️  ERRO: Nenhum dado após combinar!")
        print("Possível causa: Índices das datas não estão batendo")
        print("\nUsando merge com tolerância de data...")
        
        # Tentar merge por data aproximada (tolerance)
        df_macro_reset = df_macro.reset_index()
        df_precos_reset = df_precos.reset_index()
        
        df_completo = pd.merge_asof(
            df_macro_reset.sort_values('index'),
            df_precos_reset.sort_values('index'),
            on='index',
            direction='nearest',
            tolerance=pd.Timedelta('7 days')
        )
        df_completo = df_completo.set_index('index')
        
        print(f"✓ Dados após merge_asof: {len(df_completo)} observações")
    
    # Remover NaNs apenas de colunas importantes
    df_original = len(df_completo)
    # Remover apenas linhas onde PMI, CPI_YoY, SP500 ou US_10Y são NaN
    df_completo = df_completo.dropna(subset=['PMI', 'CPI_YoY', 'SP500', 'US_10Y'])
    
    if len(df_completo) < df_original:
        print(f"⚠️  Removidos {df_original - len(df_completo)} registros com NaN em colunas essenciais")
    
    if len(df_completo) == 0:
        print("\n❌ ERRO FATAL: Nenhum dado válido após limpeza!")
        print("Verifique se os dados foram baixados corretamente.")
        return None
    
    print(f"\n✓ Dados completos finais: {len(df_completo)} observações")
    print(f"✓ Período: {df_completo.index[0].date()} até {df_completo.index[-1].date()}")
    
    # Salvar
    output_file = 'dados_completos_macro_precos.csv'
    df_completo.to_csv(output_file)
    print(f"✓ Arquivo salvo: {output_file}")
    
    print("\n" + "="*70)
    print("PREVIEW DOS DADOS")
    print("="*70)
    print(df_completo.head(10))
    print("\n...")
    print(df_completo.tail(10))
    
    return df_completo

if __name__ == "__main__":
    # Executar download completo
    df = combinar_dados_completos()
    
    if df is not None and len(df) > 0:
        print("\n" + "="*70)
        print("DOWNLOAD CONCLUÍDO!")
        print("="*70)
        print("\nArquivos criados:")
        print("   1. dados_macro.csv (ou dados_macro_sintetico.csv)")
        print("   2. precos_ativos.csv")
        print("   3. dados_completos_macro_precos.csv")
        print("\nPróximo passo:")
        print("   Execute: python estrategia_pmi_cpi.py")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("DOWNLOAD FALHOU!")
        print("="*70)
        print("\n⚠️  Não foi possível combinar os dados.")
        print("\nVerifique:")
        print("   1. pandas_datareader está instalado? pip install pandas-datareader")
        print("   2. Conexão com internet está funcionando?")
        print("   3. FRED API está acessível?")
        print("="*70)
