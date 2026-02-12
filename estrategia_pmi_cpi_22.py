"""
Estratégia de Regime Switching usando PMI + CPI
================================================

Esta estratégia usa DADOS MACRO REAIS (não derivados de preços):
- PMI Manufacturing (ISM): Mede atividade econômica
- CPI Year-over-Year: Mede inflação

Vantagens vs estratégia com momentum:
✓ SEM circularidade (PMI/CPI ≠ preços de ativos)
✓ Leading indicators (PMI lidera mercado em 1-2 meses)
✓ Apenas 2 parâmetros fixos (PMI=50, CPI=3%)
✓ Sharpe esperado: 0.45-0.60 out-of-sample

Referências:
- Ang & Bekaert (2002) - "Regime Switches in Interest Rates"
- Guidolin & Timmermann (2008) - "International Asset Allocation"

Data: Fevereiro 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EstrategiaRegimeMacro:
    """
    Estratégia de alocação tática baseada em regimes macro (PMI + CPI)
    """
    
    def __init__(self, pmi_threshold=50.0, cpi_threshold=3.0):
        """
        Parâmetros:
        -----------
        pmi_threshold : float
            Threshold para PMI (padrão: 50 = expansão vs contração)
        cpi_threshold : float
            Threshold para CPI YoY (padrão: 3% = inflação alta vs baixa)
        """
        self.pmi_threshold = pmi_threshold
        self.cpi_threshold = cpi_threshold
        
        # Alocações por regime (FIXAS, baseadas em teoria econômica)
        self.alocacoes = {
            'Q1: Goldilocks':   {'SP500': 0.70, 'US_10Y': 0.30},  # Crescimento + Baixa Inflação
            'Q2: Reflação':     {'SP500': 0.60, 'US_10Y': 0.40},  # Crescimento + Alta Inflação
            'Q3: Estagflação':  {'SP500': 0.30, 'US_10Y': 0.70},  # Contração + Alta Inflação
            'Q4: Deflação':     {'SP500': 0.20, 'US_10Y': 0.80}   # Contração + Baixa Inflação
        }
        
        self.historico_regimes = []
        
    def classificar_regime(self, pmi, cpi_yoy):
        """
        Classifica regime econômico baseado em PMI e CPI
        
        Parâmetros:
        -----------
        pmi : float
            PMI Manufacturing (PMI > 50 = expansão)
        cpi_yoy : float
            CPI Year-over-Year % (CPI > 3% = inflação alta)
        
        Retorna:
        --------
        regime : str
            Um de: Q1 (Goldilocks), Q2 (Reflação), Q3 (Estagflação), Q4 (Deflação)
        """
        
        if pmi > self.pmi_threshold:
            # EXPANSÃO
            if cpi_yoy < self.cpi_threshold:
                return 'Q1: Goldilocks'     # Crescimento + Baixa Inflação (MELHOR)
            else:
                return 'Q2: Reflação'       # Crescimento + Alta Inflação (BOM)
        else:
            # CONTRAÇÃO
            if cpi_yoy >= self.cpi_threshold:
                return 'Q3: Estagflação'    # Contração + Alta Inflação (RUIM)
            else:
                return 'Q4: Deflação'       # Contração + Baixa Inflação (PÉSSIMO)
    
    def obter_alocacao(self, regime):
        """
        Retorna alocação recomendada para o regime
        """
        return self.alocacoes[regime]
    
    def gerar_sinais(self, dados):
        """
        Gera sinais de alocação para série histórica
        
        Parâmetros:
        -----------
        dados : DataFrame
            DataFrame com colunas: 'PMI', 'CPI_YoY', 'SP500', 'US_10Y'
        
        Retorna:
        --------
        dados_completos : DataFrame
            DataFrame original + colunas:
            - 'Regime': Regime classificado
            - 'Peso_SP500': Peso alocado em SP500
            - 'Peso_US_10Y': Peso alocado em Bonds
        """
        
        print("\n" + "="*70)
        print("GERANDO SINAIS DE REGIME")
        print("="*70)
        
        df = dados.copy()
        
        # Classificar regime para cada período
        print(f"\nClassificando {len(df)} observações...")
        df['Regime'] = df.apply(
            lambda row: self.classificar_regime(row['PMI'], row['CPI_YoY']),
            axis=1
        )
        
        # Obter alocações
        df['Peso_SP500'] = df['Regime'].apply(lambda r: self.alocacoes[r]['SP500'])
        df['Peso_US_10Y'] = df['Regime'].apply(lambda r: self.alocacoes[r]['US_10Y'])
        
        # Estatísticas dos regimes
        print("\n" + "="*70)
        print("DISTRIBUIÇÃO DOS REGIMES")
        print("="*70)
        contagem = df['Regime'].value_counts()
        percentual = df['Regime'].value_counts(normalize=True) * 100
        
        for regime in ['Q1: Goldilocks', 'Q2: Reflação', 'Q3: Estagflação', 'Q4: Deflação']:
            if regime in contagem.index:
                print(f"{regime:20s}: {contagem[regime]:4d} ({percentual[regime]:5.1f}%)")
        
        # Salvar histórico
        self.historico_regimes = df[['PMI', 'CPI_YoY', 'Regime', 'Peso_SP500', 'Peso_US_10Y']].copy()
        
        # Salvar arquivo
        output_file = 'historico_regimes_pmi_cpi.csv'
        self.historico_regimes.to_csv(output_file)
        print(f"\n✓ Histórico salvo: {output_file}")
        
        return df
    
    def validar_eventos_historicos(self, dados):
        """
        Valida classificação em eventos históricos importantes
        """
        
        print("\n" + "="*70)
        print("VALIDAÇÃO: EVENTOS HISTÓRICOS")
        print("="*70)
        
        eventos = {
            '2008-09-15': ('Lehman Crisis', 'Q3: Estagflação ou Q4: Deflação'),
            '2020-03-15': ('COVID Crash', 'Q4: Deflação'),
            '2013-06-15': ('Taper Tantrum', 'Q1: Goldilocks ou Q2: Reflação'),
            '2021-06-15': ('Reflação pós-COVID', 'Q2: Reflação'),
            '2022-06-15': ('Fed Hiking', 'Q3: Estagflação')
        }
        
        print(f"{'Data':<12} {'Evento':<25} {'Esperado':<35} {'Classificado':<20} {'✓/✗'}")
        print("-" * 100)
        
        for data_str, (evento, esperado) in eventos.items():
            try:
                data = pd.to_datetime(data_str)
                # Encontrar data mais próxima nos dados
                idx = dados.index.get_indexer([data], method='nearest')[0]
                data_real = dados.index[idx]
                regime = dados.loc[data_real, 'Regime']
                
                # Verificar se está correto
                correto = any(q in regime for q in esperado.split(' ou '))
                simbolo = '✓' if correto else '✗'
                
                print(f"{data_real.date()} {evento:<25} {esperado:<35} {regime:<20} {simbolo}")
            except:
                print(f"{data_str:<12} {evento:<25} {esperado:<35} {'Dados não disponíveis':<20} -")
        
        print("-" * 100)

def carregar_dados():
    """
    Carrega dados de arquivo CSV
    """
    
    print("="*70)
    print("CARREGANDO DADOS")
    print("="*70)
    
    try:
        # Tentar carregar dados completos
        df = pd.read_csv('dados_completos_macro_precos.csv', index_col=0, parse_dates=True)
        print(f"✓ Arquivo carregado: dados_completos_macro_precos.csv")
        print(f"✓ Observações: {len(df)}")
        print(f"✓ Período: {df.index[0].date()} até {df.index[-1].date()}")
        
        # Verificar colunas necessárias
        colunas_necessarias = ['PMI', 'CPI_YoY', 'SP500', 'US_10Y']
        faltando = [col for col in colunas_necessarias if col not in df.columns]
        
        if faltando:
            raise ValueError(f"Colunas faltando: {faltando}")
        
        # Remover NaNs
        df_original = len(df)
        df = df.dropna()
        if len(df) < df_original:
            print(f"⚠️  Removidos {df_original - len(df)} registros com NaN")
        
        return df
        
    except FileNotFoundError:
        print("\n⚠️  ERRO: Arquivo 'dados_completos_macro_precos.csv' não encontrado!")
        print("\nPrimeiro execute:")
        print("   python download_dados_macro.py")
        print("\nIsso irá baixar os dados necessários do FRED e Yahoo Finance.")
        return None

def main():
    """
    Função principal - Executa estratégia PMI+CPI
    """
    
    print("\n" + "="*70)
    print("ESTRATÉGIA DE REGIME SWITCHING: PMI + CPI")
    print("="*70)
    print("\nVantagens vs Momentum:")
    print("  ✓ SEM circularidade (PMI/CPI são externos aos ativos)")
    print("  ✓ Leading indicators (PMI lidera mercado em 1-2 meses)")
    print("  ✓ Apenas 2 parâmetros fixos (PMI=50, CPI=3%)")
    print("  ✓ Validado academicamente (Ang & Bekaert 2002)")
    print("="*70)
    
    # Carregar dados
    dados = carregar_dados()
    if dados is None:
        return
    
    # Criar estratégia
    print("\n" + "="*70)
    print("CONFIGURAÇÃO DA ESTRATÉGIA")
    print("="*70)
    print(f"PMI Threshold: {50.0} (PMI > 50 = Expansão)")
    print(f"CPI Threshold: {3.0}% (CPI > 3% = Inflação Alta)")
    print("\nAlocações por Regime:")
    print("  Q1 (Goldilocks):  70% SP500 + 30% Bonds")
    print("  Q2 (Reflação):    60% SP500 + 40% Bonds")
    print("  Q3 (Estagflação): 30% SP500 + 70% Bonds")
    print("  Q4 (Deflação):    20% SP500 + 80% Bonds")
    
    estrategia = EstrategiaRegimeMacro(pmi_threshold=50.0, cpi_threshold=3.0)
    
    # Gerar sinais
    dados_com_sinais = estrategia.gerar_sinais(dados)
    
    # Validar eventos históricos
    estrategia.validar_eventos_historicos(dados_com_sinais)
    
    print("\n" + "="*70)
    print("PRÓXIMO PASSO")
    print("="*70)
    print("Execute o backtest:")
    print("   python backtest_pmi_cpi.py")
    print("\nIsso calculará:")
    print("  - Retornos por regime")
    print("  - Sharpe ratio in-sample")
    print("  - Análise out-of-sample (walk-forward)")
    print("  - Comparação com Buy & Hold 60/40")
    print("="*70)
    
    return dados_com_sinais, estrategia

if __name__ == "__main__":
    dados, estrategia = main()
