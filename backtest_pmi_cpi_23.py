"""
Backtest da Estratégia PMI + CPI
=================================

Testa estratégia de regime switching usando PMI e CPI.
Inclui:
- Backtest in-sample (período completo)
- Walk-forward out-of-sample (2020-2025)
- Comparação com benchmarks (Buy&Hold 60/40, Momentum)
- Análise de custos de transação

Data: Fevereiro 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BacktestRegimeMacro:
    """
    Engine de backtest para estratégia PMI+CPI
    """
    
    def __init__(self, custo_transacao=0.0005):
        """
        Parâmetros:
        -----------
        custo_transacao : float
            Custo por trade em decimal (0.0005 = 5 bps)
        """
        self.custo = custo_transacao
        self.resultados = {}
        
    def calcular_retornos(self, dados):
        """
        Calcula retornos semanais dos ativos
        """
        df = dados.copy()
        
        # Retornos dos ativos
        df['Ret_SP500'] = df['SP500'].pct_change()
        df['Ret_US_10Y'] = df['US_10Y'].pct_change()
        
        return df
    
    def calcular_retorno_estrategia(self, dados):
        """
        Calcula retorno da estratégia período a período
        """
        df = dados.copy()
        
        # Retorno da carteira (weighted average dos ativos)
        df['Ret_Portfolio'] = (
            df['Peso_SP500'] * df['Ret_SP500'] + 
            df['Peso_US_10Y'] * df['Ret_US_10Y']
        )
        
        # Calcular custos de transação
        df['Mudanca_SP500'] = df['Peso_SP500'].diff().abs()
        df['Mudanca_US_10Y'] = df['Peso_US_10Y'].diff().abs()
        df['Custo'] = (df['Mudanca_SP500'] + df['Mudanca_US_10Y']) * self.custo
        
        # Retorno líquido (depois de custos)
        df['Ret_Liquido'] = df['Ret_Portfolio'] - df['Custo']
        
        # Retorno cumulativo
        df['Cumul_Strategy'] = (1 + df['Ret_Liquido']).cumprod()
        
        # Benchmarks
        df['Ret_6040'] = 0.60 * df['Ret_SP500'] + 0.40 * df['Ret_US_10Y']
        df['Cumul_6040'] = (1 + df['Ret_6040']).cumprod()
        df['Cumul_SP500'] = (1 + df['Ret_SP500']).cumprod()
        df['Cumul_US_10Y'] = (1 + df['Ret_US_10Y']).cumprod()
        
        return df
    
    def calcular_metricas(self, retornos, nome="Estratégia"):
        """
        Calcula métricas de performance
        """
        # Remover NaN
        ret = retornos.dropna()
        
        if len(ret) == 0:
            return {
                'Retorno Anualizado': 0.0,
                'Volatilidade Anualizada': 0.0,
                'Sharpe Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Calmar Ratio': 0.0
            }
        
        # Retorno anualizado (52 semanas)
        ret_total = (1 + ret).prod() - 1
        n_anos = len(ret) / 52.0
        ret_anual = (1 + ret_total) ** (1/n_anos) - 1
        
        # Volatilidade anualizada
        vol_anual = ret.std() * np.sqrt(52)
        
        # Sharpe ratio (assumindo Rf = 0 para simplificar)
        sharpe = ret_anual / vol_anual if vol_anual > 0 else 0.0
        
        # Drawdown
        cumul = (1 + ret).cumprod()
        running_max = cumul.expanding().max()
        drawdown = (cumul - running_max) / running_max
        max_dd = drawdown.min()
        
        # Calmar ratio
        calmar = ret_anual / abs(max_dd) if max_dd < 0 else 0.0
        
        return {
            'Retorno Anualizado': ret_anual,
            'Volatilidade Anualizada': vol_anual,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd,
            'Calmar Ratio': calmar
        }
    
    def executar_backtest(self, dados_com_sinais):
        """
        Executa backtest completo
        """
        
        print("\n" + "="*70)
        print("EXECUTANDO BACKTEST")
        print("="*70)
        
        # Calcular retornos
        df = self.calcular_retornos(dados_com_sinais)
        df = self.calcular_retorno_estrategia(df)
        
        # Remover NaN (primeiro período)
        df = df.dropna(subset=['Ret_Liquido'])
        
        print(f"\nPeríodo: {df.index[0].date()} até {df.index[-1].date()}")
        print(f"Observações: {len(df)}")
        
        # Estatísticas de trades
        trades = df['Mudanca_SP500'].fillna(0)
        n_trades = (trades > 0.01).sum()  # Mudanças > 1%
        custo_total = df['Custo'].sum()
        
        print(f"\nTrades executados: {n_trades}")
        print(f"Custo total: {custo_total*100:.2f}%")
        print(f"Custo médio por ano: {custo_total/len(df)*52*100:.2f}%")
        
        # Métricas
        print("\n" + "="*70)
        print("MÉTRICAS DE PERFORMANCE")
        print("="*70)
        
        metricas_strategy = self.calcular_metricas(df['Ret_Liquido'], "Estratégia PMI+CPI")
        metricas_6040 = self.calcular_metricas(df['Ret_6040'], "Buy & Hold 60/40")
        metricas_sp500 = self.calcular_metricas(df['Ret_SP500'], "SP500")
        metricas_us10y = self.calcular_metricas(df['Ret_US_10Y'], "US 10Y")
        
        # Tabela comparativa expandida
        print(f"\n{'Métrica':<25} {'PMI+CPI':>15} {'60/40':>15} {'SP500':>15} {'US10Y':>15}")
        print("-" * 85)
        
        for metrica in metricas_strategy.keys():
            val_strat = metricas_strategy[metrica]
            val_6040 = metricas_6040[metrica]
            val_sp500 = metricas_sp500[metrica]
            val_us10y = metricas_us10y[metrica]
            
            if 'Ratio' in metrica:
                print(f"{metrica:<25} {val_strat:>15.2f} {val_6040:>15.2f} {val_sp500:>15.2f} {val_us10y:>15.2f}")
            else:
                print(f"{metrica:<25} {val_strat*100:>14.1f}% {val_6040*100:>14.1f}% {val_sp500*100:>14.1f}% {val_us10y*100:>14.1f}%")
        
        # Distribuição dos regimes
        print("\n" + "="*85)
        print("DISTRIBUIÇÃO DOS REGIMES")
        print("="*85)
        contagem_regimes = df['Regime'].value_counts()
        percentual_regimes = df['Regime'].value_counts(normalize=True) * 100
        
        for regime in ['Q1: Goldilocks', 'Q2: Reflação', 'Q3: Estagflação', 'Q4: Deflação']:
            if regime in contagem_regimes.index:
                print(f"{regime:<20}: {contagem_regimes[regime]:>4} períodos ({percentual_regimes[regime]:>5.1f}%)")
        
        # Retornos por regime
        print("\n" + "="*85)
        print("RETORNOS POR REGIME")
        print("="*85)
        
        for regime in ['Q1: Goldilocks', 'Q2: Reflação', 'Q3: Estagflação', 'Q4: Deflação']:
            mask = df['Regime'] == regime
            if mask.sum() > 0:
                ret_regime = df.loc[mask, 'Ret_Liquido']
                ret_medio = ret_regime.mean() * 52 * 100  # Anualizado
                n_obs = mask.sum()
                
                print(f"{regime:<20}: {ret_medio:>6.1f}% anual ({n_obs:>4} períodos)")
        
        # Salvar resultados
        self.resultados = {
            'dados': df,
            'metricas_strategy': metricas_strategy,
            'metricas_6040': metricas_6040,
            'metricas_sp500': metricas_sp500,
            'metricas_us10y': metricas_us10y,
            'contagem_regimes': contagem_regimes.to_dict(),
            'percentual_regimes': percentual_regimes.to_dict()
        }
        
        # Salvar arquivo
        output_file = 'backtest_pmi_cpi_detalhado.csv'
        df.to_csv(output_file)
        print(f"\n✓ Backtest detalhado salvo: {output_file}")
        
        return df
    
    def walk_forward_analysis(self, dados_com_sinais, train_years=15, test_years=5):
        """
        Walk-forward out-of-sample analysis
        """
        
        print("\n" + "="*70)
        print("WALK-FORWARD OUT-OF-SAMPLE ANALYSIS")
        print("="*70)
        print(f"\nTreino: Primeiros {train_years} anos")
        print(f"Teste: Últimos {test_years} anos (2020-2025)")
        
        df = dados_com_sinais.copy()
        
        # Dividir em treino e teste
        split_date = df.index[-52*test_years]  # Últimos 5 anos
        
        df_train = df[df.index < split_date]
        df_test = df[df.index >= split_date]
        
        print(f"\nTreino: {df_train.index[0].date()} até {df_train.index[-1].date()} ({len(df_train)} obs)")
        print(f"Teste:  {df_test.index[0].date()} até {df_test.index[-1].date()} ({len(df_test)} obs)")
        
        # Backtest em cada período
        print("\n" + "-"*70)
        print("RESULTADOS IN-SAMPLE (TREINO)")
        print("-"*70)
        df_train_bt = self.calcular_retornos(df_train)
        df_train_bt = self.calcular_retorno_estrategia(df_train_bt)
        metricas_train = self.calcular_metricas(df_train_bt['Ret_Liquido'].dropna())
        print(f"Sharpe Ratio: {metricas_train['Sharpe Ratio']:.2f}")
        print(f"Retorno Anual: {metricas_train['Retorno Anualizado']*100:.1f}%")
        
        print("\n" + "-"*70)
        print("RESULTADOS OUT-OF-SAMPLE (TESTE)")
        print("-"*70)
        df_test_bt = self.calcular_retornos(df_test)
        df_test_bt = self.calcular_retorno_estrategia(df_test_bt)
        metricas_test = self.calcular_metricas(df_test_bt['Ret_Liquido'].dropna())
        print(f"Sharpe Ratio: {metricas_test['Sharpe Ratio']:.2f}")
        print(f"Retorno Anual: {metricas_test['Retorno Anualizado']*100:.1f}%")
        
        # Comparação
        print("\n" + "-"*70)
        print("DIAGNÓSTICO DE OVERFITTING")
        print("-"*70)
        sharpe_diff = metricas_train['Sharpe Ratio'] - metricas_test['Sharpe Ratio']
        print(f"Diferença de Sharpe: {sharpe_diff:.2f}")
        
        if abs(sharpe_diff) < 0.10:
            print("✓ Modelo ROBUSTO (diferença < 0.10)")
        elif abs(sharpe_diff) < 0.20:
            print("⚠️  Modelo RAZOÁVEL (diferença 0.10-0.20)")
        else:
            print("❌ Modelo com OVERFITTING (diferença > 0.20)")
        
        return metricas_train, metricas_test
    
    def plotar_resultados(self, dados):
        """
        Plota gráficos de performance com métricas expandidas
        """
        
        print("\n" + "="*70)
        print("GERANDO GRÁFICOS")
        print("="*70)
        
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
        
        metricas = self.resultados
        
        # ========== GRÁFICO 1: Retorno Cumulativo (topo, 2 colunas) ==========
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(dados.index, dados['Cumul_Strategy'], label='Estratégia PMI+CPI', 
                linewidth=2.5, color='darkblue')
        ax1.plot(dados.index, dados['Cumul_6040'], label='Buy & Hold 60/40', 
                linewidth=2, linestyle='--', color='green')
        ax1.plot(dados.index, dados['Cumul_SP500'], label='SP500', 
                linewidth=1.5, linestyle=':', color='orange', alpha=0.7)
        ax1.plot(dados.index, dados['Cumul_US_10Y'], label='US 10Y', 
                linewidth=1.5, linestyle='-.', color='red', alpha=0.7)
        
        # Adicionar texto com Sharpe ratios
        sharpe_text = (
            f"Sharpe Ratios:\n"
            f"PMI+CPI: {metricas['metricas_strategy']['Sharpe Ratio']:.2f}  |  "
            f"60/40: {metricas['metricas_6040']['Sharpe Ratio']:.2f}  |  "
            f"SP500: {metricas['metricas_sp500']['Sharpe Ratio']:.2f}  |  "
            f"US10Y: {metricas['metricas_us10y']['Sharpe Ratio']:.2f}"
        )
        ax1.text(0.02, 0.98, sharpe_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        ax1.set_title('Retorno Cumulativo - Comparação de Estratégias', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Valor ($)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # ========== GRÁFICO 2: Drawdown da Estratégia (2a linha, 2 colunas) ==========
        ax2 = fig.add_subplot(gs[1, :])
        cumul_strategy = dados['Cumul_Strategy']
        running_max = cumul_strategy.expanding().max()
        drawdown = (cumul_strategy - running_max) / running_max * 100
        
        ax2.fill_between(dados.index, 0, drawdown, alpha=0.6, color='red', label='Drawdown')
        ax2.plot(dados.index, drawdown, color='darkred', linewidth=1.5)
        
        max_dd = metricas['metricas_strategy']['Max Drawdown'] * 100
        ax2.axhline(y=max_dd, color='darkred', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'Max DD: {max_dd:.1f}%')
        
        ax2.set_title('Drawdown da Estratégia PMI+CPI', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.legend(loc='lower left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([min(max_dd * 1.2, -50), 5])
        
        # ========== GRÁFICO 3: Alocação Dinâmica (3a linha, 2 colunas) ==========
        ax3 = fig.add_subplot(gs[2, :])
        ax3.fill_between(dados.index, 0, dados['Peso_SP500']*100, 
                         alpha=0.7, label='SP500', color='blue')
        ax3.fill_between(dados.index, dados['Peso_SP500']*100, 100,
                         alpha=0.7, label='US 10Y', color='green')
        ax3.set_title('Alocação Dinâmica', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Peso (%)', fontsize=12)
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 100])
        
        # ========== GRÁFICO 4: PMI e CPI (inferior esquerda) ==========
        ax4 = fig.add_subplot(gs[3, 0])
        ax4_twin = ax4.twinx()
        
        ax4.plot(dados.index, dados['PMI'], label='PMI', color='blue', linewidth=1.5)
        ax4.axhline(y=50, color='blue', linestyle='--', alpha=0.5, label='PMI=50')
        ax4.set_ylabel('PMI', fontsize=11, color='blue')
        ax4.tick_params(axis='y', labelcolor='blue')
        
        ax4_twin.plot(dados.index, dados['CPI_YoY'], label='CPI YoY', color='red', linewidth=1.5)
        ax4_twin.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='CPI=3%')
        ax4_twin.set_ylabel('CPI YoY (%)', fontsize=11, color='red')
        ax4_twin.tick_params(axis='y', labelcolor='red')
        
        ax4.set_title('Indicadores Macroeconômicos', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Data', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
        
        # ========== GRÁFICO 5: Distribuição dos Regimes (inferior direita) ==========
        ax5 = fig.add_subplot(gs[3, 1])
        contagem = metricas['contagem_regimes']
        percentual = metricas['percentual_regimes']
        
        regimes_ordem = ['Q1: Goldilocks', 'Q2: Reflação', 'Q3: Estagflação', 'Q4: Deflação']
        cores = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
        
        valores = [contagem.get(r, 0) for r in regimes_ordem]
        labels_pie = [f"{r.split(':')[0]}\n{percentual.get(r, 0):.1f}%" for r in regimes_ordem]
        
        wedges, texts, autotexts = ax5.pie(valores, labels=labels_pie, colors=cores, 
                                             autopct='%d', startangle=90, 
                                             textprops={'fontsize': 9})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax5.set_title('Distribuição dos Regimes', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Salvar gráfico
        output_file = 'backtest_pmi_cpi_graficos_completo.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Gráficos salvos: {output_file}")
        
        plt.show()

def main():
    """
    Função principal - Executa backtest completo
    """
    
    print("\n" + "="*70)
    print("BACKTEST: ESTRATÉGIA PMI + CPI")
    print("="*70)
    
    # Carregar dados com sinais
    try:
        dados = pd.read_csv('historico_regimes_pmi_cpi.csv', index_col=0, parse_dates=True)
        print(f"✓ Histórico de regimes carregado")
        
        # Carregar preços
        dados_completos = pd.read_csv('dados_completos_macro_precos.csv', index_col=0, parse_dates=True)
        
        # Combinar
        dados_backtest = dados.join(dados_completos[['SP500', 'US_10Y']], how='inner')
        
        print(f"✓ Dados completos: {len(dados_backtest)} observações")
        
    except FileNotFoundError:
        print("\n⚠️  ERRO: Arquivos necessários não encontrados!")
        print("\nExecute primeiro:")
        print("   1. python download_dados_macro.py")
        print("   2. python estrategia_pmi_cpi.py")
        return
    
    # Criar engine de backtest
    backtest = BacktestRegimeMacro(custo_transacao=0.0005)
    
    # Executar backtest in-sample
    dados_resultados = backtest.executar_backtest(dados_backtest)
    
    # Walk-forward out-of-sample
    metricas_train, metricas_test = backtest.walk_forward_analysis(dados_backtest, train_years=15, test_years=5)
    
    # Plotar gráficos
    backtest.plotar_resultados(dados_resultados)
    
    print("\n" + "="*70)
    print("BACKTEST CONCLUÍDO!")
    print("="*70)
    print("\nArquivos gerados:")
    print("   1. backtest_pmi_cpi_detalhado.csv")
    print("   2. backtest_pmi_cpi_graficos.png")
    print("\n✓ Análise completa!")
    print("="*70)

if __name__ == "__main__":
    main()
