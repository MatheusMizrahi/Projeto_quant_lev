"""
Estratégia de Momentum Simples (Moskowitz et al. 2012)
========================================================

BASEADO EM:
-----------
"Time Series Momentum" 
Moskowitz, Ooi & Pedersen (2012)
Journal of Financial Economics

LÓGICA:
-------
1. Calcular retorno dos últimos 12 meses (skip último mês para evitar reversão)
2. Se retorno > 0 → LONG (peso positivo)
3. Se retorno < 0 → FLAT ou SHORT (peso zero ou negativo)
4. Rebalancear mensalmente (reduz custos)

RESULTADOS ESPERADOS:
---------------------
- Sharpe ratio: 0.6-0.8 (paper original)
- Drawdown máximo: -20% (vs -50% buy-and-hold)
- Funciona em 58 mercados globais

AUTOR: Matheus Mizrahi
DATA: Fevereiro 2026
INSTITUIÇÃO: Insper - IQF + LEV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantstats as qs


class EstrategiaMomentumSimples:
    """
    Time-series momentum puro (sem regimes).
    """
    
    def __init__(self, 
                 lookback_meses=12,
                 skip_mes=1,
                 rebal_frequencia='M',
                 custo_transacao=0.0005):
        """
        Args:
            lookback_meses: Janela de momentum (padrão: 12 meses)
            skip_mes: Pular último mês para evitar reversão (padrão: 1)
            rebal_frequencia: 'M'=mensal, 'W'=semanal (padrão: mensal)
            custo_transacao: Custos (padrão: 5 bps = 0.0005)
        """
        self.lookback_meses = lookback_meses
        self.skip_mes = skip_mes
        self.rebal_freq = rebal_frequencia
        self.custo = custo_transacao
    
    def calcular_sinal_momentum(self, retornos, lookback_dias, skip_dias):
        """
        Calcula sinal de momentum (+1 se positivo, 0 se negativo).
        
        Args:
            retornos: Serie de retornos
            lookback_dias: Janela em dias úteis (~252/12 por mês)
            skip_dias: Dias a pular (~21 = 1 mês)
        
        Returns:
            Serie com sinais (-1, 0, +1)
        """
        # Retorno acumulado de [t-lookback-skip : t-skip]
        retorno_mom = (1 + retornos).rolling(window=lookback_dias).apply(
            lambda x: np.prod(x) - 1, raw=True
        ).shift(skip_dias)
        
        # Sinal binário: +1 se positivo, 0 se negativo
        sinal = (retorno_mom > 0).astype(int)
        
        return sinal
    
    def executar_backtest(self, precos_df, verbose=True):
        """
        Executa backtest completo.
        
        Args:
            precos_df: DataFrame com colunas 'SP500' e 'US_10Y'
            verbose: Mostrar progresso
        
        Returns:
            DataFrame com resultados
        """
        if verbose:
            print("\n" + "="*70)
            print("🚀 ESTRATÉGIA: MOMENTUM SIMPLES (Moskowitz 2012)")
            print("="*70)
            print(f"\n📊 Parâmetros:")
            print(f"   • Lookback: {self.lookback_meses} meses")
            print(f"   • Skip: {self.skip_mes} mês")
            print(f"   • Rebalanceamento: {self.rebal_freq}")
            print(f"   • Custos: {self.custo*100:.2f} bps\n")
        
        # Calcular retornos diários
        ret_sp500 = precos_df['SP500'].pct_change()
        ret_us10y = precos_df['US_10Y'].pct_change()
        
        # Parâmetros em dias úteis (assumindo dados semanais)
        dias_por_mes = 4  # 4 semanas = 1 mês
        lookback_dias = self.lookback_meses * dias_por_mes
        skip_dias = self.skip_mes * dias_por_mes
        
        # Calcular sinais de momentum
        sinal_sp500 = self.calcular_sinal_momentum(ret_sp500, lookback_dias, skip_dias)
        sinal_us10y = self.calcular_sinal_momentum(ret_us10y, lookback_dias, skip_dias)
        
        # Alocar: 60/40 se ambos positivos, ajustar se um negativo
        peso_sp500 = sinal_sp500 * 0.60
        peso_us10y = sinal_us10y * 0.40
        
        # Normalizar para sempre somar 100% (realocar se um ativo sai)
        soma_pesos = peso_sp500 + peso_us10y
        peso_sp500 = peso_sp500 / soma_pesos.replace(0, 1)  # Evitar divisão por zero
        peso_us10y = peso_us10y / soma_pesos.replace(0, 1)
        
        # Preencher NaN inicial
        peso_sp500 = peso_sp500.fillna(0)
        peso_us10y = peso_us10y.fillna(0)
        
        # Calcular retorno da estratégia
        retorno_estrategia = (
            peso_sp500.shift(1) * ret_sp500 + 
            peso_us10y.shift(1) * ret_us10y
        )
        
        # Custos de transação (quando peso muda)
        mudanca_peso = (peso_sp500.diff().abs() + peso_us10y.diff().abs()) / 2
        custos = mudanca_peso * self.custo
        retorno_liquido = retorno_estrategia - custos
        
        # Criar DataFrame de resultados
        resultados = pd.DataFrame({
            'data': precos_df.index,
            'ret_sp500': ret_sp500,
            'ret_us10y': ret_us10y,
            'sinal_sp500': sinal_sp500,
            'sinal_us10y': sinal_us10y,
            'peso_sp500': peso_sp500,
            'peso_us10y': peso_us10y,
            'retorno_estrategia': retorno_liquido,
            'retorno_sp500_bh': ret_sp500,  # Buy & Hold
            'retorno_6040_bh': 0.6 * ret_sp500 + 0.4 * ret_us10y  # 60/40 fixo
        }).set_index('data')
        
        # Calcular patrimônio acumulado
        resultados['patrimonio_estrategia'] = (1 + resultados['retorno_estrategia']).cumprod()
        resultados['patrimonio_sp500'] = (1 + resultados['retorno_sp500_bh']).cumprod()
        resultados['patrimonio_6040'] = (1 + resultados['retorno_6040_bh']).cumprod()
        
        if verbose:
            self._imprimir_metricas(resultados)
        
        return resultados
    
    def _imprimir_metricas(self, resultados):
        """Imprime métricas de performance."""
        ret_anual_est = resultados['retorno_estrategia'].mean() * 52  # Semanal → anual
        ret_anual_sp500 = resultados['retorno_sp500_bh'].mean() * 52
        ret_anual_6040 = resultados['retorno_6040_bh'].mean() * 52
        
        vol_anual_est = resultados['retorno_estrategia'].std() * np.sqrt(52)
        vol_anual_sp500 = resultados['retorno_sp500_bh'].std() * np.sqrt(52)
        vol_anual_6040 = resultados['retorno_6040_bh'].std() * np.sqrt(52)
        
        sharpe_est = ret_anual_est / vol_anual_est if vol_anual_est > 0 else 0
        sharpe_sp500 = ret_anual_sp500 / vol_anual_sp500 if vol_anual_sp500 > 0 else 0
        sharpe_6040 = ret_anual_6040 / vol_anual_6040 if vol_anual_6040 > 0 else 0
        
        print("="*70)
        print("📊 RESULTADOS DO BACKTEST")
        print("="*70)
        print(f"\n{'Métrica':<25} {'Estratégia':>12} {'SP500 B&H':>12} {'60/40 B&H':>12}")
        print("-"*70)
        print(f"{'Retorno Anualizado':<25} {ret_anual_est:>11.2%} {ret_anual_sp500:>11.2%} {ret_anual_6040:>11.2%}")
        print(f"{'Volatilidade Anual':<25} {vol_anual_est:>11.2%} {vol_anual_sp500:>11.2%} {vol_anual_6040:>11.2%}")
        print(f"{'Sharpe Ratio':<25} {sharpe_est:>12.2f} {sharpe_sp500:>12.2f} {sharpe_6040:>12.2f}")
        print("="*70 + "\n")
    
    def plotar_resultados(self, resultados, salvar=True):
        """Plota gráficos de performance."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 1. Patrimônio acumulado
        axes[0].plot(resultados.index, resultados['patrimonio_estrategia'], 
                     label='Momentum Simples', linewidth=2, color='blue')
        axes[0].plot(resultados.index, resultados['patrimonio_sp500'], 
                     label='SP500 Buy & Hold', linewidth=1, alpha=0.7, color='green')
        axes[0].plot(resultados.index, resultados['patrimonio_6040'], 
                     label='60/40 Buy & Hold', linewidth=1, alpha=0.7, color='orange')
        axes[0].set_yscale('log')
        axes[0].set_title('Evolução do Patrimônio (Escala Log)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Capital (R$)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Alocação ao longo do tempo
        axes[1].fill_between(resultados.index, 0, resultados['peso_sp500'], 
                             label='SP500', alpha=0.6, color='green')
        axes[1].fill_between(resultados.index, resultados['peso_sp500'], 1, 
                             label='US 10Y', alpha=0.6, color='orange')
        axes[1].set_title('Alocação de Ativos (Momentum Simples)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Peso (%)')
        axes[1].set_xlabel('Data')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig('backtest_momentum_simples.png', dpi=300)
            print("📊 Gráfico salvo: 'backtest_momentum_simples.png'\n")
        
        plt.show()


def main():
    """Executa estratégia de momentum simples."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Momentum Simples (Moskowitz 2012)')
    parser.add_argument('--lookback', type=int, default=12, help='Meses de lookback (padrão: 12)')
    parser.add_argument('--skip', type=int, default=1, help='Meses a pular (padrão: 1)')
    
    args = parser.parse_args()
    
    # Carregar dados
    precos = pd.read_csv('data_prices.csv', index_col=0, parse_dates=True)
    
    # Criar estratégia
    estrategia = EstrategiaMomentumSimples(
        lookback_meses=args.lookback,
        skip_mes=args.skip
    )
    
    # Executar backtest
    resultados = estrategia.executar_backtest(precos)
    
    # Plotar
    estrategia.plotar_resultados(resultados)
    
    # Salvar resultados
    resultados.to_csv('backtest_momentum_simples.csv')
    print("💾 Resultados salvos: 'backtest_momentum_simples.csv'")
    
    # Relatório QuantStats (se disponível)
    try:
        qs.reports.html(resultados['retorno_estrategia'], output='relatorio_momentum_simples.html')
        print("📄 Relatório HTML: 'relatorio_momentum_simples.html'")
    except:
        print("⚠️  QuantStats não disponível (pip install quantstats)")
    
    return resultados


if __name__ == '__main__':
    resultados = main()
