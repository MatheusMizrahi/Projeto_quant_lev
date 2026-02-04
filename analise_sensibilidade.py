"""
Análise de Sensibilidade: Suavização + Alocação
================================================

PROBLEMA IDENTIFICADO:
- Classificador V2 distribui regimes DIFERENTEMENTE de V1 ✅
- Mas Sharpe continua negativo (-0.18) em ambos ❌
- 55% do tempo em G2 (Reflação), mas estratégia não captura isso

HIPÓTESES:
1. Suavização (EWM span=5) atrasa sinais demais
2. Alocação por regime está invertida/inadequada
3. Custos de transação (10 bps) muito altos

OBJETIVO: Testar combinações de parâmetros

AUTOR: Matheus Mizrahi
DATA: Fevereiro 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analisar_distribuicao_regimes():
    """Análise 1: Qual regime domina e qual deveria ser lucrativo?"""
    print("\n" + "="*70)
    print("📊 ANÁLISE 1: DISTRIBUIÇÃO DE REGIMES")
    print("="*70)
    
    # Carregar histórico com intensidades
    df = pd.read_csv('historico_intensidade_12_simples_v2.csv', parse_dates=['data'])
    
    # Extrair quadrante simples
    def extrair_q(s):
        if 'Q1' in s: return 'Q1'
        elif 'Q2' in s: return 'Q2'
        elif 'Q3' in s: return 'Q3'
        else: return 'Q4'
    
    df['Q'] = df['quadrante'].apply(extrair_q)
    
    print("\n📈 DISTRIBUIÇÃO TOTAL:")
    dist = df['Q'].value_counts(sort=True)
    for q, count in dist.items():
        pct = count / len(df) * 100
        print(f"   {q}: {count:4} períodos ({pct:5.1f}%)")
    
    # Carregar preços para calcular retornos por regime
    prices = pd.read_csv('data_prices.csv', index_col=0, parse_dates=True)
    
    # Calcular retornos semanais
    ret_sp500 = prices['SP500'].pct_change()
    ret_us10y = prices['US_10Y'].pct_change()
    
    # Merge com regimes
    df_ret = df.copy()
    df_ret = df_ret.set_index('data')
    df_ret['ret_sp500'] = ret_sp500
    df_ret['ret_us10y'] = ret_us10y
    df_ret = df_ret.dropna()
    
    print("\n📈 RETORNO MÉDIO POR REGIME (% semanal):")
    print("   Regime  |  SP500  |  US_10Y  |  Long-Short (0.8 SP500 - 0.2 bonds)")
    print("   " + "-"*60)
    
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = df_ret[df_ret['Q'] == q]
        if len(subset) > 0:
            ret_sp = subset['ret_sp500'].mean() * 100
            ret_bond = subset['ret_us10y'].mean() * 100
            
            # Simular alocação Q1 (0.80 SP500 + 0.20 bonds)
            ret_strat = (0.80 * subset['ret_sp500'] + 0.20 * subset['ret_us10y']).mean() * 100
            
            print(f"   {q:4}    |  {ret_sp:+6.2f}% |  {ret_bond:+6.2f}%  |  {ret_strat:+6.2f}%")
    
    print("\n💡 INTERPRETAÇÃO:")
    print("   • Se retorno da estratégia for NEGATIVO no regime dominante (G2),")
    print("     a alocação está ERRADA para esse regime!")
    print("   • Se SP500 for POSITIVO mas estratégia NEGATIVA → underweight em ações")
    print("   • Se US_10Y for POSITIVO mas estratégia NEGATIVA → underweight em bonds")


def analisar_efeito_suavizacao():
    """Análise 2: Suavização está atrasando sinais?"""
    print("\n" + "="*70)
    print("📊 ANÁLISE 2: EFEITO DA SUAVIZAÇÃO")
    print("="*70)
    
    df = pd.read_csv('historico_quadrantes_v2.csv', parse_dates=['data'])
    
    # Calcular quantas vezes o regime muda
    df['mudanca'] = (df['quadrante'] != df['quadrante'].shift(1)).astype(int)
    n_mudancas = df['mudanca'].sum()
    freq_mudanca = n_mudancas / len(df) * 100
    
    print(f"\n📈 FREQUÊNCIA DE MUDANÇA DE REGIME:")
    print(f"   Total de mudanças: {n_mudancas} em {len(df)} períodos")
    print(f"   Frequência: {freq_mudanca:.1f}% (muda a cada ~{100/freq_mudanca:.1f} semanas)")
    
    print("\n🎯 DIAGNÓSTICO:")
    if freq_mudanca > 30:
        print("   ⚠️  MUITO ALTA (>30%) - Troca demais, custos matam performance")
        print("   💡 SOLUÇÃO: AUMENTAR suavização (span=10 ou 15)")
    elif freq_mudanca < 10:
        print("   ⚠️  MUITO BAIXA (<10%) - Reage muito devagar")
        print("   💡 SOLUÇÃO: REDUZIR suavização (span=3) ou REMOVER")
    else:
        print("   ✅ ADEQUADA (10-30%) - Suavização não é o problema")
    
    # Calcular autocorrelação dos scores
    auto_infl = df['inflacao_score'].autocorr(lag=1)
    auto_ativ = df['atividade_score'].autocorr(lag=1)
    
    print(f"\n📈 AUTOCORRELAÇÃO LAG-1:")
    print(f"   Inflação:  {auto_infl:.3f}")
    print(f"   Atividade: {auto_ativ:.3f}")
    print(f"   Média:     {(auto_infl + auto_ativ)/2:.3f}")
    
    if (auto_infl + auto_ativ)/2 > 0.85:
        print("\n   ⚠️  MUITO SUAVIZADO (>0.85) - Sinais lentos demais")
        print("   💡 SOLUÇÃO: Reduzir span EWM para 3 ou remover completamente")
    else:
        print("\n   ✅ Autocorrelação OK - Suavização não é problema principal")


def sugestoes_alocacao():
    """Análise 3: Sugestões de alocação alternativas."""
    print("\n" + "="*70)
    print("📊 ANÁLISE 3: SUGESTÕES DE ALOCAÇÃO")
    print("="*70)
    
    print("\n🔍 ALOCAÇÃO ATUAL (backtest_6.py):")
    print("""
    Q1 (Goldilocks): 80% SP500 + 20% US_10Y  = +100% (risco máximo)
    Q2 (Reflação):   50% SP500 - 10% US_10Y  = +40%  (cauteloso)
    Q3 (Estagflação): -20% SP500 + 60% US_10Y = +40% (defensivo)
    Q4 (Deflação):   -40% SP500 + 80% US_10Y = +40% (bonds)
    """)
    
    print("\n❌ PROBLEMAS IDENTIFICADOS:")
    print("   1. Q1 expõe 100% do capital - drawdown severo se errar")
    print("   2. Q2 tem SHORT em bonds (-10%) - mas bonds sobem em reflação!")
    print("   3. Q3 tem LONG em bonds (+60%) - correto, mas pode estar suave")
    print("   4. Exposição total varia muito (40% a 100%) - instável")
    
    print("\n" + "="*70)
    print("💡 SUGESTÕES DE MELHORIAS")
    print("="*70)
    
    print("\n🎯 OPÇÃO A: MARKET NEUTRAL (Exposição constante 100%)")
    print("""
    Q1 (Goldilocks):  +60% SP500, -60% US_10Y  (Long ações, Short bonds)
    Q2 (Reflação):    +40% SP500, -40% US_10Y  (Moderado)
    Q3 (Estagflação): -60% SP500, +60% US_10Y  (Short ações, Long bonds)
    Q4 (Deflação):    -80% SP500, +80% US_10Y  (Heavy bonds)
    
    VANTAGENS:
    ✅ Exposição constante = menos ruído
    ✅ Long-Short puro = beta-neutral
    ✅ Simétrico = fácil interpretar
    
    DESVANTAGENS:
    ⚠️  Precisa margem para short
    ⚠️  Custos de short podem ser altos
    """)
    
    print("\n🎯 OPÇÃO B: LONG-ONLY DEFENSIVO (Exposição 60-100%)")
    print("""
    Q1 (Goldilocks):  70% SP500, 30% US_10Y  (Otimista mas controlado)
    Q2 (Reflação):    60% SP500, 40% US_10Y  (Balanceado)
    Q3 (Estagflação): 30% SP500, 70% US_10Y  (Defensivo)
    Q4 (Deflação):    20% SP500, 80% US_10Y  (Heavy bonds)
    
    VANTAGENS:
    ✅ Sem short = sem custos de borrow
    ✅ Sempre investido = não perde rallies
    ✅ Conservador = drawdowns menores
    
    DESVANTAGENS:
    ⚠️  Sempre long = sofre em crashes
    ⚠️  Retorno esperado menor
    """)
    
    print("\n🎯 OPÇÃO C: REGIME-DRIVEN (Exposição 0-100%)")
    print("""
    Q1 (Goldilocks):  80% SP500, 20% US_10Y   (Risk-on)
    Q2 (Reflação):    50% SP500, 50% US_10Y   (Neutro)
    Q3 (Estagflação):  0% SP500, 100% US_10Y  (100% bonds!)
    Q4 (Deflação):    20% SP500, 80% US_10Y   (Heavy bonds)
    
    VANTAGENS:
    ✅ Q3 vai 100% bonds = proteção máxima
    ✅ Sem short = implementação simples
    ✅ Intuitivo = segue teoria econômica
    
    DESVANTAGENS:
    ⚠️  Timing arriscado (tudo em bonds pode errar)
    ⚠️  Menos diversificado que opções A/B
    """)
    
    print("\n🎯 OPÇÃO D: INTENSIDADE-DRIVEN (Usa K-Means)")
    print("""
    Base: Opção B (long-only)
    Modificador por intensidade:
    
    • FRACO:    Reduzir exposição em 30% (manter 40% cash)
    • MODERADO: Manter alocação padrão
    • FORTE:    Aumentar exposição em 20% (usar margem ou concentrar)
    
    Exemplo Q1 Forte: 84% SP500 (70% × 1.2), 36% US_10Y
    Exemplo Q1 Fraco:  49% SP500 (70% × 0.7), 21% US_10Y, 30% Cash
    
    VANTAGENS:
    ✅ Usa informação de intensidade (K-Means não é em vão)
    ✅ Reduz risco em sinais fracos
    ✅ Aumenta retorno em sinais fortes
    
    DESVANTAGENS:
    ⚠️  Mais complexo
    ⚠️  Pode overfittar
    """)


def teste_custos_transacao():
    """Análise 4: Custos estão matando performance?"""
    print("\n" + "="*70)
    print("📊 ANÁLISE 4: SENSIBILIDADE A CUSTOS")
    print("="*70)
    
    df = pd.read_csv('historico_quadrantes_v2.csv', parse_dates=['data'])
    
    # Calcular número de rebalanceamentos
    df['mudanca'] = (df['quadrante'] != df['quadrante'].shift(1)).astype(int)
    n_rebal = df['mudanca'].sum()
    
    # Custos por cenário
    custos_atual = n_rebal * 0.001 * 2  # 10 bps × 2 (compra + venda)
    custos_reduz = n_rebal * 0.0005 * 2  # 5 bps × 2
    custos_zero = 0
    
    print(f"\n📈 CUSTOS DE TRANSAÇÃO:")
    print(f"   Rebalanceamentos totais: {n_rebal}")
    print(f"   Período: {(df['data'].max() - df['data'].min()).days / 365:.1f} anos")
    print(f"\n   Cenário           Custo/Trade    Custo Total    Impacto/Ano")
    print(f"   " + "-"*65)
    print(f"   Atual (10 bps)    0.10%          {custos_atual:.2%}          {custos_atual / ((df['data'].max() - df['data'].min()).days / 365):.2%}")
    print(f"   Reduzido (5 bps)  0.05%          {custos_reduz:.2%}          {custos_reduz / ((df['data'].max() - df['data'].min()).days / 365):.2%}")
    print(f"   Zero              0.00%          {custos_zero:.2%}          {custos_zero:.2%}")
    
    print("\n💡 INTERPRETAÇÃO:")
    print(f"   Se Sharpe = -0.18 com custos de {custos_atual:.2%},")
    print(f"   sem custos seria ≈ Sharpe = {-0.18 + (custos_atual / ((df['data'].max() - df['data'].min()).days / 365) * 2):.2f}")
    print("\n   Se ainda negativo SEM custos → estratégia fundamentalmente falha")
    print("   Se positivo SEM custos → reduzir turnover ou custos")


def main():
    """Executa todas as análises."""
    print("\n" + "="*70)
    print(" "*15 + "ANÁLISE DE SENSIBILIDADE COMPLETA")
    print("="*70)
    
    analisar_distribuicao_regimes()
    analisar_efeito_suavizacao()
    sugestoes_alocacao()
    teste_custos_transacao()
    
    print("\n" + "="*70)
    print("🎯 PLANO DE AÇÃO RECOMENDADO")
    print("="*70)
    print("""
PASSO 1: Testar sem suavização
   • Edite Definicao_quadrante_3_CALIBRADO.py
   • Mude: suavizacao_span=5  →  suavizacao_span=1 (sem suavizar)
   • Rode: python pipeline_completo.py --versao v2
   • Anote Sharpe: __________

PASSO 2: Testar com custos reduzidos
   • Edite backtest_6.py
   • Mude: custo_transacao=0.001  →  custo_transacao=0.0005
   • Rode: python backtest_6.py
   • Anote Sharpe: __________

PASSO 3: Testar OPÇÃO B (Long-Only Defensivo)
   • Edite backtest_6.py, seção ALOCACAO_POR_REGIME
   • Cole:
     "Q1": {"SP500": 0.70, "US_10Y": 0.30},
     "Q2": {"SP500": 0.60, "US_10Y": 0.40},
     "Q3": {"SP500": 0.30, "US_10Y": 0.70},
     "Q4": {"SP500": 0.20, "US_10Y": 0.80}
   • Rode: python backtest_6.py
   • Anote Sharpe: __________

PASSO 4: Se PASSO 3 melhorou, testar OPÇÃO C (Regime-Driven)
   • Q3 = 100% bonds pode ser melhor
   
EXPECTATIVA:
- Sem suavização: Sharpe deve subir ~0.05
- Custos reduzidos: Sharpe deve subir ~0.10
- Opção B/C: Sharpe deve ser POSITIVO (>0.2)

Se após TODOS os testes Sharpe continua negativo:
→ Problema está nos SINAIS DE MOMENTUM (Regressoes_lineares_2.py)
→ Ou nos dados (download_1.py)
→ Ou a estratégia simplesmente não funciona no período testado
    """)


if __name__ == '__main__':
    main()
