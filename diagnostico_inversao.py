"""
Diagnóstico de Inversão de Regimes
===================================

PROBLEMA IDENTIFICADO (pelo usuário):
Os retornos históricos por regime NÃO batem com teoria econômica:

Q1 (Goldilocks): SP500 -0.26% (deveria ser POSITIVO!)
Q3 (Estagflação): SP500 +0.24% (deveria ser NEGATIVO!)
Q4 (Deflação): Bonds -0.72% (deveriam ser POSITIVOS!)

HIPÓTESE: Classificador está invertendo os regimes
            (quando marca Q1, na verdade é Q4 - recessão)

AUTOR: Matheus Mizrahi
DATA: Fevereiro 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analisar_correlacao_regimes():
    """Verifica se regimes batem com teoria econômica."""
    print("\n" + "="*70)
    print("🔍 DIAGNÓSTICO: REGIMES vs TEORIA ECONÔMICA")
    print("="*70)
    
    # Carregar dados
    df_reg = pd.read_csv('historico_quadrantes_v2.csv', parse_dates=['data'])
    df_prices = pd.read_csv('data_prices.csv', index_col=0, parse_dates=True)
    
    # Calcular retornos semanais
    ret_sp500 = df_prices['SP500'].pct_change()
    ret_us10y = df_prices['US_10Y'].pct_change()
    
    # Merge com regimes
    df_reg = df_reg.set_index('data')
    df_reg['ret_sp500'] = ret_sp500
    df_reg['ret_us10y'] = ret_us10y
    df_reg = df_reg.dropna()
    
    # Extrair quadrante
    def extrair_q(s):
        if 'Q1' in s: return 'Q1'
        elif 'Q2' in s: return 'Q2'
        elif 'Q3' in s: return 'Q3'
        else: return 'Q4'
    
    df_reg['Q'] = df_reg['quadrante'].apply(extrair_q)
    
    # Analisar retornos por quadrante
    print("\n📊 RETORNOS HISTÓRICOS POR REGIME:")
    print("   " + "-"*65)
    print("   Regime  |  SP500  |  US_10Y  |  Expectativa Teórica")
    print("   " + "-"*65)
    
    expectativas = {
        'Q1': ('POSITIVO', 'NEGATIVO'),   # Goldilocks: ações sobem, bonds caem
        'Q2': ('POSITIVO', 'POSITIVO'),   # Reflação: ambos sobem
        'Q3': ('NEGATIVO', 'POSITIVO'),   # Estagflação: ações caem, bonds sobem
        'Q4': ('NEGATIVO', 'NEGATIVO/POSITIVO')  # Deflação: ambos caem OU flight-to-quality (bonds sobem)
    }
    
    problemas = []
    
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = df_reg[df_reg['Q'] == q]
        if len(subset) > 0:
            ret_sp = subset['ret_sp500'].mean()
            ret_bond = subset['ret_us10y'].mean()
            
            # Determinar se bate com expectativa
            exp_sp, exp_bond = expectativas[q]
            
            status_sp = '✅' if (exp_sp == 'POSITIVO' and ret_sp > 0) or (exp_sp == 'NEGATIVO' and ret_sp < 0) else '❌'
            status_bond = '✅' if ('POSITIVO' in exp_bond and ret_bond > 0) or ('NEGATIVO' in exp_bond and ret_bond < 0) else '❌'
            
            print(f"   {q:4}    |  {ret_sp*100:+6.2f}% {status_sp} |  {ret_bond*100:+6.2f}% {status_bond}  |  {exp_sp} / {exp_bond}")
            
            if status_sp == '❌' or status_bond == '❌':
                problemas.append(q)
    
    print("\n🎯 DIAGNÓSTICO:")
    if len(problemas) > 2:
        print("   ❌ PROBLEMA GRAVE: 3+ regimes não batem com teoria!")
        print("   💡 CAUSA PROVÁVEL: Pesos ou thresholds INVERTIDOS")
        print(f"   📍 Regimes problemáticos: {', '.join(problemas)}")
    elif len(problemas) > 0:
        print(f"   ⚠️  PROBLEMA MODERADO: {len(problemas)} regime(s) não batem")
        print(f"   📍 Regimes problemáticos: {', '.join(problemas)}")
    else:
        print("   ✅ TODOS os regimes batem com teoria econômica!")


def analisar_scores_por_periodo():
    """Analisa se scores fazem sentido nos eventos históricos conhecidos."""
    print("\n" + "="*70)
    print("📅 ANÁLISE: EVENTOS HISTÓRICOS vs CLASSIFICAÇÃO")
    print("="*70)
    
    df = pd.read_csv('historico_quadrantes_v2.csv', parse_dates=['data'])
    
    # Eventos históricos conhecidos
    eventos = {
        '2008-09-15': ('Lehman Crisis', 'Q4 ou Q3', 'Recessão'),
        '2020-03-15': ('COVID Crash', 'Q4', 'Deflação/Pânico'),
        '2013-05-01': ('Taper Tantrum', 'Q3 ou Q2', 'Inflação em alta'),
        '2021-06-01': ('Reflação pós-COVID', 'Q2', 'Crescimento + Inflação'),
        '2022-06-01': ('Fed hiking', 'Q3', 'Estagflação emergente'),
    }
    
    print("\n📍 CLASSIFICAÇÃO EM EVENTOS CONHECIDOS:")
    print("   Data       | Evento            | Esperado | Classificado | Score Infl | Score Ativ")
    print("   " + "-"*90)
    
    for data_str, (nome, esperado, desc) in eventos.items():
        data = pd.to_datetime(data_str)
        # Pegar classificação mais próxima
        idx = (df['data'] - data).abs().argmin()
        row = df.iloc[idx]
        
        classificado = 'Q1' if 'Q1' in row['quadrante'] else \
                      'Q2' if 'Q2' in row['quadrante'] else \
                      'Q3' if 'Q3' in row['quadrante'] else 'Q4'
        
        match = '✅' if classificado in esperado else '❌'
        
        print(f"   {row['data'].strftime('%Y-%m-%d')} | {nome:17} | {esperado:8} | {classificado:12} {match} | {row['inflacao_score']:+7.2f}   | {row['atividade_score']:+7.2f}")


def visualizar_scores_vs_retornos():
    """Plot: scores vs retornos futuros (validar poder preditivo)."""
    print("\n" + "="*70)
    print("📊 ANÁLISE: PODER PREDITIVO DOS SCORES")
    print("="*70)
    
    df_reg = pd.read_csv('historico_quadrantes_v2.csv', parse_dates=['data'])
    df_prices = pd.read_csv('data_prices.csv', index_col=0, parse_dates=True)
    
    # Calcular retornos forward (próxima semana)
    ret_sp500_fwd = df_prices['SP500'].pct_change().shift(-1)
    ret_us10y_fwd = df_prices['US_10Y'].pct_change().shift(-1)
    
    df_reg = df_reg.set_index('data')
    df_reg['ret_sp500_fwd'] = ret_sp500_fwd
    df_reg['ret_us10y_fwd'] = ret_us10y_fwd
    df_reg = df_reg.dropna()
    
    # Correlação entre scores e retornos futuros
    corr_ativ_sp = df_reg['atividade_score'].corr(df_reg['ret_sp500_fwd'])
    corr_infl_bond = df_reg['inflacao_score'].corr(df_reg['ret_us10y_fwd'])
    
    print(f"\n📈 CORRELAÇÃO (Score vs Retorno Futuro):")
    print(f"   Atividade Score vs SP500 Futuro:  {corr_ativ_sp:+.3f}")
    print(f"   Inflação Score vs US10Y Futuro:   {corr_infl_bond:+.3f}")
    
    print("\n💡 INTERPRETAÇÃO:")
    if corr_ativ_sp < -0.05:
        print("   ❌ Atividade Score NEGATIVAMENTE correlacionado com SP500!")
        print("      → ERRO: Score alto deveria indicar SP500 subindo")
        print("      → SOLUÇÃO: INVERTER pesos de atividade (multiplicar por -1)")
    elif corr_ativ_sp < 0.05:
        print("   ⚠️  Atividade Score SEM correlação com SP500 (inútil)")
    else:
        print("   ✅ Atividade Score positivamente correlacionado (correto)")
    
    if corr_infl_bond < -0.05:
        print("\n   ❌ Inflação Score NEGATIVAMENTE correlacionado com Bonds!")
        print("      → ERRO: Inflação alta deveria fazer bonds CAÍREM (relação negativa esperada)")
        print("      → Mas score atual faz bonds SUBIREM quando inflação sobe")
    elif abs(corr_infl_bond) < 0.05:
        print("\n   ⚠️  Inflação Score SEM correlação com US10Y (fraco)")
    else:
        print("\n   ✅ Inflação Score correlacionado com bonds")
    
    # Plotar
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Atividade vs SP500
    axes[0].scatter(df_reg['atividade_score'], df_reg['ret_sp500_fwd']*100, alpha=0.3, s=10)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=0.5)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=0.5)
    axes[0].set_xlabel('Atividade Score')
    axes[0].set_ylabel('SP500 Retorno Futuro (%)')
    axes[0].set_title(f'Atividade vs SP500 Futuro (corr={corr_ativ_sp:.3f})')
    axes[0].grid(True, alpha=0.2)
    
    # Plot 2: Inflação vs Bonds
    axes[1].scatter(df_reg['inflacao_score'], df_reg['ret_us10y_fwd']*100, alpha=0.3, s=10)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=0.5)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('Inflação Score')
    axes[1].set_ylabel('US 10Y Retorno Futuro (%)')
    axes[1].set_title(f'Inflação vs Bonds Futuro (corr={corr_infl_bond:.3f})')
    axes[1].grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('diagnostico_poder_preditivo.png', dpi=300)
    print("\n📊 Gráfico salvo: 'diagnostico_poder_preditivo.png'")
    plt.show()


def sugerir_correcao():
    """Sugere correção baseada no diagnóstico."""
    print("\n" + "="*70)
    print("🔧 SUGESTÕES DE CORREÇÃO")
    print("="*70)
    
    print("""
PROBLEMA IDENTIFICADO:
Regimes classificados não batem com retornos históricos esperados.

POSSÍVEIS CAUSAS:
1. Pesos INVERTIDOS (sinais trocados)
2. Thresholds INVERTIDOS (corta no lugar errado)
3. Interpretação ERRADA dos scores (alto vs baixo)

CORREÇÕES A TESTAR:

OPÇÃO 1: Inverter TODOS os pesos de atividade
   Edite Definicao_quadrante_3_CALIBRADO.py:
   
   PESOS_ATIVIDADE = {
       'SP500': -0.40,          # ← Multiplica por -1
       'MSCI_EM': -0.25,        # ← Multiplica por -1
       'HighYield_ETF': -0.20,  # ← Multiplica por -1
       'US_10Y': -0.10,         # ← Multiplica por -1
       'DXY': +0.05             # ← Multiplica por -1
   }

OPÇÃO 2: Inverter thresholds (trocar sinais)
   Na função identificar_quadrante(), trocar > por < e vice-versa
   
   if atividade < limiar_atividade:  # Era >
       if inflacao > limiar_inflacao:  # Era <
           return "Q1: GOLDILOCKS"
   ...

OPÇÃO 3: Inverter interpretação dos quadrantes
   Renomear os quadrantes (Q1 vira Q4, Q4 vira Q1, etc.)

RECOMENDAÇÃO:
Execute este script completo e veja qual correlação está negativa.
Se atividade_score vs SP500 for negativa → OPÇÃO 1
Se nenhuma correlação negativa → OPÇÃO 3 (renomear)
    """)


def main():
    """Executa diagnóstico completo."""
    print("\n" + "="*70)
    print(" "*15 + "DIAGNÓSTICO DE INVERSÃO DE REGIMES")
    print("="*70)
    print("\nMOTIVAÇÃO: Retornos históricos não batem com teoria econômica")
    print("="*70)
    
    analisar_correlacao_regimes()
    analisar_scores_por_periodo()
    visualizar_scores_vs_retornos()
    sugerir_correcao()
    
    print("\n" + "="*70)
    print("✅ Diagnóstico completo!")
    print("="*70)


if __name__ == '__main__':
    main()
