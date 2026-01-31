"""
DIAGNÓSTICO: Por que o classificador v3 piorou o backtest?
==========================================================

Este script analisa as diferenças entre classificadores e identifica
problemas que podem estar degradando performance.

POSSÍVEIS CAUSAS:
-----------------
1. **Excesso de trocas de regime** (turnover alto → custos de transação)
2. **Pesos calibrados inadequados** (teoria ≠ prática)
3. **Thresholds adaptativos ruins** (percentis causando instabilidade)
4. **K-Means over-segmentando** (12 clusters = overfit)
5. **Suavização inadequada** (EWM span muito curto/longo)
6. **Distribuição de regimes desbalanceada** (muito Q4, pouco Q1)

AUTOR: Matheus Mizrahi
DATA: Janeiro 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def carregar_historicos():
    """Carrega os 3 históricos de quadrantes."""
    try:
        v1 = pd.read_csv('historico_quadrantes_v1.csv', parse_dates=['data'])
        v2 = pd.read_csv('historico_quadrantes_v2.csv', parse_dates=['data'])
        v3 = pd.read_csv('historico_quadrantes_v3.csv', parse_dates=['data'])
        
        print(f"✓ V1 (Original): {len(v1)} períodos")
        print(f"✓ V2 (Calibrado): {len(v2)} períodos")
        print(f"✓ V3 (Calibrado+Percentis): {len(v3)} períodos")
        
        return v1, v2, v3
    except FileNotFoundError as e:
        print(f"❌ Erro: {e}")
        print("\n💡 Execute primeiro:")
        print("   python visualizacao_analise_historica/analise_historica_4.py --versao v1")
        print("   python visualizacao_analise_historica/analise_historica_4.py --versao v2")
        print("   python visualizacao_analise_historica/analise_historica_4.py --versao v3")
        raise


def diagnostico_1_turnover(v1, v2, v3):
    """
    DIAGNÓSTICO 1: Taxa de mudança de regime (turnover)
    
    Problema: Trocar muito de regime → custos de transação destroem alpha
    Métrica: % de mudanças regime/semana
    """
    print("\n" + "="*70)
    print("📊 DIAGNÓSTICO 1: TURNOVER DE REGIMES")
    print("="*70)
    
    def calcular_turnover(df):
        """Calcula % de semanas onde regime mudou."""
        mudancas = (df['quadrante'] != df['quadrante'].shift(1)).sum()
        return mudancas / len(df) * 100
    
    turn_v1 = calcular_turnover(v1)
    turn_v2 = calcular_turnover(v2)
    turn_v3 = calcular_turnover(v3)
    
    print(f"\nTurnover de Regimes (% semanas com mudança):")
    print(f"  V1 (Original):           {turn_v1:.1f}%")
    print(f"  V2 (Calibrado):          {turn_v2:.1f}%")
    print(f"  V3 (Calibrado+Percentis): {turn_v3:.1f}%")
    
    print("\n🎯 Análise:")
    if turn_v3 > turn_v1 * 1.5:
        print("  ⚠️  V3 tem turnover 50%+ MAIOR que V1 → PROBLEMA CRÍTICO")
        print("  💡 Solução: Aumentar span EWM (ex: 10 semanas) ou usar hysteresis")
    elif turn_v3 > turn_v1 * 1.2:
        print("  ⚠️  V3 tem turnover 20%+ maior → ATENÇÃO")
        print("  💡 Solução: Ajustar thresholds ou suavização")
    else:
        print("  ✅ Turnover similar entre versões")
    
    return turn_v1, turn_v2, turn_v3


def diagnostico_2_distribuicao(v1, v2, v3):
    """
    DIAGNÓSTICO 2: Distribuição de regimes
    
    Problema: Classificador concentrado em poucos regimes → não captura diversidade
    Métrica: Entropia e % por quadrante
    """
    print("\n" + "="*70)
    print("📊 DIAGNÓSTICO 2: DISTRIBUIÇÃO DE REGIMES")
    print("="*70)
    
    def extrair_quadrante(q):
        """Extrai Q1/Q2/Q3/Q4."""
        if 'Q1' in q.upper():
            return 'Q1'
        elif 'Q2' in q.upper():
            return 'Q2'
        elif 'Q3' in q.upper():
            return 'Q3'
        else:
            return 'Q4'
    
    v1['quad_simples'] = v1['quadrante'].apply(extrair_quadrante)
    v2['quad_simples'] = v2['quadrante'].apply(extrair_quadrante)
    v3['quad_simples'] = v3['quadrante'].apply(extrair_quadrante)
    
    print("\nDistribuição por Quadrante:")
    print("\nV1 (Original):")
    print(v1['quad_simples'].value_counts(normalize=True).sort_index() * 100)
    
    print("\nV2 (Calibrado):")
    print(v2['quad_simples'].value_counts(normalize=True).sort_index() * 100)
    
    print("\nV3 (Calibrado+Percentis):")
    print(v3['quad_simples'].value_counts(normalize=True).sort_index() * 100)
    
    # Calcular entropia (diversidade)
    def calcular_entropia(serie):
        """Entropia de Shannon (0=concentrado, log(4)=uniforme)."""
        probs = serie.value_counts(normalize=True)
        return -(probs * np.log(probs)).sum()
    
    ent_v1 = calcular_entropia(v1['quad_simples'])
    ent_v2 = calcular_entropia(v2['quad_simples'])
    ent_v3 = calcular_entropia(v3['quad_simples'])
    max_ent = np.log(4)  # Máxima entropia para 4 categorias
    
    print(f"\n📈 Entropia (diversidade de regimes):")
    print(f"  V1: {ent_v1:.3f} ({ent_v1/max_ent*100:.1f}% da máxima)")
    print(f"  V2: {ent_v2:.3f} ({ent_v2/max_ent*100:.1f}% da máxima)")
    print(f"  V3: {ent_v3:.3f} ({ent_v3/max_ent*100:.1f}% da máxima)")
    
    print("\n🎯 Análise:")
    if ent_v3 < ent_v1 * 0.8:
        print("  ⚠️  V3 tem entropia 20%+ MENOR → muito concentrado")
        print("  💡 Problema: Classificador não está capturando diversidade de regimes")
    elif ent_v3 < ent_v1 * 0.9:
        print("  ⚠️  V3 tem entropia 10%+ menor → ATENÇÃO")
    else:
        print("  ✅ Entropia similar ou melhor")


def diagnostico_3_scores_dispersion(v1, v2, v3):
    """
    DIAGNÓSTICO 3: Dispersão dos scores
    
    Problema: Scores muito próximos de 0 → sinais fracos → alpha baixo
    Métrica: Desvio padrão e range dos scores
    """
    print("\n" + "="*70)
    print("📊 DIAGNÓSTICO 3: DISPERSÃO DOS SCORES")
    print("="*70)
    
    def stats_scores(df, nome):
        print(f"\n{nome}:")
        print(f"  Inflação Score:")
        print(f"    Média: {df['inflacao_score'].mean():.3f}")
        print(f"    Std:   {df['inflacao_score'].std():.3f}")
        print(f"    Range: [{df['inflacao_score'].min():.3f}, {df['inflacao_score'].max():.3f}]")
        print(f"  Atividade Score:")
        print(f"    Média: {df['atividade_score'].mean():.3f}")
        print(f"    Std:   {df['atividade_score'].std():.3f}")
        print(f"    Range: [{df['atividade_score'].min():.3f}, {df['atividade_score'].max():.3f}]")
    
    stats_scores(v1, "V1 (Original)")
    stats_scores(v2, "V2 (Calibrado)")
    stats_scores(v3, "V3 (Calibrado+Percentis)")
    
    print("\n🎯 Análise:")
    std_v3 = np.mean([v3['inflacao_score'].std(), v3['atividade_score'].std()])
    std_v1 = np.mean([v1['inflacao_score'].std(), v1['atividade_score'].std()])
    
    if std_v3 < std_v1 * 0.7:
        print("  ⚠️  V3 tem dispersão 30%+ MENOR → sinais muito fracos")
        print("  💡 Problema: Suavização excessiva ou thresholds ruins")
    elif std_v3 < std_v1 * 0.85:
        print("  ⚠️  V3 tem dispersão menor → sinais mais fracos")
    else:
        print("  ✅ Dispersão adequada")


def diagnostico_4_correlation_momentum(v1, v2, v3):
    """
    DIAGNÓSTICO 4: Correlação temporal dos scores
    
    Problema: Scores muito autocorrelacionados → lento para reagir
            Scores pouco autocorrelacionados → muito ruído
    Métrica: Autocorrelação lag-1
    """
    print("\n" + "="*70)
    print("📊 DIAGNÓSTICO 4: AUTOCORRELAÇÃO DOS SCORES")
    print("="*70)
    
    def calc_autocorr(df):
        auto_infl = df['inflacao_score'].autocorr(lag=1)
        auto_ativ = df['atividade_score'].autocorr(lag=1)
        return auto_infl, auto_ativ
    
    auto_v1 = calc_autocorr(v1)
    auto_v2 = calc_autocorr(v2)
    auto_v3 = calc_autocorr(v3)
    
    print(f"\nAutocorrelação lag-1:")
    print(f"  V1: Inflação={auto_v1[0]:.3f}, Atividade={auto_v1[1]:.3f}")
    print(f"  V2: Inflação={auto_v2[0]:.3f}, Atividade={auto_v2[1]:.3f}")
    print(f"  V3: Inflação={auto_v3[0]:.3f}, Atividade={auto_v3[1]:.3f}")
    
    print("\n🎯 Análise:")
    media_v3 = np.mean(auto_v3)
    if media_v3 > 0.9:
        print("  ⚠️  Autocorrelação > 0.9 → MUITO suavizado (reage lento)")
        print("  💡 Solução: Reduzir span EWM (ex: 3 semanas)")
    elif media_v3 < 0.3:
        print("  ⚠️  Autocorrelação < 0.3 → MUITO ruidoso (troca demais)")
        print("  💡 Solução: Aumentar span EWM (ex: 8 semanas)")
    else:
        print("  ✅ Autocorrelação adequada (0.3-0.9)")


def diagnostico_5_thresholds(v2, v3):
    """
    DIAGNÓSTICO 5: Efeito dos thresholds adaptativos
    
    Problema: Percentis podem estar gerando thresholds ruins
    """
    print("\n" + "="*70)
    print("📊 DIAGNÓSTICO 5: THRESHOLDS (V2 vs V3)")
    print("="*70)
    
    print(f"\nV2 (Thresholds fixos em 0, 0):")
    print(f"  Inflação: sempre 0.0")
    print(f"  Atividade: sempre 0.0")
    
    print(f"\nV3 (Thresholds adaptativos - percentil 50):")
    print(f"  Inflação: mediana = {v3['inflacao_score'].median():.3f}")
    print(f"  Atividade: mediana = {v3['atividade_score'].median():.3f}")
    
    print("\n🎯 Análise:")
    med_infl = v3['inflacao_score'].median()
    med_ativ = v3['atividade_score'].median()
    
    if abs(med_infl) > 0.3 or abs(med_ativ) > 0.3:
        print("  ⚠️  Medianas LONGE de zero → thresholds adaptativos ruins")
        print("  💡 Problema: Percentis não são zero-centered → viés sistemático")
        print("  💡 Solução: Use thresholds fixos em 0 (V2) ou normalize scores")
    else:
        print("  ✅ Medianas próximas de zero")


def visualizar_comparacao_scores():
    """Gera scatter plots comparando V1 vs V2 vs V3."""
    v1, v2, v3 = carregar_historicos()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (df, nome) in enumerate([(v1, 'V1: Original'), 
                                       (v2, 'V2: Calibrado'), 
                                       (v3, 'V3: Calibrado+Percentis')]):
        ax = axes[idx]
        scatter = ax.scatter(df['inflacao_score'], df['atividade_score'], 
                            alpha=0.5, s=30, c=range(len(df)), cmap='viridis')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Inflação Score')
        ax.set_ylabel('Atividade Score')
        ax.set_title(nome)
        ax.grid(True, alpha=0.2)
        
        # Estatísticas
        std_infl = df['inflacao_score'].std()
        std_ativ = df['atividade_score'].std()
        ax.text(0.02, 0.98, f'σ_infl={std_infl:.2f}\nσ_ativ={std_ativ:.2f}',
                transform=ax.transAxes, va='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('diagnostico_scores_comparacao.png', dpi=300)
    print("\n📊 Gráfico salvo em 'diagnostico_scores_comparacao.png'")
    plt.show()


def main():
    """Executa todos os diagnósticos."""
    print("\n" + "="*70)
    print(" "*15 + "DIAGNÓSTICO DO CLASSIFICADOR V3")
    print("="*70)
    print("\nObjetivo: Identificar por que V3 piorou o backtest")
    print("="*70)
    
    try:
        v1, v2, v3 = carregar_historicos()
        
        # Executar diagnósticos
        diagnostico_1_turnover(v1, v2, v3)
        diagnostico_2_distribuicao(v1, v2, v3)
        diagnostico_3_scores_dispersion(v1, v2, v3)
        diagnostico_4_correlation_momentum(v1, v2, v3)
        diagnostico_5_thresholds(v2, v3)
        
        # Visualização
        visualizar_comparacao_scores()
        
        print("\n" + "="*70)
        print("💡 RECOMENDAÇÕES FINAIS")
        print("="*70)
        print("""
1. Se turnover V3 >> V1:
   → Aumentar span EWM (10 semanas)
   → Adicionar hysteresis (threshold diferente para entrada/saída)

2. Se distribuição muito concentrada:
   → Revisar pesos calibrados (podem estar errados)
   → Testar thresholds fixos (V2) ao invés de adaptativos

3. Se scores muito fracos (std baixo):
   → Problema: suavização excessiva
   → Reduzir span EWM ou remover suavização

4. Se autocorrelação > 0.9:
   → Sistema muito lento para reagir
   → Reduzir span EWM

5. Se thresholds adaptativos ruins:
   → Voltar para thresholds fixos em 0 (V2)
   → Ou normalizar scores antes de aplicar percentis

6. SUGESTÃO PRINCIPAL:
   → Teste V2 (pesos calibrados + thresholds 0/0) SEM percentis
   → V2 deve ter performance entre V1 e V3
   → Se V2 também for ruim: problema está nos PESOS, não nos thresholds
        """)
        
        print("\n✅ Diagnóstico completo!")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
