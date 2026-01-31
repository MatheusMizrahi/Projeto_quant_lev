"""
Compara 3 versões do classificador de quadrantes:
1. ORIGINAL (pesos arbitrários + thresholds fixos 0.5/0.3)
2. CALIBRADO (pesos calibrados + thresholds fixos 0/0)
3. CALIBRADO + PERCENTIS (pesos calibrados + mediana adaptativa)
"""

import pandas as pd
import numpy as np
from Regressoes_lineares_2 import AnalisadorMomentum
from Definicao_quadrante_3 import ClassificadorQuadrantes  # Original
from Definicao_quadrante_3_CALIBRADO import ClassificadorQuadrantesCalibrado  # Calibrado

print("\n" + "="*80)
print(" "*25 + "COMPARAÇÃO DE MÉTODOS")
print("="*80)

# 1. Calcular momentum (comum para todos)
print("\n📊 Calculando momentum dos ativos...")
analisador = AnalisadorMomentum(verbose=False)
dic_r_ativos = analisador.executar_analise_completa()

# 2. VERSÃO 1: Original (pesos arbitrários + thresholds fixos 0.5/0.3)
print("\n🔵 VERSÃO 1: Original (Arbitrário)")
print("-"*80)
classificador_v1 = ClassificadorQuadrantes(
    limiar_inflacao=0.5,
    limiar_atividade=0.3
)
resultado_v1 = classificador_v1.analisar(dic_r_ativos)

print(f"Quadrante:  {resultado_v1['quadrante']}")
print(f"Inflação:   {resultado_v1['inflacao_score']:.3f}")
print(f"Atividade:  {resultado_v1['atividade_score']:.3f}")
print(f"Thresholds: Inflação={0.5:.3f}, Atividade={0.3:.3f} (fixos)")
print(f"Pesos:      Oil=0.40, Gold=0.30, US10Y=0.20, DXY=-0.10")
print(f"            SP500=0.35, EM=0.25, HY=0.25, US10Y=0.10, DXY=-0.05")

# 3. VERSÃO 2: Calibrado (pesos calibrados + thresholds fixos 0/0)
print("\n🟢 VERSÃO 2: Calibrado + Thresholds Fixos (0, 0)")
print("-"*80)
classificador_v2 = ClassificadorQuadrantesCalibrado(
    usar_percentis=False,
    limiar_inflacao_fixo=0.0,
    limiar_atividade_fixo=0.0
)
resultado_v2 = classificador_v2.analisar(dic_r_ativos, verbose=False)

print(f"Quadrante:  {resultado_v2['quadrante']}")
print(f"Inflação:   {resultado_v2['inflacao_score']:.3f}")
print(f"Atividade:  {resultado_v2['atividade_score']:.3f}")
print(f"Thresholds: Inflação={0.0:.3f}, Atividade={0.0:.3f} (fixos)")
print(f"Pesos:      Oil=0.45, Gold=0.25, US10Y=0.20, DXY=-0.10")
print(f"            SP500=0.40, EM=0.25, HY=0.20, US10Y=0.10, DXY=0.05")

# 4. VERSÃO 3: Calibrado + Percentis (mediana adaptativa)
print("\n🟡 VERSÃO 3: Calibrado + Thresholds Adaptativos (Mediana)")
print("-"*80)
classificador_v3 = ClassificadorQuadrantesCalibrado(
    usar_percentis=True,
    percentil_limiar=50
)
resultado_v3 = classificador_v3.analisar(dic_r_ativos, verbose=False)

print(f"Quadrante:  {resultado_v3['quadrante']}")
print(f"Inflação:   {resultado_v3['inflacao_score']:.3f}")
print(f"Atividade:  {resultado_v3['atividade_score']:.3f}")
print(f"Thresholds: Inflação={resultado_v3['limiar_inflacao_usado']:.3f}, "
      f"Atividade={resultado_v3['limiar_atividade_usado']:.3f} (adaptativos)")

# 5. RESUMO COMPARATIVO
print("\n" + "="*80)
print(" "*30 + "RESUMO COMPARATIVO")
print("="*80)
print(f"\n{'Método':<40} {'Quadrante':<25} {'Infl':<8} {'Ativ':<8}")
print("-"*80)
print(f"{'1. Original (0.4/0.3/0.2...)':<40} {resultado_v1['quadrante']:<25} "
      f"{resultado_v1['inflacao_score']:+.3f}   {resultado_v1['atividade_score']:+.3f}")
print(f"{'2. Calibrado (0.45/0.25...)':<40} {resultado_v2['quadrante']:<25} "
      f"{resultado_v2['inflacao_score']:+.3f}   {resultado_v2['atividade_score']:+.3f}")
print(f"{'3. Calibrado + Percentis':<40} {resultado_v3['quadrante']:<25} "
      f"{resultado_v3['inflacao_score']:+.3f}   {resultado_v3['atividade_score']:+.3f}")
print("="*80)

print("\n💡 RECOMENDAÇÃO:")
print("   • Versão 2 ou 3 são mais robustas (pesos calibrados)")
print("   • Versão 3 adapta thresholds automaticamente (menos arbitrário)")
print("   • Para backtest: teste ambas e compare Sharpe out-of-sample\n")
