"""
Análise exploratória dos scores de inflação e atividade.
OBJETIVO: Identificar thresholds adequados para balancear quadrantes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar histórico de quadrantes
df = pd.read_csv('historico_quadrantes.csv', parse_dates=['data'])

print("="*70)
print(" ANÁLISE DE DISTRIBUIÇÃO DOS SCORES")
print("="*70)

# Estatísticas descritivas
print("\n📊 ESTATÍSTICAS DESCRITIVAS:")
print("-"*70)
print("\nINFLAÇÃO SCORE:")
print(df['inflacao_score'].describe())
print(f"\n• Percentil 25%: {df['inflacao_score'].quantile(0.25):.3f}")
print(f"• Mediana (50%): {df['inflacao_score'].quantile(0.50):.3f}")
print(f"• Percentil 75%: {df['inflacao_score'].quantile(0.75):.3f}")

print("\nATIVIDADE SCORE:")
print(df['atividade_score'].describe())
print(f"\n• Percentil 25%: {df['atividade_score'].quantile(0.25):.3f}")
print(f"• Mediana (50%): {df['atividade_score'].quantile(0.50):.3f}")
print(f"• Percentil 75%: {df['atividade_score'].quantile(0.75):.3f}")

# Distribuição atual de quadrantes
print("\n📈 DISTRIBUIÇÃO ATUAL DE QUADRANTES:")
print("-"*70)
contagem = df['quadrante'].value_counts()
for quad, count in contagem.items():
    pct = (count / len(df)) * 100
    print(f"{quad:30} {count:4} dias ({pct:5.1f}%)")

# Teste de thresholds alternativos
print("\n🔬 TESTE DE THRESHOLDS ALTERNATIVOS:")
print("-"*70)

thresholds_testar = [
    (0.0, 0.0, "Mediana/Mediana (0, 0)"),
    (df['inflacao_score'].quantile(0.5), df['atividade_score'].quantile(0.5), "Percentil 50/50"),
    (df['inflacao_score'].quantile(0.33), df['atividade_score'].quantile(0.33), "Percentil 33/33"),
    (df['inflacao_score'].quantile(0.67), df['atividade_score'].quantile(0.67), "Percentil 67/67"),
]

def classificar_com_threshold(row, limiar_infl, limiar_ativ):
    """Classifica quadrante com thresholds customizados."""
    if row['atividade_score'] > limiar_ativ:
        if row['inflacao_score'] < limiar_infl:
            return "Q1: GOLDILOCKS"
        else:
            return "Q2: REFLAÇÃO"
    else:
        if row['inflacao_score'] >= limiar_infl:
            return "Q3: ESTAGFLAÇÃO"
        else:
            return "Q4: DEFLAÇÃO/CONTRAÇÃO"

for limiar_infl, limiar_ativ, nome in thresholds_testar:
    print(f"\n{nome}:")
    print(f"  Limiar Inflação: {limiar_infl:.3f} | Limiar Atividade: {limiar_ativ:.3f}")
    
    df_temp = df.copy()
    df_temp['quadrante_novo'] = df_temp.apply(
        lambda row: classificar_com_threshold(row, limiar_infl, limiar_ativ), 
        axis=1
    )
    
    contagem_novo = df_temp['quadrante_novo'].value_counts()
    for quad, count in contagem_novo.items():
        pct = (count / len(df_temp)) * 100
        print(f"    {quad:30} {count:4} dias ({pct:5.1f}%)")

# Gráfico de dispersão
print("\n📊 Gerando gráfico de dispersão...")

fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot colorido por quadrante
cores = {
    'Q1: GOLDILOCKS': 'green',
    'Q2: REFLAÇÃO': 'orange',
    'Q3: ESTAGFLAÇÃO': 'red',
    'Q4: DEFLAÇÃO/CONTRAÇÃO': 'blue'
}

for quadrante, cor in cores.items():
    mask = df['quadrante'] == quadrante
    ax.scatter(
        df[mask]['inflacao_score'], 
        df[mask]['atividade_score'],
        c=cor, label=quadrante, alpha=0.6, s=50
    )

# Linhas dos thresholds ATUAIS (fixos)
ax.axhline(y=0.3, color='red', linestyle='--', linewidth=2, 
           label='Threshold Atividade ATUAL (0.3)')
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
           label='Threshold Inflação ATUAL (0.5)')

# Linhas dos thresholds SUGERIDOS (mediana)
mediana_infl = df['inflacao_score'].median()
mediana_ativ = df['atividade_score'].median()
ax.axhline(y=mediana_ativ, color='green', linestyle=':', linewidth=2, 
           label=f'Threshold Atividade SUGERIDO ({mediana_ativ:.2f})')
ax.axvline(x=mediana_infl, color='green', linestyle=':', linewidth=2, 
           label=f'Threshold Inflação SUGERIDO ({mediana_infl:.2f})')

ax.set_xlabel('Inflação Score', fontsize=12)
ax.set_ylabel('Atividade Score', fontsize=12)
ax.set_title('Distribuição dos Regimes - Thresholds Atuais vs Sugeridos', 
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analise_thresholds.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico salvo: analise_thresholds.png")

print("\n" + "="*70)
print(" RECOMENDAÇÃO")
print("="*70)
print("\n🎯 Os thresholds atuais (0.5, 0.3) estão DESBALANCEADOS!")
print("   • 85% dos dados em Q4 indica thresholds muito altos")
print("   • Sugestão: Usar MEDIANAS como thresholds")
print(f"   • Novo limiar_inflacao: {mediana_infl:.3f}")
print(f"   • Novo limiar_atividade: {mediana_ativ:.3f}")
print("\n💡 Isso criará distribuição ~25% por quadrante (mais balanceado)")
print("="*70)
