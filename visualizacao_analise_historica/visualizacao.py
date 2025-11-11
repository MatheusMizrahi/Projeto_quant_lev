"""
Visualiza evolução dos quadrantes ao longo do tempo.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def plotar_evolucao_quadrantes():
    """Plota evolução dos quadrantes no tempo."""
    # Carregar dados da pasta raiz
    caminho_dados = Path(__file__).parent.parent / 'historico_quadrantes.csv'
    df = pd.read_csv(caminho_dados, parse_dates=['data'])
    
    # Criar figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico 1: Scatter plot Inflação vs Atividade
    cores = {
        'Q1: GOLDILOCKS': 'green',
        'Q2: REFLAÇÃO': 'orange',
        'Q3: ESTAGFLAÇÃO': 'red',
        'Q4: DEFLAÇÃO/CONTRAÇÃO': 'blue'
    }
    
    for quadrante, cor in cores.items():
        mask = df['quadrante'] == quadrante
        ax1.scatter(df[mask]['inflacao_score'], 
                   df[mask]['atividade_score'],
                   c=cor, label=quadrante, alpha=0.6, s=50)
    
    ax1.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Limiar Atividade')
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Limiar Inflação')
    ax1.set_xlabel('Inflação Score', fontsize=12)
    ax1.set_ylabel('Atividade Score', fontsize=12)
    ax1.set_title('Distribuição dos Regimes Macroeconômicos', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Linha do tempo
    df['quadrante_num'] = df['quadrante'].map({
        'Q1: GOLDILOCKS': 1,
        'Q2: REFLAÇÃO': 2,
        'Q3: ESTAGFLAÇÃO': 3,
        'Q4: DEFLAÇÃO/CONTRAÇÃO': 4
    })
    
    ax2.plot(df['data'], df['quadrante_num'], marker='o', linestyle='-', linewidth=2)
    ax2.set_yticks([1, 2, 3, 4])
    ax2.set_yticklabels(['Goldilocks', 'Reflação', 'Estagflação', 'Deflação'])
    ax2.set_xlabel('Data', fontsize=12)
    ax2.set_ylabel('Regime', fontsize=12)
    ax2.set_title('Evolução dos Regimes ao Longo do Tempo', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    
    # Salvar na pasta raiz do projeto
    caminho_saida = Path(__file__).parent.parent / 'evolucao_quadrantes.png'
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    print("✓ Gráfico salvo: evolucao_quadrantes.png")
    plt.show()


if __name__ == "__main__":
    plotar_evolucao_quadrantes()