"""trading_rule.py

Regra de trading Long/Short entre S&P500 e Treasury 10Y
usando diretamente os resultados do modelo atual:

- `Regressoes_lineares.py`  → fornece `dic_r_ativos[ativo]['score']`
- `Definicao_quadrante.py` → fornece o regime atual (Q1..Q4)
- `Analise_intensidade.py` → fornece a intensidade (Fraco/Moderado/Forte)

Nada aqui executa trades reais. É apenas uma ideia de
"função cola" entre os módulos existentes.

ATIVOS OPERADOS:
- SP500 (S&P 500 Index)
- US_10Y (Treasury 10 anos)
"""

from typing import Dict, Tuple

import pandas as pd

from Regressoes_lineares import dic_r_ativos
from Definicao_quadrante import ClassificadorQuadrantes

# Tipos básicos
Regime = str          # exemplo: "Q1: GOLDILOCKS"
Intensidade = str     # exemplo: "Fraco", "Moderado", "Forte"


# ============================================================================
# CONFIGURAÇÃO DOS PESOS POR REGIME
# ============================================================================
# Cada regime define a alocação base entre SP500 e US_10Y
# Valores positivos = LONG, negativos = SHORT
# Os pesos serão multiplicados pelo fator de intensidade

ALOCACAO_POR_REGIME: Dict[str, Dict[str, float]] = {
    # Q1 GOLDILOCKS: Alta atividade + Baixa inflação
    # → Risk-on: Long bolsa, Short bonds (rates podem subir com crescimento)
    "Q1": {"SP500": 0.70, "US_10Y": -0.30},
    
    # Q2 REFLAÇÃO: Alta atividade + Alta inflação
    # → Risk-on moderado: Long bolsa (com cuidado), Short bonds (inflação alta)
    "Q2": {"SP500": 0.40, "US_10Y": -0.60},
    
    # Q3 ESTAGFLAÇÃO: Baixa atividade + Alta inflação
    # → Risk-off: Short bolsa, posição mista em bonds
    "Q3": {"SP500": -0.50, "US_10Y": -0.50},
    
    # Q4 DEFLAÇÃO/CONTRAÇÃO: Baixa atividade + Baixa inflação
    # → Flight to quality: Short bolsa, Long bonds
    "Q4": {"SP500": -0.60, "US_10Y": 0.40},
}

# Fator de escala baseado na intensidade do sinal
FATOR_INTENSIDADE: Dict[str, float] = {
    "forte": 1.0,      # Sinal forte → posição máxima
    "moderado": 0.6,   # Sinal moderado → posição reduzida
    "fraco": 0.3,      # Sinal fraco → posição mínima
}


def extrair_codigo_regime(regime: str) -> str:
    """Extrai o código do quadrante (Q1, Q2, Q3 ou Q4) do nome completo."""
    if regime.upper().startswith("Q1"):
        return "Q1"
    elif regime.upper().startswith("Q2"):
        return "Q2"
    elif regime.upper().startswith("Q3"):
        return "Q3"
    elif regime.upper().startswith("Q4"):
        return "Q4"
    else:
        return "Q1"  # fallback


def obter_fator_intensidade(intensidade: str) -> float:
    """Retorna o fator multiplicador baseado na intensidade."""
    intensidade_lower = intensidade.lower().strip()
    
    if intensidade_lower.startswith("forte"):
        return FATOR_INTENSIDADE["forte"]
    elif intensidade_lower.startswith("moderado"):
        return FATOR_INTENSIDADE["moderado"]
    else:
        return FATOR_INTENSIDADE["fraco"]


def calcular_posicoes(
    regime_atual: Regime,
    intensidade_atual: Intensidade,
) -> Dict[str, Dict[str, float]]:
    """Calcula as posições Long/Short para SP500 e US_10Y.

    Args:
        regime_atual: regime macro atual (ex: "Q1: GOLDILOCKS").
        intensidade_atual: intensidade do regime ("Fraco/Moderado/Forte").

    Returns:
        dict: {
            'SP500': {'peso': float, 'direcao': 'LONG'/'SHORT'},
            'US_10Y': {'peso': float, 'direcao': 'LONG'/'SHORT'},
            'fator_intensidade': float,
            'regime_codigo': str
        }
    """
    print("\n[TRADING_RULE] Calculando posições SP500 vs US_10Y...")
    print(f"  Regime atual: {regime_atual}")
    print(f"  Intensidade atual: {intensidade_atual}")

    # 1. Identifica o código do regime
    codigo_regime = extrair_codigo_regime(regime_atual)
    print(f"  Código do regime: {codigo_regime}")

    # 2. Obtém alocação base para o regime
    alocacao_base = ALOCACAO_POR_REGIME.get(codigo_regime, ALOCACAO_POR_REGIME["Q1"])
    print(f"\n  Alocação base para {codigo_regime}:")
    for ativo, peso in alocacao_base.items():
        direcao = "LONG" if peso > 0 else "SHORT"
        print(f"    - {ativo}: {peso:+.2f} ({direcao})")

    # 3. Obtém fator de intensidade
    fator = obter_fator_intensidade(intensidade_atual)
    print(f"\n  Fator de intensidade ({intensidade_atual}): {fator:.2f}")

    # 4. Calcula pesos finais
    posicoes = {}
    print("\n  Posições finais (peso_base × fator_intensidade):")
    
    for ativo in ["SP500", "US_10Y"]:
        peso_base = alocacao_base[ativo]
        peso_final = peso_base * fator
        direcao = "LONG" if peso_final > 0 else "SHORT" if peso_final < 0 else "NEUTRO"
        
        posicoes[ativo] = {
            "peso": peso_final,
            "peso_abs": abs(peso_final),
            "direcao": direcao,
            "peso_base": peso_base,
        }
        print(f"    - {ativo}: {peso_final:+.2%} ({direcao})")

    posicoes["_meta"] = {
        "fator_intensidade": fator,
        "regime_codigo": codigo_regime,
        "regime_completo": regime_atual,
        "intensidade": intensidade_atual,
    }

    return posicoes


def gerar_resumo_operacional(posicoes: Dict) -> None:
    """Imprime um resumo operacional das posições."""
    meta = posicoes.get("_meta", {})
    
    print("\n" + "=" * 60)
    print(" RESUMO OPERACIONAL - TRADING RULE")
    print("=" * 60)
    print(f" Regime: {meta.get('regime_completo', 'N/A')}")
    print(f" Intensidade: {meta.get('intensidade', 'N/A')}")
    print(f" Fator de escala: {meta.get('fator_intensidade', 0):.0%}")
    print("-" * 60)
    
    for ativo in ["SP500", "US_10Y"]:
        if ativo in posicoes:
            info = posicoes[ativo]
            emoji = "📈" if info["direcao"] == "LONG" else "📉" if info["direcao"] == "SHORT" else "⏸️"
            nome_completo = "S&P 500" if ativo == "SP500" else "Treasury 10Y"
            print(f" {emoji} {nome_completo:15} | {info['direcao']:6} | Peso: {info['peso']:+.1%}")
    
    print("=" * 60)


def exemplo_uso() -> None:
    """Exemplo usando os dados reais do modelo para SP500 e US_10Y.

    Fluxo:
      1. Usa `ClassificadorQuadrantes` para obter o regime atual.
      2. Lê a intensidade do CSV gerado por `Analise_intensidade.py`.
      3. Calcula as posições Long/Short para os dois ativos.
    """

    print("\n" + "=" * 60)
    print(" TRADING RULE - SP500 vs TREASURY 10Y")
    print("=" * 60)

    # 1) Regime atual: usa o classificador para o último snapshot
    classificador = ClassificadorQuadrantes()
    resultado_quadrante = classificador.analisar(dic_r_ativos)
    regime_atual: Regime = resultado_quadrante["quadrante"]
    
    print("\n[1] REGIME MACRO (ClassificadorQuadrantes)")
    print(f"    Quadrante: {regime_atual}")
    print(f"    Coordenadas (inflação, atividade): {resultado_quadrante['coordenadas']}")
    print(f"    Score Inflação: {resultado_quadrante['inflacao_score']:.4f}")
    print(f"    Score Atividade: {resultado_quadrante['atividade_score']:.4f}")

    # 2) Intensidade atual: lê a última linha do CSV gerado
    #    por `Analise_intensidade.py` (coluna "intensidade_12").
    print("\n[2] INTENSIDADE DO REGIME")
    try:
        hist = pd.read_csv("historico_intensidade_12_simples.csv")
        if "intensidade_12" in hist.columns and not hist.empty:
            intensidade_atual: Intensidade = str(hist["intensidade_12"].iloc[-1])
            print(f"    Fonte: historico_intensidade_12_simples.csv (última linha)")
        else:
            intensidade_atual = "Moderado"
            print(f"    Coluna 'intensidade_12' não encontrada. Usando fallback.")
    except FileNotFoundError:
        intensidade_atual = "Moderado"
        print(f"    Arquivo não encontrado. Usando fallback.")
    
    print(f"    Intensidade: {intensidade_atual}")

    # 3) Mostra scores dos ativos de interesse (para referência)
    print("\n[3] SCORES DOS ATIVOS (Regressoes_lineares)")
    for ativo in ["SP500", "US_10Y"]:
        if ativo in dic_r_ativos:
            info = dic_r_ativos[ativo]
            print(f"    - {ativo}: score = {info['score']:.4f}, R² = {info['r_squared']:.4f}")

    # 4) Calcula posições
    print("\n[4] CÁLCULO DAS POSIÇÕES")
    posicoes = calcular_posicoes(
        regime_atual=regime_atual,
        intensidade_atual=intensidade_atual,
    )

    # 5) Resumo operacional
    gerar_resumo_operacional(posicoes)

    # 6) Retorna para uso programático se necessário
    return posicoes


if __name__ == "__main__":
    exemplo_uso()
