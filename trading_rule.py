"""trading_rule.py

Esboço MUITO simples de como selecionar ativos Long/Short
usando diretamente os resultados do modelo atual:

- `Regressoes_lineares.py`  → fornece `dic_r_ativos[ativo]['score']`
- `Definicao_quadrante.py` → fornece o regime atual (Q1..Q4)
- `Analise_intensidade.py` → fornece a intensidade (Fraco/Moderado/Forte)

Nada aqui executa trades reais. É apenas uma ideia de
"função cola" entre os módulos existentes.
"""

from typing import Dict, List, Tuple

import pandas as pd

from Regressoes_lineares import dic_r_ativos
from Definicao_quadrante import ClassificadorQuadrantes

# Tipos básicos
Regime = str          # exemplo: "Q1: GOLDILOCKS"
Intensidade = str     # exemplo: "Fraco", "Moderado", "Forte"
Ativo = str           # ticker amigável, ex: "SP500"


def escolher_ativos_simples(
    regime_atual: Regime,
    intensidade_atual: Intensidade,
    scores_ativos: Dict[Ativo, float],
) -> Tuple[Dict[Ativo, float], Dict[Ativo, float]]:
    """Esboço simples de regra de seleção de ativos.

    Args:
        regime_atual: regime macro atual (ex: "Q1: GOLDILOCKS").
        intensidade_atual: intensidade do regime ("Fraco/Moderado/Forte").
        scores_ativos: dicionário com um score por ativo
            (por exemplo, o "score" vindo de `Regressoes_lineares.py`).

    Returns:
        tuple: (longs, shorts), onde cada um é um dicionário
        {ativo: peso_sugerido}.
    """

    print("\n[TRADING_RULE] Iniciando seleção simples de ativos...")
    print(f"  Regime atual: {regime_atual}")
    print(f"  Intensidade atual: {intensidade_atual}")

    # 1. Ordena ativos por score (maior = melhor tendência)
    ordenados: List[Tuple[Ativo, float]] = sorted(
        scores_ativos.items(), key=lambda x: x[1], reverse=True
    )

    if not ordenados:
        print("  Nenhum ativo com score disponível. Abortando.")
        return {}, {}

    print("\n  Ranking de scores (do maior para o menor):")
    for ativo, s in ordenados:
        print(f"    - {ativo}: score = {s:.4f}")

    # 2. Escolha bem ingênua: top N long, bottom N short
    top_n = 5
    longs_brutos = ordenados[:top_n]
    shorts_brutos = ordenados[-top_n:]

    # 3. Ajusta tamanho pela intensidade (fraco < moderado < forte)
    if intensidade_atual.lower().startswith("forte"):
        fator = 1.0
    elif intensidade_atual.lower().startswith("moderado"):
        fator = 0.7
    else:  # fraco ou desconhecido
        fator = 0.4

    print(f"\n  Usando top_n = {top_n} para Long e Short.")
    print(f"  Fator de risco pela intensidade = {fator:.2f}")

    # 4. Converte em pesos simplificados (normaliza para somar ~fator)
    def normalizar(lista: List[Tuple[Ativo, float]]) -> Dict[Ativo, float]:
        soma_abs = sum(abs(s) for _, s in lista) or 1.0
        return {a: fator * (s / soma_abs) for a, s in lista}

    longs = normalizar(longs_brutos)
    shorts = normalizar(shorts_brutos)

    # 5. (Opcional) Poderia filtrar por regime aqui
    #    Ex: se regime_atual começa com 'Q3', penalizar ações etc.

    return longs, shorts


def exemplo_uso() -> None:
    """Exemplo MUITO simples usando os dados reais do modelo.

    Fluxo ilustrativo:
      1. Usa `dic_r_ativos` já calculado em `Regressoes_lineares`.
      2. Usa `ClassificadorQuadrantes` para obter o regime atual.
      3. RECEBE a intensidade de fora (ex.: vinda do K-Means
         em `Analise_intensidade.py`). Aqui mantemos um valor
         fixo só para não complicar o esboço.
    """

    print("\n================ TRADING_RULE.EXEMPLO_USO ================")
    # 1) Scores reais: aproveita o dicionário global `dic_r_ativos`
    scores_ativos = {ativo: info["score"] for ativo, info in dic_r_ativos.items()}
    print("Ativos e scores vindos de Regressoes_lineares:")
    for ativo, info in dic_r_ativos.items():
        print(f"  - {ativo}: score = {info['score']:.4f}, R2 = {info['r_squared']:.4f}")

    # 2) Regime atual: usa o classificador para o último snapshot
    classificador = ClassificadorQuadrantes()
    resultado_quadrante = classificador.analisar(dic_r_ativos)
    regime_atual: Regime = resultado_quadrante["quadrante"]
    print("\nRegime atual calculado por ClassificadorQuadrantes:")
    print(f"  Quadrante: {regime_atual}")
    print(f"  Coordenadas (inflação, atividade): {resultado_quadrante['coordenadas']}")

    # 3) Intensidade atual: lê a última linha do CSV gerado
    #    por `Analise_intensidade.py` (coluna "intensidade_12").
    try:
        hist = pd.read_csv("historico_intensidade_12_simples.csv")
        if "intensidade_12" in hist.columns and not hist.empty:
            intensidade_atual: Intensidade = str(hist["intensidade_12"].iloc[-1])
        else:
            intensidade_atual = "Moderado"  # fallback simples
    except FileNotFoundError:
        intensidade_atual = "Moderado"  # fallback caso ainda não exista o arquivo

    print("\nIntensidade atual vinda de Analise_intensidade/histórico:")
    print(f"  Intensidade_12 (última linha do CSV): {intensidade_atual}")

    longs, shorts = escolher_ativos_simples(
        regime_atual=regime_atual,
        intensidade_atual=intensidade_atual,
        scores_ativos=scores_ativos,
    )

    print("\n---------------- RESULTADO FINAL ----------------")
    print("Regime:", regime_atual, "/ Intensidade:", intensidade_atual)
    print("\nLongs sugeridos:")
    for ativo, peso in longs.items():
        print(f"  {ativo}: {peso:.2f}")

    print("\nShorts sugeridos:")
    for ativo, peso in shorts.items():
        print(f"  {ativo}: {peso:.2f}")


if __name__ == "__main__":
    exemplo_uso()
