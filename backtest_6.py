"""backtest.py

Backtest da estratégia Long/Short entre SP500 e Treasury 10Y
usando os regimes macroeconômicos e intensidade do sinal.

Período: 2016 até hoje (dados semanais do histórico de quadrantes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from datetime import datetime

# QuantStats para relatórios profissionais
try:
    import quantstats as qs
    qs.extend_pandas()  # Adiciona métodos do quantstats ao pandas
    QUANTSTATS_DISPONIVEL = True
except ImportError:
    QUANTSTATS_DISPONIVEL = False
    print("⚠️ QuantStats não instalado. Execute: pip install quantstats")

# ============================================================================
# CONFIGURAÇÃO DOS PESOS POR REGIME (mesma lógica do trading_rule.py)
# ============================================================================
#ALOCAÇÃO INICIAL - 2°
# ALOCACAO_POR_REGIME: Dict[str, Dict[str, float]] = {
#     "Q1": {"SP500": 0.75, "US_10Y": 0.25},   # Goldilocks: Long bolsa, Short bonds
#     "Q2": {"SP500": 0.40, "US_10Y": -0.60},   # Reflação: Long bolsa moderado, Short bonds
#     "Q3": {"SP500": -0.50, "US_10Y": -0.50},  # Estagflação: Short ambos
#     "Q4": {"SP500": -0.60, "US_10Y": 0.40},   # Deflação: Short bolsa, Long bonds
# }

#ALOCAÇÃO MARKET NEUTRAL - 3°
# ALOCACAO_POR_REGIME: Dict[str, Dict[str, float]] = {
#     "Q1": {"SP500": 0.60, "US_10Y": -0.60},   # Long ações, Short bonds (sem viés)
#     "Q2": {"SP500": 0.40, "US_10Y": -0.40},   # Long moderado
#     "Q3": {"SP500": -0.50, "US_10Y": 0.50},   # Short ações, Long bonds
#     "Q4": {"SP500": -0.60, "US_10Y": 0.60},   # Short ações forte
# }

#ALOCAÇÃO LONG-ONLY BALANCEADO - 4°
# ALOCACAO_POR_REGIME: Dict[str, Dict[str, float]] = {
#     "Q1": {"SP500": 0.75, "US_10Y": 0.25},    # Risk-on
#     "Q2": {"SP500": 0.60, "US_10Y": 0.40},    # Balanceado
#     "Q3": {"SP500": 0.30, "US_10Y": 0.70},    # Risk-off
#     "Q4": {"SP500": 0.40, "US_10Y": 0.60},    # Defensivo
#     }


#LONG-ONLY DEFENSIVO CALIBRADO - NOVO
ALOCACAO_POR_REGIME: Dict[str, Dict[str, float]] = {
    "Q1": {"SP500": 0.70, "US_10Y": 0.30},    # +100% (risk-on controlado)
    "Q2": {"SP500": 0.60, "US_10Y": 0.40},    # +100% (ambos sobem em reflação!)
    "Q3": {"SP500": 0.30, "US_10Y": 0.70},    # +100% (defensivo)
    "Q4": {"SP500": 0.20, "US_10Y": 0.80},    # +100% (bonds heavy)
    }



FATOR_INTENSIDADE: Dict[str, float] = {
    "forte": 1.0,
    "moderado": 0.6,
    "fraco": 0.3,
}


def extrair_codigo_regime(regime: str) -> str:
    """Extrai Q1, Q2, Q3 ou Q4 do nome completo do regime."""
    regime_upper = regime.upper()
    if regime_upper.startswith("Q1"):
        return "Q1"
    elif regime_upper.startswith("Q2"):
        return "Q2"
    elif regime_upper.startswith("Q3"):
        return "Q3"
    elif regime_upper.startswith("Q4"):
        return "Q4"
    return "Q1"


def obter_fator_intensidade(intensidade: str) -> float:
    """Retorna o fator multiplicador baseado na intensidade."""
    intensidade_lower = str(intensidade).lower().strip()
    if intensidade_lower.startswith("forte"):
        return FATOR_INTENSIDADE["forte"]
    elif intensidade_lower.startswith("moderado"):
        return FATOR_INTENSIDADE["moderado"]
    return FATOR_INTENSIDADE["fraco"]


class Backtest:
    """Classe para executar backtest da estratégia SP500 vs Treasury 10Y."""
    
    def __init__(
        self,
        arquivo_precos: str = "data_prices.csv",
        arquivo_regimes: str = "historico_intensidade_12_simples.csv",
        capital_inicial: float = 100000.0,
        custo_transacao: float = 0.001,  # 0.1% por operação (10 bps)
        rebalanceamento: str = "semanal",  # "semanal" ou "diario"
    ):
        """Inicializa o backtest.
        
        TIMING DA ESTRATÉGIA:
        =====================
        1. CÁLCULO DO SINAL (Fim de semana):
           - Sexta-feira após fechamento do mercado
           - Análise dos dados da semana completa
           - Determinação do regime (Q1/Q2/Q3/Q4) e intensidade
        
        2. EXECUÇÃO DO TRADE (Início da próxima semana):
           - Segunda-feira na abertura do mercado
           - Rebalanceamento para as novas posições
           - Custos de transação aplicados
        
        3. MANUTENÇÃO (Durante a semana):
           - Posições mantidas fixas até o próximo rebalanceamento
           - SEM trades intra-semanais
           - Retornos diários acumulados sobre as posições fixas
        
        Args:
            rebalanceamento: Frequência de rebalanceamento
                - "semanal": Trades apenas 1x por semana (RECOMENDADO)
                - "diario": Rebalanceamento diário (maior custo)
        """
        self.arquivo_precos = arquivo_precos
        self.arquivo_regimes = arquivo_regimes
        self.capital_inicial = capital_inicial
        self.custo_transacao = custo_transacao
        self.rebalanceamento = rebalanceamento
        
        self.precos: pd.DataFrame = None
        self.regimes: pd.DataFrame = None
        self.resultados: pd.DataFrame = None
        
    def carregar_dados(self) -> None:
        """Carrega os dados de preços e regimes."""
        print("=" * 60)
        print(" CARREGANDO DADOS PARA BACKTEST")
        print("=" * 60)
        
        # Carregar preços
        self.precos = pd.read_csv(self.arquivo_precos, index_col=0, parse_dates=True)
        self.precos.index.name = "Date"
        print(f"\n✓ Preços carregados: {len(self.precos)} registros")
        print(f"  Período: {self.precos.index.min().date()} a {self.precos.index.max().date()}")
        print(f"  Ativos: {list(self.precos.columns)}")
        
        # Carregar regimes com intensidade
        self.regimes = pd.read_csv(self.arquivo_regimes)
        self.regimes["data"] = pd.to_datetime(self.regimes["data"])
        self.regimes.set_index("data", inplace=True)

        # ⚠️ CORREÇÃO DO LOOK-AHEAD BIAS
        # Shiftar em 1 período: usamos o sinal da semana ANTERIOR
        # Isso garante que só usamos informação disponível no momento do trade
        print(f"\n✓ Regimes carregados: {len(self.regimes)} registros (antes do shift)")
        print(f"  Período original: {self.regimes.index.min().date()} a {self.regimes.index.max().date()}")
        
        self.regimes = self.regimes.shift(1)
        
        # Remover a primeira linha (será NaN após shift)
        self.regimes = self.regimes.dropna()
        
        print(f"\n⚠️ IMPORTANTE: Sinais shiftados em 1 período")
        print(f"   → Sinal calculado no FIM da semana N")
        print(f"   → Trade executado no INÍCIO da semana N+1")
        print(f"  Período após shift: {self.regimes.index.min().date()} a {self.regimes.index.max().date()}")
        
        # Verificar se temos os ativos necessários
        ativos_necessarios = ["SP500", "US_10Y"]
        for ativo in ativos_necessarios:
            if ativo not in self.precos.columns:
                raise ValueError(f"Ativo '{ativo}' não encontrado nos dados de preços!")
        print(f"\n✓ Ativos para backtest: {ativos_necessarios}")
        
    def calcular_retornos(self) -> pd.DataFrame:
        """Calcula os retornos diários dos ativos."""
        retornos = self.precos[["SP500", "US_10Y"]].pct_change()
        return retornos
    
    def calcular_benchmarks(self, retornos: pd.DataFrame) -> pd.DataFrame:
        """Calcula retornos dos benchmarks: ERC (Risk Parity).
        
        Returns:
            DataFrame com colunas: ret_ERC
        """
        benchmarks = pd.DataFrame(index=retornos.index)
        
        # BENCHMARK: ERC (Risk Parity) com rebalanceamento mensal
        # Pesos ajustados pela volatilidade inversa
        benchmarks["peso_SP500_ERC"] = np.nan
        benchmarks["peso_US10Y_ERC"] = np.nan
        benchmarks["ret_ERC"] = np.nan
        
        # Calcular pesos ERC mensalmente
        janela_vol = 60  # 60 dias de histórico para calcular volatilidade
        
        for i in range(janela_vol, len(retornos)):
            data = retornos.index[i]
            
            # Apenas recalcular no primeiro dia útil do mês (rebalanceamento mensal)
            if i > janela_vol and data.month == retornos.index[i-1].month:
                # Propagar pesos do dia anterior
                benchmarks.loc[data, "peso_SP500_ERC"] = benchmarks.iloc[i-1]["peso_SP500_ERC"]
                benchmarks.loc[data, "peso_US10Y_ERC"] = benchmarks.iloc[i-1]["peso_US10Y_ERC"]
            else:
                # Recalcular pesos (novo mês)
                ret_historico = retornos.iloc[i-janela_vol:i]
                vol_SP500 = ret_historico["SP500"].std() * np.sqrt(252)
                vol_US10Y = ret_historico["US_10Y"].std() * np.sqrt(252)
                
                # Evitar divisão por zero
                if vol_SP500 > 0 and vol_US10Y > 0:
                    # Peso = inverso da volatilidade (normalizado)
                    inv_vol_SP = 1 / vol_SP500
                    inv_vol_US = 1 / vol_US10Y
                    soma_inv_vol = inv_vol_SP + inv_vol_US
                    
                    peso_SP500_ERC = inv_vol_SP / soma_inv_vol
                    peso_US10Y_ERC = inv_vol_US / soma_inv_vol
                else:
                    # Fallback para pesos iguais
                    peso_SP500_ERC = 0.5
                    peso_US10Y_ERC = 0.5
                
                benchmarks.loc[data, "peso_SP500_ERC"] = peso_SP500_ERC
                benchmarks.loc[data, "peso_US10Y_ERC"] = peso_US10Y_ERC
            
            # Calcular retorno ERC do dia
            benchmarks.loc[data, "ret_ERC"] = (
                benchmarks.loc[data, "peso_SP500_ERC"] * retornos.loc[data, "SP500"] +
                benchmarks.loc[data, "peso_US10Y_ERC"] * retornos.loc[data, "US_10Y"]
            )
        
        return benchmarks
    
    def obter_posicoes_regime(self, regime: str, intensidade: str) -> Dict[str, float]:
        """Calcula as posições baseadas no regime e intensidade."""
        codigo = extrair_codigo_regime(regime)
        fator = obter_fator_intensidade(intensidade)
        alocacao_base = ALOCACAO_POR_REGIME.get(codigo, ALOCACAO_POR_REGIME["Q1"])
        
        return {
            ativo: peso * fator 
            for ativo, peso in alocacao_base.items()
        }
    
    def executar_backtest(self) -> pd.DataFrame:
        """Executa o backtest completo.
        
        LÓGICA DE REBALANCEAMENTO:
        ==========================
        SEMANAL (padrão):
        - Trades executados apenas nas datas dos sinais (semanais)
        - Posições mantidas fixas durante toda a semana
        - Menor custo de transação
        - Mais realista para estratégias macro
        
        DIÁRIO:
        - Rebalanceamento diário baseado no último sinal semanal
        - Maior custo de transação
        - Útil para comparação ou estratégias mais ativas
        """
        print("\n" + "=" * 60)
        print(" EXECUTANDO BACKTEST")
        print("=" * 60)
        print(f"\n📊 Modo de rebalanceamento: {self.rebalanceamento.upper()}")
        
        # Calcular retornos diários
        retornos = self.calcular_retornos()
        
        # Calcular benchmarks
        benchmarks = self.calcular_benchmarks(retornos)
        
        # Preparar DataFrame de resultados
        resultados = pd.DataFrame(index=self.precos.index)
        resultados["ret_SP500"] = retornos["SP500"]
        resultados["ret_US_10Y"] = retornos["US_10Y"]
        resultados["ret_ERC"] = benchmarks["ret_ERC"]
        
        # Inicializar colunas de posição
        resultados["pos_SP500"] = 0.0
        resultados["pos_US_10Y"] = 0.0
        resultados["regime"] = ""
        resultados["intensidade"] = ""
        resultados["codigo_regime"] = ""
        resultados["is_rebalance_day"] = False  # Marca dias de rebalanceamento
        
        # Preencher posições baseadas nos regimes
        regime_atual = None
        intensidade_atual = None
        pos_atual = {"SP500": 0.0, "US_10Y": 0.0}
        
        if self.rebalanceamento == "semanal":
            # MODO SEMANAL: Trades apenas nas datas dos sinais
            # Propagamos as posições até o próximo sinal
            for data in resultados.index:
                # Verifica se há novo sinal de regime nesta data
                if data in self.regimes.index:
                    regime_atual = self.regimes.loc[data, "quadrante"]
                    intensidade_atual = self.regimes.loc[data, "intensidade_12"]
                    pos_atual = self.obter_posicoes_regime(regime_atual, intensidade_atual)
                    resultados.loc[data, "is_rebalance_day"] = True
                
                if regime_atual is not None:
                    resultados.loc[data, "regime"] = regime_atual
                    resultados.loc[data, "intensidade"] = intensidade_atual
                    resultados.loc[data, "codigo_regime"] = extrair_codigo_regime(regime_atual)
                    resultados.loc[data, "pos_SP500"] = pos_atual["SP500"]
                    resultados.loc[data, "pos_US_10Y"] = pos_atual["US_10Y"]
        
        else:
            # MODO DIÁRIO: Rebalanceamento diário (mantido por compatibilidade)
            for data in resultados.index:
                if data in self.regimes.index:
                    regime_atual = self.regimes.loc[data, "quadrante"]
                    intensidade_atual = self.regimes.loc[data, "intensidade_12"]
                    pos_atual = self.obter_posicoes_regime(regime_atual, intensidade_atual)
                
                if regime_atual is not None:
                    resultados.loc[data, "regime"] = regime_atual
                    resultados.loc[data, "intensidade"] = intensidade_atual
                    resultados.loc[data, "codigo_regime"] = extrair_codigo_regime(regime_atual)
                    resultados.loc[data, "pos_SP500"] = pos_atual["SP500"]
                    resultados.loc[data, "pos_US_10Y"] = pos_atual["US_10Y"]
                    resultados.loc[data, "is_rebalance_day"] = True
        
        # Calcular retorno da estratégia
        # Retorno = soma dos (peso_ativo * retorno_ativo)
        resultados["ret_estrategia"] = (
            resultados["pos_SP500"] * resultados["ret_SP500"] +
            resultados["pos_US_10Y"] * resultados["ret_US_10Y"]
        )
        
        # Detectar mudanças de posição para calcular custos
        # Custos aplicados APENAS nos dias de rebalanceamento
        resultados["mudanca_SP500"] = resultados["pos_SP500"].diff().abs()
        resultados["mudanca_US_10Y"] = resultados["pos_US_10Y"].diff().abs()
        
        if self.rebalanceamento == "semanal":
            # Custos apenas em dias de rebalanceamento
            resultados["custo_transacao"] = 0.0
            resultados.loc[resultados["is_rebalance_day"], "custo_transacao"] = (
                (resultados.loc[resultados["is_rebalance_day"], "mudanca_SP500"] + 
                 resultados.loc[resultados["is_rebalance_day"], "mudanca_US_10Y"]) * 
                self.custo_transacao
            )
        else:
            # Custos aplicados em todos os dias (modo diário)
            resultados["custo_transacao"] = (
                (resultados["mudanca_SP500"] + resultados["mudanca_US_10Y"]) * 
                self.custo_transacao
            )
        
        resultados["custo_transacao"] = resultados["custo_transacao"].fillna(0)
        
        # Contar número de trades
        num_trades = (resultados["custo_transacao"] > 0).sum()
        print(f"\n📊 Número de rebalanceamentos: {num_trades}")
        print(f"   Custo total de transação: {resultados['custo_transacao'].sum():.4%}")
        
        # Retorno líquido (após custos)
        resultados["ret_estrategia_liq"] = (
            resultados["ret_estrategia"] - resultados["custo_transacao"]
        )
        
        # Calcular retorno acumulado (equity curve)
        resultados["equity_estrategia"] = (
            (1 + resultados["ret_estrategia_liq"]).cumprod() * self.capital_inicial
        )
        resultados["equity_ERC"] = (
            (1 + resultados["ret_ERC"]).cumprod() * self.capital_inicial
        )
        resultados["equity_SP500"] = (
            (1 + resultados["ret_SP500"]).cumprod() * self.capital_inicial
        )
        resultados["equity_US_10Y"] = (
            (1 + resultados["ret_US_10Y"]).cumprod() * self.capital_inicial
        )
        
        # Remover linhas sem dados
        resultados = resultados.dropna(subset=["ret_SP500", "ret_US_10Y"])
        
        # Filtrar a partir da primeira data de regime disponível
        primeira_data_regime = self.regimes.index.min()
        resultados = resultados[resultados.index >= primeira_data_regime]
        
        self.resultados = resultados
        
        print(f"\n✓ Backtest executado com sucesso!")
        print(f"  Período: {resultados.index.min().date()} a {resultados.index.max().date()}")
        print(f"  Total de dias: {len(resultados)}")
        print(f"  Dias de trading: {(resultados['regime'] != '').sum()}")
        if self.rebalanceamento == "semanal":
            print(f"  Rebalanceamentos: {resultados['is_rebalance_day'].sum()}")
            print(f"\n💡 Estratégia semanal: posições mantidas por ~5 dias úteis")
        
        return resultados
    
    def calcular_metricas(self) -> Dict:
        """Calcula métricas de performance."""
        if self.resultados is None:
            raise ValueError("Execute o backtest primeiro!")
        
        df = self.resultados
        
        # Funções auxiliares
        def cagr(equity_series):
            """Calcula CAGR (Compound Annual Growth Rate)."""
            total_return = equity_series.iloc[-1] / equity_series.iloc[0]
            years = (equity_series.index[-1] - equity_series.index[0]).days / 365.25
            return (total_return ** (1/years)) - 1
        
        def sharpe_ratio(returns, rf=0.02):
            """Calcula Sharpe Ratio anualizado."""
            excess_ret = returns.mean() * 252 - rf
            vol = returns.std() * np.sqrt(252)
            return excess_ret / vol if vol > 0 else 0
        
        def max_drawdown(equity_series):
            """Calcula Maximum Drawdown."""
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max
            return drawdown.min()
        
        def sortino_ratio(returns, rf=0.02):
            """Calcula Sortino Ratio."""
            excess_ret = returns.mean() * 252 - rf
            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252)
            return excess_ret / downside_vol if downside_vol > 0 else 0
        
        def calmar_ratio(equity_series, returns):
            """Calcula Calmar Ratio."""
            cagr_val = cagr(equity_series)
            mdd = abs(max_drawdown(equity_series))
            return cagr_val / mdd if mdd > 0 else 0
        
        # Calcular métricas para cada estratégia
        metricas = {}
        
        estrategias = {
            "Estratégia": ("ret_estrategia_liq", "equity_estrategia"),
            "SP500 (B&H)": ("ret_SP500", "equity_SP500"),
            "Treasury 10Y (B&H)": ("ret_US_10Y", "equity_US_10Y"),
            "ERC (Risk Parity)": ("ret_ERC", "equity_ERC")
        }
        
        for nome, (col_ret, col_equity) in estrategias.items():
            ret = df[col_ret].dropna()
            equity = df[col_equity].dropna()
            
            metricas[nome] = {
                "Retorno Total": (equity.iloc[-1] / equity.iloc[0]) - 1,
                "CAGR": cagr(equity),
                "Volatilidade (anual)": ret.std() * np.sqrt(252),
                "Sharpe Ratio": sharpe_ratio(ret),
                "Sortino Ratio": sortino_ratio(ret),
                "Max Drawdown": max_drawdown(equity),
                "Calmar Ratio": calmar_ratio(equity, ret),
                "Capital Final": equity.iloc[-1],
            }
        
        return metricas
    
    def imprimir_metricas(self, metricas: Dict) -> None:
        """Imprime as métricas de forma formatada."""
        print("\n" + "=" * 80)
        print(" MÉTRICAS DE PERFORMANCE - COMPARAÇÃO COM BENCHMARKS")
        print("=" * 80)
        
        # Criar DataFrame para exibição
        df_metricas = pd.DataFrame(metricas).T
        
        # Reordenar para colocar estratégia primeiro
        ordem = ["Estratégia", "SP500 (B&H)", "ERC (Risk Parity)", "Treasury 10Y (B&H)"]
        ordem_existente = [o for o in ordem if o in df_metricas.index]
        df_metricas = df_metricas.reindex(ordem_existente)
        
        # Formatar valores
        formatters = {
            "Retorno Total": "{:.2%}",
            "CAGR": "{:.2%}",
            "Volatilidade (anual)": "{:.2%}",
            "Sharpe Ratio": "{:.2f}",
            "Sortino Ratio": "{:.2f}",
            "Max Drawdown": "{:.2%}",
            "Calmar Ratio": "{:.2f}",
            "Capital Final": "R$ {:,.2f}",
        }
        
        for metrica, fmt in formatters.items():
            print(f"\n{metrica}:")
            for estrategia in df_metricas.index:
                valor = df_metricas.loc[estrategia, metrica]
                # Destacar estratégia principal
                prefix = "►" if estrategia == "Estratégia" else " "
                print(f"  {prefix} {estrategia:25} : {fmt.format(valor)}")
        
        # Análise comparativa
        print("\n" + "=" * 80)
        print(" ANÁLISE COMPARATIVA")
        print("=" * 80)
        
        estrategia_sharpe = df_metricas.loc["Estratégia", "Sharpe Ratio"]
        estrategia_cagr = df_metricas.loc["Estratégia", "CAGR"]
        estrategia_dd = df_metricas.loc["Estratégia", "Max Drawdown"]
        
        # Comparar com cada benchmark
        for bench in ["SP500 (B&H)", "ERC (Risk Parity)", "Treasury 10Y (B&H)"]:
            if bench in df_metricas.index:
                bench_sharpe = df_metricas.loc[bench, "Sharpe Ratio"]
                bench_cagr = df_metricas.loc[bench, "CAGR"]
                bench_dd = df_metricas.loc[bench, "Max Drawdown"]
                
                diff_sharpe = estrategia_sharpe - bench_sharpe
                diff_cagr = estrategia_cagr - bench_cagr
                diff_dd = estrategia_dd - bench_dd
                
                print(f"\nvs {bench}:")
                print(f"  Sharpe:  {diff_sharpe:+.2f} {'✅ Melhor' if diff_sharpe > 0 else '❌ Pior'}")
                print(f"  CAGR:    {diff_cagr:+.2%} {'✅ Melhor' if diff_cagr > 0 else '❌ Pior'}")
                print(f"  DrawDown: {diff_dd:+.2%} {'✅ Menor' if diff_dd > 0 else '❌ Maior'}")
    
    def analisar_por_regime(self) -> pd.DataFrame:
        """Analisa a performance por regime."""
        if self.resultados is None:
            raise ValueError("Execute o backtest primeiro!")
        
        df = self.resultados[self.resultados["codigo_regime"] != ""].copy()
        
        analise = df.groupby("codigo_regime").agg({
            "ret_estrategia_liq": ["mean", "std", "count"],
            "ret_SP500": "mean",
            "ret_US_10Y": "mean",
        })
        
        analise.columns = [
            "Ret Médio Estratégia", "Vol Estratégia", "Dias",
            "Ret Médio SP500", "Ret Médio US10Y"
        ]
        
        # Anualizar
        analise["Ret Médio Estratégia (anual)"] = analise["Ret Médio Estratégia"] * 252
        analise["Vol Estratégia (anual)"] = analise["Vol Estratégia"] * np.sqrt(252)
        
        return analise
    
    def plotar_resultados(self, salvar: bool = True) -> None:
        """Plota os resultados do backtest."""
        if self.resultados is None:
            raise ValueError("Execute o backtest primeiro!")
        
        df = self.resultados
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle("Backtest: Estratégia SP500 vs Treasury 10Y", fontsize=14, fontweight="bold")
        
        # 1. Equity Curves com Benchmarks
        ax1 = axes[0, 0]
        ax1.plot(df.index, df["equity_estrategia"], label="Estratégia", linewidth=2.5, color="blue")
        ax1.plot(df.index, df["equity_SP500"], label="SP500", linewidth=1.5, alpha=0.8, color="green")
        ax1.plot(df.index, df["equity_ERC"], label="ERC (Risk Parity)", linewidth=1.5, alpha=0.8, color="orange")
        ax1.plot(df.index, df["equity_US_10Y"], label="Treasury 10Y", linewidth=1.5, alpha=0.8, color="gray")
        ax1.set_title("Evolução do Patrimônio (Escala Log)")
        ax1.set_ylabel("Capital (R$)")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")
        
        # 2. Drawdown
        ax2 = axes[0, 1]
        rolling_max = df["equity_estrategia"].cummax()
        drawdown = (df["equity_estrategia"] - rolling_max) / rolling_max
        ax2.fill_between(df.index, drawdown, 0, alpha=0.5, color="red")
        ax2.set_title("Drawdown da Estratégia")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)
        
        # 3. Posições ao longo do tempo
        ax3 = axes[1, 0]
        ax3.plot(df.index, df["pos_SP500"], label="SP500", linewidth=1, color="green")
        ax3.plot(df.index, df["pos_US_10Y"], label="Treasury 10Y", linewidth=1, color="orange")
        ax3.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax3.set_title("Alocação dos Ativos")
        ax3.set_ylabel("Peso (%)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-1, 1)
        
        # 4. Distribuição dos regimes
        ax4 = axes[1, 1]
        regime_counts = df["codigo_regime"].value_counts()
        colors = {"Q1": "green", "Q2": "yellow", "Q3": "red", "Q4": "blue"}
        bar_colors = [colors.get(r, "gray") for r in regime_counts.index]
        regime_counts.plot(kind="bar", ax=ax4, color=bar_colors, edgecolor="black")
        ax4.set_title("Distribuição dos Regimes (dias)")
        ax4.set_ylabel("Número de dias")
        ax4.tick_params(axis="x", rotation=0)
        
        # 5. Retornos mensais da estratégia
        ax5 = axes[2, 0]
        ret_mensal = df["ret_estrategia_liq"].resample("M").apply(
            lambda x: (1 + x).prod() - 1
        )
        colors_ret = ["green" if r > 0 else "red" for r in ret_mensal]
        ax5.bar(ret_mensal.index, ret_mensal.values, width=20, color=colors_ret, alpha=0.7)
        ax5.set_title("Retornos Mensais da Estratégia")
        ax5.set_ylabel("Retorno (%)")
        ax5.grid(True, alpha=0.3, axis="y")
        
        # 6. Comparação de Sharpe Ratios
        ax6 = axes[2, 1]
        metricas_todas = self.calcular_metricas()
        nomes = list(metricas_todas.keys())
        sharpes = [metricas_todas[nome]["Sharpe Ratio"] for nome in nomes]
        cores = ["blue", "green", "gray", "orange"][:len(nomes)]
        
        bars = ax6.barh(nomes, sharpes, color=cores, alpha=0.7, edgecolor="black")
        ax6.set_title("Comparação de Sharpe Ratios")
        ax6.set_xlabel("Sharpe Ratio")
        ax6.axvline(x=0, color="black", linestyle="--", linewidth=0.5)
        ax6.grid(True, alpha=0.3, axis="x")
        
        # Adicionar valores nas barras
        for i, (bar, valor) in enumerate(zip(bars, sharpes)):
            ax6.text(valor + 0.05, i, f"{valor:.2f}", va="center", fontsize=9)
        
        plt.tight_layout()
        
        if salvar:
            nome_arquivo = "backtest_resultados.png"
            plt.savefig(nome_arquivo, dpi=150, bbox_inches="tight")
            print(f"\n✓ Gráfico salvo em: {nome_arquivo}")
        
        plt.show()
    
    def exportar_resultados(self, nome_arquivo: str = "backtest_detalhado.csv") -> None:
        """Exporta os resultados detalhados para CSV."""
        if self.resultados is None:
            raise ValueError("Execute o backtest primeiro!")
        
        self.resultados.to_csv(nome_arquivo)
        print(f"\n✓ Resultados exportados para: {nome_arquivo}")
    
    def gerar_relatorio_quantstats(self, benchmark: str = "SP500", nome_html: str = "relatorio_quantstats.html") -> None:
        """Gera relatório completo usando QuantStats.
        
        Args:
            benchmark: Ativo para usar como benchmark ("SP500" ou "US_10Y")
            nome_html: Nome do arquivo HTML do relatório
        """
        if not QUANTSTATS_DISPONIVEL:
            print("\n❌ QuantStats não está instalado!")
            print("   Execute: pip install quantstats")
            return
        
        if self.resultados is None:
            raise ValueError("Execute o backtest primeiro!")
        
        print("\n" + "=" * 60)
        print(" GERANDO RELATÓRIO QUANTSTATS")
        print("=" * 60)
        
        # Preparar retornos da estratégia
        retornos_estrategia = self.resultados["ret_estrategia_liq"].dropna()
        retornos_estrategia.index = pd.to_datetime(retornos_estrategia.index)
        retornos_estrategia.name = "Estratégia"
        
        # Preparar benchmark
        col_bench = f"ret_{benchmark}"
        if col_bench in self.resultados.columns:
            retornos_benchmark = self.resultados[col_bench].dropna()
            retornos_benchmark.index = pd.to_datetime(retornos_benchmark.index)
            retornos_benchmark.name = benchmark
        else:
            retornos_benchmark = None
            print(f"   ⚠️ Benchmark '{benchmark}' não encontrado. Gerando sem benchmark.")
        
        # Gerar relatório HTML completo
        print(f"\n📊 Gerando relatório HTML: {nome_html}")
        qs.reports.html(
            retornos_estrategia,
            benchmark=retornos_benchmark,
            output=nome_html,
            title="Backtest: Estratégia SP500 vs Treasury 10Y",
            download_filename=nome_html
        )
        print(f"✓ Relatório HTML salvo em: {nome_html}")
        
        # Exibir métricas principais no console
        print("\n" + "-" * 60)
        print(" MÉTRICAS QUANTSTATS")
        print("-" * 60)
        
        # Métricas básicas
        print(f"\n📈 CAGR: {qs.stats.cagr(retornos_estrategia):.2%}")
        print(f"📉 Max Drawdown: {qs.stats.max_drawdown(retornos_estrategia):.2%}")
        print(f"📊 Volatilidade (anual): {qs.stats.volatility(retornos_estrategia):.2%}")
        print(f"⚖️ Sharpe Ratio: {qs.stats.sharpe(retornos_estrategia):.2f}")
        print(f"🎯 Sortino Ratio: {qs.stats.sortino(retornos_estrategia):.2f}")
        print(f"📅 Calmar Ratio: {qs.stats.calmar(retornos_estrategia):.2f}")
        print(f"🏆 Win Rate: {qs.stats.win_rate(retornos_estrategia):.2%}")
        print(f"💰 Profit Factor: {qs.stats.profit_factor(retornos_estrategia):.2f}")
        print(f"📆 Best Day: {qs.stats.best(retornos_estrategia):.2%}")
        print(f"📆 Worst Day: {qs.stats.worst(retornos_estrategia):.2%}")
        
        if retornos_benchmark is not None:
            print(f"\n🔄 vs {benchmark}:")
            print(f"   Alpha: {qs.stats.greeks(retornos_estrategia, retornos_benchmark)['alpha']:.4f}")
            print(f"   Beta: {qs.stats.greeks(retornos_estrategia, retornos_benchmark)['beta']:.4f}")
        
        return retornos_estrategia
    
    def plotar_quantstats(self, benchmark: str = "SP500") -> None:
        """Gera gráficos individuais do QuantStats.
        
        Args:
            benchmark: Ativo para usar como benchmark
        """
        if not QUANTSTATS_DISPONIVEL:
            print("\n❌ QuantStats não está instalado!")
            return
        
        if self.resultados is None:
            raise ValueError("Execute o backtest primeiro!")
        
        # Preparar retornos
        retornos = self.resultados["ret_estrategia_liq"].dropna()
        retornos.index = pd.to_datetime(retornos.index)
        
        col_bench = f"ret_{benchmark}"
        bench = None
        if col_bench in self.resultados.columns:
            bench = self.resultados[col_bench].dropna()
            bench.index = pd.to_datetime(bench.index)
        
        print("\n📊 Gerando gráficos QuantStats...")
        
        # 1. Snapshot - visão geral
        print("   → Snapshot")
        qs.plots.snapshot(retornos, title="Snapshot da Estratégia", savefig="qs_snapshot.png")
        
        # 2. Retornos mensais (heatmap)
        print("   → Heatmap de retornos mensais")
        qs.plots.monthly_heatmap(retornos, savefig="qs_monthly_heatmap.png")
        
        # 3. Drawdown
        print("   → Drawdowns")
        qs.plots.drawdown(retornos, savefig="qs_drawdown.png")
        
        # 4. Distribuição dos retornos
        print("   → Distribuição dos retornos")
        qs.plots.histogram(retornos, savefig="qs_histogram.png")
        
        # 5. Rolling Sharpe
        print("   → Rolling Sharpe Ratio")
        qs.plots.rolling_sharpe(retornos, savefig="qs_rolling_sharpe.png")
        
        # 6. Rolling Volatility
        print("   → Rolling Volatility")
        qs.plots.rolling_volatility(retornos, savefig="qs_rolling_vol.png")
        
        print("\n✓ Gráficos salvos: qs_*.png")


def main():
    """Executa o backtest completo."""
    print("\n" + "=" * 60)
    print(" BACKTEST - ESTRATÉGIA SP500 vs TREASURY 10Y")
    print(" Período: 2016 - Presente")
    print(" Timing: Sinais semanais, execução no início da próxima semana")
    print("=" * 60)
    
    # Criar e configurar backtest
    bt = Backtest(
        arquivo_precos="data_prices.csv",
        arquivo_regimes="historico_intensidade_12_simples_v2.csv",
        capital_inicial=100000.0,
        custo_transacao=0.001,  # 10 bps por operação
        rebalanceamento="semanal",  # Rebalanceamento semanal (mais realista)
    )
    
    # Carregar dados
    bt.carregar_dados()
    
    # Executar backtest
    resultados = bt.executar_backtest()
    
    # Calcular e imprimir métricas
    metricas = bt.calcular_metricas()
    bt.imprimir_metricas(metricas)
    
    # Análise por regime
    print("\n" + "=" * 80)
    print(" ANÁLISE POR REGIME")
    print("=" * 80)
    analise_regime = bt.analisar_por_regime()
    print(analise_regime.to_string())
    
    # Plotar resultados
    bt.plotar_resultados(salvar=True)
    
    # Exportar resultados
    bt.exportar_resultados("backtest_detalhado.csv")
    
    # Gerar relatório QuantStats
    if QUANTSTATS_DISPONIVEL:
        bt.gerar_relatorio_quantstats(benchmark="SP500", nome_html="relatorio_quantstats.html")
        bt.plotar_quantstats(benchmark="SP500")
    
    print("\n" + "=" * 60)
    print(" BACKTEST FINALIZADO COM SUCESSO!")
    print("=" * 60)
    
    return bt


if __name__ == "__main__":
    bt = main()
