import pandas as pd
import statsmodels.api as sm
import numpy as np
import warnings


class AnalisadorRegressao:
    """
    Classe para realizar análises de regressão linear dos ativos vs. tempo.
    """
    
    def __init__(self, arquivo_csv='data_prices.csv', verbose=True):
        """
        Inicializa o analisador.
        
        Args:
            arquivo_csv (str): Caminho do arquivo CSV com os dados
            verbose (bool): Se True, imprime informações durante o processamento
        """
        self.arquivo_csv = arquivo_csv
        self.verbose = verbose
        self.data_prices = None
        self.X_com_constante = None
        self.dic_r_ativos = {}
        
        # Ignorar avisos
        warnings.simplefilter(action='ignore', category=FutureWarning)
    
    def carregar_dados(self):
        """Carrega os dados do arquivo CSV."""
        if self.verbose:
            print("Carregando dados salvos...")
        
        try:
            self.data_prices = pd.read_csv(self.arquivo_csv, index_col=0, parse_dates=True)
            if self.verbose:
                print(f"✓ Dados carregados com sucesso!")
                print(f"✓ Total de dias: {len(self.data_prices)}")
                print(self.data_prices.tail())
        except FileNotFoundError:
            print(f"✗ Arquivo '{self.arquivo_csv}' não encontrado!")
            print("Execute primeiro o arquivo 'download.py' para baixar os dados.")
            raise
    
    def preparar_variavel_tempo(self):
        """Prepara a variável independente (Tempo) para as regressões."""
        # Criar a variável X (Tempo) como um índice numérico (0, 1, 2, ...)
        X_tempo = np.arange(len(self.data_prices))
        
        # Adicionar a constante (intercepto, ou Beta_0)
        X_com_constante = sm.add_constant(X_tempo)
        
        # Renomear colunas para clareza
        self.X_com_constante = pd.DataFrame(
            X_com_constante, 
            columns=['const', 'Time_Index'], 
            index=self.data_prices.index
        )
        
        if self.verbose:
            print("\nVariável X (Tempo) preparada:")
            print(self.X_com_constante.tail())
    
    def executar_regressoes(self):
        """Executa as regressões lineares para todos os ativos."""
        if self.verbose:
            print("\n--- Iniciando Análises de Regressão (Ativo vs. Tempo) ---")
        
        # Iterar sobre cada ativo
        for asset_name in self.data_prices.columns:
            if self.verbose:
                print(f"\n\n=======================================================")
                print(f" REGRESSÃO: {asset_name} (Y) vs. Time_Index (X)")
                print(f"=======================================================")
            
            # Definir variável dependente (Y)
            Y = self.data_prices[asset_name]
            if self.verbose:
                print(f"Y.tail {asset_name} = {Y.tail()}")
            
            # Garantir que Y não tenha NaNs
            Y = Y.dropna()
            X_temp = self.X_com_constante.loc[Y.index]  # Alinha os índices
            
            # Criar e ajustar o modelo de regressão
            model = sm.OLS(Y, X_temp)
            results = model.fit()
            
            # Imprimir o resumo completo da regressão
            if self.verbose:
                print(results.summary())
            
            # Extrair valores principais
            beta_0_preco_inicial = results.params['const']
            beta_1_tendencia = results.params['Time_Index']
            r_squared = results.rsquared
            p_value = results.pvalues['Time_Index']
            
            # Score ponderado: sinal(β₁) × √R² (raiz para não penalizar muito)
            if p_value < 0.05:
                score = np.sign(beta_1_tendencia) * np.sqrt(r_squared)
            else:
                score = 0  # Sem tendência confiável
            
            # Armazenar resultados no dicionário
            self.dic_r_ativos[asset_name] = {
                'beta_1': beta_1_tendencia,
                'r_squared': r_squared,
                'score': score,
                'significativo': p_value < 0.05
            }
            
            if self.verbose:
                print(f"\n--- Interpretação Resumida do {asset_name} ---")
                print(f"  Preço Inicial Estimado (const): {beta_0_preco_inicial:.4f}")
                print(f"  Tendência Diária (Time_Index):  {beta_1_tendencia:.4f}")
                print(f"  R-quadrado:                     {r_squared*100:.2f}%")
                print("------------------------------")
    
    def executar_analise_completa(self):
        """
        Executa toda a análise: carrega dados, prepara variáveis e executa regressões.
        
        Returns:
            dict: Dicionário com os resultados das regressões
        """
        self.carregar_dados()
        self.preparar_variavel_tempo()
        self.executar_regressoes()
        
        if self.verbose:
            print("\n\nScript concluído.")
        
        return self.dic_r_ativos
    
    def get_resultados(self):
        """
        Retorna o dicionário com os resultados das análises.
        
        Returns:
            dict: Dicionário com os R² e scores dos ativos
        """
        return self.dic_r_ativos
    

class AnalisadorMomentum:
    """
    Momentum multi-timeframe baseado em Moreira & Muir (2017).
    
    REFERÊNCIA ACADÊMICA:
    ---------------------
    Moreira, A., & Muir, T. (2017). "Volatility-Managed Portfolios"
    Journal of Finance, 72(4), 1611-1644.
    
    PRINCÍPIO:
    ----------
    Separação entre SINAL (momentum) e SIZING (volatilidade):
    1. SINAL: Identificar direção/força da tendência (momentum bruto)
    2. SIZING: Ajustar exposição por volatilidade (implementado no backtest)
    
    Evita duplicação de volatilidade (vol²) que não tem embasamento teórico.
    """

    def __init__(self, arquivo_csv='data_prices.csv', verbose=True):
        """
        Inicializa o analisador.
        
        Args:
            arquivo_csv (str): Caminho do arquivo CSV com os dados
            verbose (bool): Se True, imprime informações durante o processamento
        """
        self.arquivo_csv = arquivo_csv
        self.verbose = verbose
        self.data_prices = None
        self.dic_r_ativos = {}
        
        # Ignorar avisos
        warnings.simplefilter(action='ignore', category=FutureWarning)
    
    def carregar_dados(self):
        """Carrega os dados do arquivo CSV."""
        if self.verbose:
            print("Carregando dados para análise de momentum...")
        
        try:
            self.data_prices = pd.read_csv(self.arquivo_csv, index_col=0, parse_dates=True)
            if self.verbose:
                print(f"✓ Dados carregados com sucesso!")
                print(f"✓ Total de períodos: {len(self.data_prices)}")
        except FileNotFoundError:
            print(f"✗ Arquivo '{self.arquivo_csv}' não encontrado!")
            print("Execute primeiro o arquivo 'download_1.py' para baixar os dados.")
            raise
    
    def calcular_momentum_multi_timeframe(self, prices, lookbacks=[4, 13, 26, 52]):
        """
        Time-Series Momentum robusto (Moreira & Muir 2017 + Moskowitz et al. 2012).
        
        METODOLOGIA:
        ------------
        1. Calcula retorno simples em múltiplas janelas [1m, 3m, 6m, 12m]
        2. Pondera por √(janela) - janelas maiores = mais peso
        3. Retorna score [-1, +1] indicando força/direção da tendência
        
        IMPORTANTE: Volatilidade é usada SEPARADAMENTE no backtest (vol-targeting),
        NÃO no cálculo do sinal (evita duplicação vol²).
        
        Args:
            prices (pd.Series): Série de preços históricos
            lookbacks (list): Janelas temporais em OBSERVAÇÕES (semanas para dados semanais)
                - 4 obs ≈ 1 mês
                - 13 obs ≈ 3 meses (1 quarter)
                - 26 obs ≈ 6 meses
                - 52 obs ≈ 12 meses (1 ano)
        
        Returns:
            float: Score normalizado entre -1 (forte tendência baixa) e +1 (forte tendência alta)
            
        Referências:
            - Moskowitz et al. (2012): "Time Series Momentum" - JFE
            - Moreira & Muir (2017): "Volatility-Managed Portfolios" - JF
        """
        momentums = []
        weights = []
        
        for lb in lookbacks:
            if len(prices) < lb:
                continue
            
            # Retorno simples (SEM ajuste por volatilidade)
            # Moreira & Muir (2017): "momentum sign" - apenas direção
            ret = (prices.iloc[-1] / prices.iloc[-lb]) - 1
            
            # Peso proporcional à raiz da janela
            # Heurística: janelas maiores capturam tendências mais sustentáveis
            # √52 ≈ 7.21, √26 ≈ 5.10, √13 ≈ 3.61, √4 = 2.00
            weight = np.sqrt(lb)
            
            momentums.append(ret)
            weights.append(weight)
        
        if len(momentums) == 0:
            return 0
        
        # Normalizar pesos para somarem 1.0
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Média ponderada dos retornos
        score = sum(m * w for m, w in zip(momentums, normalized_weights))
        
        # Normalizar para escala [-1, +1]
        # Heurística: retorno de ±50% = score máximo (±1)
        # Justificativa: retornos típicos anuais estão entre -30% e +30%
        # 50% é um limite conservador para eventos extremos
        score_normalizado = np.clip(score / 0.5, -1, 1)
        
        return score_normalizado
    
    def executar_analise_momentum(self):
        """
        Executa análise de momentum para todos os ativos.
        
        Returns:
            dict: Dicionário com scores de momentum para cada ativo
        """
        if self.verbose:
            print("\n" + "="*70)
            print(" ANÁLISE DE MOMENTUM MULTI-TIMEFRAME")
            print(" Baseado em: Moreira & Muir (2017) + Moskowitz et al. (2012)")
            print("="*70)
        
        for asset_name in self.data_prices.columns:
            if self.verbose:
                print(f"\n📊 Analisando {asset_name}...")
            
            # Obter série de preços do ativo
            prices = self.data_prices[asset_name].dropna()
            
            # Calcular momentum
            score = self.calcular_momentum_multi_timeframe(prices)
            
            # Calcular retornos individuais para relatório (adaptado para dados semanais)
            ret_1m = (prices.iloc[-1] / prices.iloc[-4] - 1) if len(prices) >= 4 else 0
            ret_3m = (prices.iloc[-1] / prices.iloc[-13] - 1) if len(prices) >= 13 else 0
            ret_6m = (prices.iloc[-1] / prices.iloc[-26] - 1) if len(prices) >= 26 else 0
            ret_12m = (prices.iloc[-1] / prices.iloc[-52] - 1) if len(prices) >= 52 else 0
            
            # Armazenar resultados
            self.dic_r_ativos[asset_name] = {
                'score': score,
                'ret_1m': ret_1m,
                'ret_3m': ret_3m,
                'ret_6m': ret_6m,
                'ret_12m': ret_12m,
                'tendencia': 'ALTA' if score > 0.2 else ('BAIXA' if score < -0.2 else 'NEUTRO')
            }
            
            if self.verbose:
                print(f"  └─ Score: {score:+.3f} | Tendência: {self.dic_r_ativos[asset_name]['tendencia']}")
                print(f"     Retornos: 1m={ret_1m:+.2%} | 3m={ret_3m:+.2%} | 6m={ret_6m:+.2%} | 12m={ret_12m:+.2%}")
        
        return self.dic_r_ativos
    
    def executar_analise_completa(self):
        """
        Executa toda a análise: carrega dados e calcula momentum.
        
        Returns:
            dict: Dicionário com os resultados das análises
        """
        self.carregar_dados()
        resultados = self.executar_analise_momentum()
        
        if self.verbose:
            print("\n✓ Análise de momentum concluída.")
        
        return resultados
    
    def get_resultados(self):
        """
        Retorna o dicionário com os resultados das análises.
        
        Returns:
            dict: Dicionário com scores e métricas dos ativos
        """
        return self.dic_r_ativos


def main():
    """
    Função principal para executar o script diretamente.
    
    ESCOLHA DO MÉTODO:
    ------------------
    - AnalisadorRegressao: OLS tradicional (baseline para comparação)
    - AnalisadorMomentum: Momentum multi-timeframe (Moreira & Muir 2017) ✅ RECOMENDADO
    """
    print("\n🔬 Escolha o método de análise:")
    print("1. OLS (Regressão Linear) - Baseline")
    print("2. Momentum Multi-Timeframe (Moreira & Muir 2017) ✅ RECOMENDADO")
    
    try:
        escolha = input("\nDigite 1 ou 2 (padrão=2): ").strip()
        escolha = escolha if escolha else "2"
    except:
        escolha = "2"
    
    if escolha == "1":
        print("\n📊 Executando análise com OLS (Regressão Linear)...\n")
        analisador = AnalisadorRegressao(verbose=True)
    else:
        print("\n📊 Executando análise com Momentum Multi-Timeframe...\n")
        analisador = AnalisadorMomentum(verbose=True)
    
    dic_r_ativos = analisador.executar_analise_completa()
    return dic_r_ativos


# Executar análise e criar variável global para importação fácil
# Apenas quando o script for executado diretamente
if __name__ == "__main__":
    dic_r_ativos = main()
else:
    # Quando importado por outros scripts, apenas inicializa variável
    # A análise será executada por cada script que precisar
    dic_r_ativos = None