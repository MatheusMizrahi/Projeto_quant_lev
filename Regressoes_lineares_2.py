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


def main():
    """
    Função principal para executar o script diretamente.
    """
    analisador = AnalisadorRegressao(verbose=True)
    dic_r_ativos = analisador.executar_analise_completa()
    return dic_r_ativos


# Executar análise e criar variável global para importação fácil
# Apenas quando o script for executado diretamente
if __name__ == "__main__":
    dic_r_ativos = main()
else:
    # Quando importado, executa silenciosamente
    analisador = AnalisadorRegressao(verbose=False)
    dic_r_ativos = analisador.executar_analise_completa()