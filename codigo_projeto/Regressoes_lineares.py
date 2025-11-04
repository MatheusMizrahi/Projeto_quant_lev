
import pandas as pd
import statsmodels.api as sm
import numpy as np  # Importar numpy
import warnings

# Ignorar avisos
warnings.simplefilter(action='ignore', category=FutureWarning)
print("Carregando dados salvos...")

#--- 1. Carregando os dados salvos em CSV
try:
    data_prices=pd.read_csv('data_prices.csv',index_col=0, parse_dates=True)
    print(f"✓ Dados carregados com sucesso!")
    print(f"✓ Total de dias: {len(data_prices)}")
    print(data_prices.tail())

except FileNotFoundError:
    print("✗ Arquivo 'data_prices.csv' não encontrado!")
    print("Execute primeiro o arquivo 'download.py' para baixar os dados.")
    exit()


#--- 2. Preparação da Variável Independente (Tempo) ---

# Criar a variável X (Tempo) como um índice numérico (0, 1, 2, ...)
# Usamos np.arange(len(data_prices)) que cria um array [0, 1, 2, ..., N-1]
X_tempo = np.arange(len(data_prices))

# Adicionar a constante (intercepto, ou Beta_0)
X_com_constante = sm.add_constant(X_tempo)

# Renomear a coluna do índice de tempo para clareza no relatório
# A coluna 0 é a constante, a coluna 1 é nosso índice de tempo
X_com_constante = pd.DataFrame(X_com_constante, 
                               columns=['const', 'Time_Index'], 
                               index=data_prices.index)

print("\nVariável X (Tempo) preparada:")
print(X_com_constante.tail())


# --- 3. Execução das Regressões Lineares (Ativo vs. Tempo) ---

print("\n--- Iniciando Análises de Regressão (Ativo vs. Tempo) ---")

#Dicionário R^2 e R
dic_r_ativos={}
# Iterar sobre cada ativo (INCLUINDO o S&P 500 desta vez)
for asset_name in data_prices.columns:
  

    print(f"\n\n=======================================================")
    print(f" REGRESSÃO: {asset_name} (Y) vs. Time_Index (X)")
    print(f"=======================================================")

    # Definir nossa variável dependente (Y)
    Y = data_prices[asset_name]
    print(f"Y.tail {asset_name} = {Y.tail()}")
    
    # Garantir que Y não tenha NaNs que possam ter sobrado
    Y = Y.dropna()
    X_temp = X_com_constante.loc[Y.index] # Alinha os índices

    # Criar e ajustar o modelo de regressão (Mínimos Quadrados Ordinários)
    model = sm.OLS(Y, X_temp)
    results = model.fit()

    # Imprimir o resumo completo da regressão
    print(results.summary())
    
    # Extrair e explicar os principais valores
    beta_0_preco_inicial = results.params['const']
    beta_1_tendencia = results.params['Time_Index']
    r_squared = results.rsquared
    p_value = results.pvalues['Time_Index']
    
    # Score ponderado: sinal(β₁) × √R² (raiz para não penalizar muito)
    if p_value < 0.05:
        score = np.sign(beta_1_tendencia) * np.sqrt(r_squared)
    else:
        score = 0  # Sem tendência confiável
    
    dic_r_ativos[asset_name] = {
        'beta_1': beta_1_tendencia,
        'r_squared': r_squared,
        'score': score,
        'significativo': p_value < 0.05
    }
   
    

    print(f"\n--- Interpretação Resumida do {asset_name} ---")
    print(f"  Preço Inicial Estimado (const): {beta_0_preco_inicial:.4f}")
    print(f"  Tendência Diária (Time_Index):  {beta_1_tendencia:.4f}")
    print(f"  R-quadrado:                     {r_squared*100:.2f}%")
    print("------------------------------")



print("\n\nScript concluído.")