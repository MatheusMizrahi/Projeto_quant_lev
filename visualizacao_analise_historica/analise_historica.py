"""
Analisa regimes macroeconômicos em janelas móveis históricas.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Adicionar pasta raiz ao path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from Regressoes_lineares import AnalisadorRegressao
from Definicao_quadrante import ClassificadorQuadrantes


class AnalisadorHistorico:
    """
    Roda análise de quadrantes para múltiplos períodos históricos.
    """
    
    def __init__(self, janela_regressao=60, passo_dias=5):
        """
        Args:
            janela_regressao: dias para calcular cada regressão (padrão: 60)
            passo_dias: frequência da análise - 1=diário, 5=semanal, 21=mensal
        """
        self.janela_regressao = janela_regressao
        self.passo_dias = passo_dias
        self.historico_quadrantes = []
    
    def carregar_dados_completos(self):
        """Carrega todos os dados históricos."""
        # Caminho relativo à pasta raiz do projeto
        caminho_dados = Path(__file__).parent.parent / 'data_prices.csv'
        self.data_prices = pd.read_csv(caminho_dados, index_col=0, parse_dates=True)
        print(f"✓ Dados carregados: {len(self.data_prices)} dias")
        print(f"✓ Período: {self.data_prices.index[0]} a {self.data_prices.index[-1]}")
    
    def analisar_periodo(self, data_fim):
        """
        Analisa um período específico (últimos N dias até data_fim).
        
        Returns:
            dict com quadrante, scores e métricas
        """
        # Pegar últimos N dias até data_fim
        data_inicio = data_fim - pd.Timedelta(days=self.janela_regressao)
        dados_janela = self.data_prices[data_inicio:data_fim]
        
        if len(dados_janela) < 30:  # Mínimo para regressão
            return None
        
        # Salvar temporariamente na pasta raiz
        caminho_temp = Path(__file__).parent.parent / 'temp_window.csv'
        dados_janela.to_csv(caminho_temp)
        
        # Rodar regressões na janela
        analisador = AnalisadorRegressao(str(caminho_temp), verbose=False)
        dic_r_ativos = analisador.executar_analise_completa()
        
        # Classificar quadrante
        classificador = ClassificadorQuadrantes()
        resultado = classificador.analisar(dic_r_ativos)
        
        # Adicionar data
        resultado['data'] = data_fim
        
        return resultado
    
    def analisar_historico_completo(self):
        """
        Analisa todos os períodos históricos com step de passo_dias.
        """
        print(f"\n🔄 Iniciando análise histórica...")
        print(f"   Janela: {self.janela_regressao} dias")
        print(f"   Passo: {self.passo_dias} dias")
        
        # Datas para analisar (após janela inicial)
        datas = self.data_prices.index[self.janela_regressao::self.passo_dias]
        
        print(f"   Total de análises: {len(datas)}\n")
        
        for i, data in enumerate(datas):
            resultado = self.analisar_periodo(data)
            if resultado:
                self.historico_quadrantes.append(resultado)
                
                if (i + 1) % 10 == 0:
                    print(f"   Processado: {i+1}/{len(datas)} períodos...")
        
        print(f"\n✓ Análise completa! {len(self.historico_quadrantes)} períodos analisados.\n")
        
        return pd.DataFrame(self.historico_quadrantes)
    
    def gerar_relatorio(self):
        """Gera relatório resumido."""
        df = pd.DataFrame(self.historico_quadrantes)
        
        print("\n" + "="*70)
        print(" "*20 + "RELATÓRIO DE REGIMES HISTÓRICOS")
        print("="*70)
        
        # Distribuição por quadrante
        print("\n📊 DISTRIBUIÇÃO DE QUADRANTES:")
        print("-"*70)
        contagem = df['quadrante'].value_counts()
        for quad, count in contagem.items():
            pct = (count / len(df)) * 100
            print(f"   {quad:25} {count:4} períodos ({pct:5.1f}%)")
        
        # Estatísticas dos scores
        print("\n📈 ESTATÍSTICAS DOS SCORES:")
        print("-"*70)
        print(f"   Inflação Média:   {df['inflacao_score'].mean():.3f}")
        print(f"   Inflação Máxima:  {df['inflacao_score'].max():.3f}")
        print(f"   Inflação Mínima:  {df['inflacao_score'].min():.3f}")
        print(f"\n   Atividade Média:  {df['atividade_score'].mean():.3f}")
        print(f"   Atividade Máxima: {df['atividade_score'].max():.3f}")
        print(f"   Atividade Mínima: {df['atividade_score'].min():.3f}")
        
        # Períodos mais recentes
        print("\n📅 ÚLTIMOS 10 PERÍODOS:")
        print("-"*70)
        for _, row in df.tail(10).iterrows():
            print(f"   {row['data'].strftime('%Y-%m-%d')}  |  {row['quadrante']:25}  |  "
                  f"Infl: {row['inflacao_score']:5.2f}  Ativ: {row['atividade_score']:5.2f}")
        
        print("="*70 + "\n")
        
        return df
    
    def salvar_resultados(self, df):
        """Salva resultados em CSV."""
        # Salvar na pasta raiz do projeto
        caminho_saida = Path(__file__).parent.parent / 'historico_quadrantes.csv'
        df.to_csv(caminho_saida, index=False)
        print(f"✓ Resultados salvos em 'historico_quadrantes.csv'\n")


def main():
    """Executa análise histórica completa."""
    # Criar analisador
    analisador = AnalisadorHistorico(
        janela_regressao=60,  # 60 dias = ~3 meses
        passo_dias=5          # Análise semanal
    )
    
    # Carregar dados
    analisador.carregar_dados_completos()
    
    # Analisar histórico
    df_resultados = analisador.analisar_historico_completo()
    
    # Gerar relatório
    df_resultados = analisador.gerar_relatorio()
    
    # Salvar
    analisador.salvar_resultados(df_resultados)
    
    return df_resultados


if __name__ == "__main__":
    df = main()