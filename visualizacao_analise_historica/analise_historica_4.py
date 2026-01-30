"""
Analisa regimes macroeconômicos em janelas móveis históricas.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Adicionar pasta raiz ao path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from Regressoes_lineares_2 import AnalisadorRegressao, AnalisadorMomentum
from Definicao_quadrante_3 import ClassificadorQuadrantes


class AnalisadorHistorico:
    """
    Roda análise de quadrantes para múltiplos períodos históricos.
    """
    
    def __init__(self, janela_obs=52, passo_obs=1, verbose=True):
        """
        Args:
            janela_obs: número de observações (linhas) para cada janela.
                       Com dados semanais: 52 obs = 1 ano, 26 obs = 6 meses
                       Padrão: 52 (1 ano)
            passo_obs: frequência da análise em observações.
                       1=toda semana, 4=mensal, 13=trimestral
                       Padrão: 1 (análise semanal)
            verbose: Se True, imprime mensagens de diagnóstico
        """
        self.janela_obs = janela_obs
        self.passo_obs = passo_obs
        self.verbose = verbose
        self.historico_quadrantes = []
    
    def carregar_dados_completos(self):
        """Carrega todos os dados históricos."""
        # Caminho relativo à pasta raiz do projeto
        caminho_dados = Path(__file__).parent.parent / 'data_prices.csv'
        self.data_prices = pd.read_csv(caminho_dados, index_col=0, parse_dates=True)
        print(f"✓ Dados carregados: {len(self.data_prices)} observações (semanas)")
        print(f"✓ Período: {self.data_prices.index[0]} a {self.data_prices.index[-1]}")
        print(f"✓ Frequência: ~{len(self.data_prices) * 7 / 365:.1f} anos de dados")
    
    def analisar_periodo(self, idx_fim):
        """
        Analisa um período específico (últimas N observações até idx_fim).
        
        Args:
            idx_fim: índice (posição) da última observação da janela
        
        Returns:
            dict com quadrante, scores e métricas
        """
        try:
            # Pegar últimas N observações até idx_fim
            idx_inicio = max(0, idx_fim - self.janela_obs)
            dados_janela = self.data_prices.iloc[idx_inicio:idx_fim]
            
            # Mínimo: 52 obs para momentum completo (12m), mas pelo menos 4 obs (1m)
            if len(dados_janela) < 4:
                if self.verbose:
                    data_fim = self.data_prices.index[idx_fim-1]
                    print(f"⚠️  Janela {data_fim.strftime('%Y-%m-%d')}: apenas {len(dados_janela)} obs (< 4)")
                return None
            
            # Salvar temporariamente na pasta raiz
            caminho_temp = Path(__file__).parent.parent / 'temp_window.csv'
            dados_janela.to_csv(caminho_temp)
            
            # Rodar momentum na janela
            analisador = AnalisadorMomentum(str(caminho_temp), verbose=False)
            dic_r_ativos = analisador.executar_analise_completa()
            
            # Verificar se dic_r_ativos é válido
            if not dic_r_ativos or len(dic_r_ativos) == 0:
                if self.verbose:
                    data_fim = self.data_prices.index[idx_fim-1]
                    print(f"⚠️  Janela {data_fim.strftime('%Y-%m-%d')}: dic_r_ativos vazio")
                return None
            
            # Classificar quadrante
            classificador = ClassificadorQuadrantes()
            resultado = classificador.analisar(dic_r_ativos)
            
            # Adicionar data (usar última data da janela)
            resultado['data'] = self.data_prices.index[idx_fim-1]
            
            return resultado
            
        except Exception as e:
            if self.verbose:
                data_fim = self.data_prices.index[idx_fim-1]
                print(f"❌ Erro ao analisar {data_fim.strftime('%Y-%m-%d')}: {str(e)}")
            return None
    
    def analisar_historico_completo(self):
        """
        Analisa todos os períodos históricos com step de passo_obs.
        """
        print(f"\n🔄 Iniciando análise histórica...")
        print(f"   Janela: {self.janela_obs} observações (~{self.janela_obs} semanas = {self.janela_obs/52:.1f} anos)")
        print(f"   Passo: {self.passo_obs} observações")
        
        # Índices para analisar (após janela inicial)
        # range(início, fim, passo)
        indices = range(self.janela_obs, len(self.data_prices), self.passo_obs)
        
        print(f"   Total de análises: {len(list(indices))}\n")
        
        for i, idx in enumerate(indices): #O que esse enumerate faz?
            resultado = self.analisar_periodo(idx)
            if resultado:
                self.historico_quadrantes.append(resultado)
                
                if (i + 1) % 10 == 0:
                    print(f"   Processado: {i+1}/{len(list(indices))} períodos...")
        
        print(f"\n✓ Análise completa! {len(self.historico_quadrantes)} períodos analisados.\n")
        
        return pd.DataFrame(self.historico_quadrantes)
    
    def gerar_relatorio(self):
        """Gera relatório resumido."""
        df = pd.DataFrame(self.historico_quadrantes)
        
        # Verificar se há dados para gerar relatório
        if len(df) == 0:
            print("\n" + "="*70)
            print("⚠️  NENHUM PERÍODO FOI ANALISADO COM SUCESSO")
            print("="*70)
            print("\nPossíveis causas:")
            print("  1. Janela muito pequena (< 4 observações semanais)")
            print("  2. Dados insuficientes para momentum (precisa >= 4 obs para 1m)")
            print("  3. Erros nos dados (NaNs, dados faltantes)")
            print("\nSugestões:")
            print("  • Para momentum 1m: janela_obs >= 4 (mínimo)")
            print("  • Para momentum completo: janela_obs >= 52 (12 meses)")
            print("  • Verifique data_prices.csv se tem dados suficientes")
            print("  • Execute com verbose=True para ver detalhes dos erros")
            print("="*70 + "\n")
            return df
        
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
        janela_obs=52,   # 52 semanas = ~1 ano (bom para momentum 12m)
        passo_obs=1      # Análise a cada semana (rolling window)
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