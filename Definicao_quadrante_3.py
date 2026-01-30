
import pandas as pd
import numpy as np
from Regressoes_lineares_2 import AnalisadorMomentum


class ClassificadorQuadrantes:
    """
    Classifica o regime macroeconômico em 4 quadrantes baseado em proxies de
    Atividade Econômica (eixo Y) e Inflação (eixo X).
    """
    
    def __init__(self, limiar_inflacao=0.5, limiar_atividade=0.3):
        """
        Args:
            limiar_inflacao: threshold para separar inflação alta/baixa (padrão: 0.5)
            limiar_atividade: threshold para separar atividade alta/baixa (padrão: 0.3)
        """
        # TODO: Considerar usar percentis históricos ao invés de valores fixos
        self.limiar_inflacao = limiar_inflacao
        self.limiar_atividade = limiar_atividade
    
    def calcular_proxies(self, dic_r_ativos): #função que calcula o scores de atividade econômica e inflação
        """
        Calcula índices compostos de Inflação e Atividade Econômica.
        
        Returns:
            tuple: (atividade_score, inflacao_score)
        """
        # PROXY DE INFLAÇÃO (eixo X)
        # TODO: Testar outros pesos (ex: adicionar commodities diversas)
        inflacao_score = (
            dic_r_ativos['Oil_WTI']['score'] * 0.40 +      # Petróleo = inflação energética
            dic_r_ativos['Gold']['score'] * 0.30 +         # Ouro = expectativa inflação + safe haven
            dic_r_ativos['US_10Y']['score'] * 0.20 +       # Taxa longa = expectativa inflação
            dic_r_ativos['DXY']['score'] * -0.10           # Dólar forte = inflação baixa (inverso)
        )
        
        # PROXY DE ATIVIDADE ECONÔMICA (eixo Y)
        # TODO: Considerar adicionar peso negativo para DXY quando significativo
        atividade_score = (
            dic_r_ativos['SP500']['score'] * 0.35 +        # Crescimento USA (economia #1)
            dic_r_ativos['MSCI_EM']['score'] * 0.25 +      # Crescimento emergentes (motor global)
            dic_r_ativos['HighYield_ETF']['score'] * 0.25 + # Condições de crédito corporativo
            dic_r_ativos['US_10Y']['score'] * 0.10 +       # Expansão fiscal (taxa alta pode = crescimento)
            dic_r_ativos['DXY']['score'] * -0.05           # Dólar forte = freio em emergentes
        )
        
        return atividade_score, inflacao_score
    
    def identificar_quadrante(self, atividade, inflacao):
        """
        Classifica o regime em 4 quadrantes baseado nas coordenadas (inflacao, atividade).
        
        Quadrantes:
        Q1 (Goldilocks): Alta atividade + Baixa inflação
        Q2 (Reflação): Alta atividade + Alta inflação
        Q3 (Estagflação): Baixa atividade + Alta inflação
        Q4 (Deflação): Baixa atividade + Baixa inflação
        
        Args:
            atividade: score de atividade econômica (eixo Y)
            inflacao: score de inflação (eixo X)
            
        Returns:
            str: Nome do quadrante
        """
        # TODO: Adicionar intensidade do sinal (forte/fraco) usando K-Means
        if atividade > self.limiar_atividade:  # Atividade ALTA
            if inflacao < self.limiar_inflacao:
                return "Q1: GOLDILOCKS"
            else:
                return "Q2: REFLAÇÃO"
        else:  # Atividade BAIXA
            if inflacao >= self.limiar_inflacao:
                return "Q3: ESTAGFLAÇÃO"
            else:
                return "Q4: DEFLAÇÃO/CONTRAÇÃO"
    
    def analisar(self, dic_r_ativos): #Calcula proxies e identifica o quadrante
        """
        Executa análise completa: calcula proxies e identifica quadrante.
        
        Returns:
            dict: Resultados da análise
        """
        atividade, inflacao = self.calcular_proxies(dic_r_ativos)
        quadrante = self.identificar_quadrante(atividade, inflacao)
        
        return {
            'quadrante': quadrante,
            'coordenadas': (inflacao, atividade),
            'inflacao_score': float(inflacao),
            'atividade_score': float(atividade)
        }


def main():
    """Executa classificação e exibe resultados."""
    # Executar análise de momentum primeiro
    analisador = AnalisadorMomentum(verbose=False)
    dic_r_ativos = analisador.executar_analise_completa()
    
    classificador = ClassificadorQuadrantes()
    resultado = classificador.analisar(dic_r_ativos)
    
    print("\n" + "="*60)
    print(" "*15 + "ANÁLISE DE REGIME MACROECONÔMICO")
    print("="*60)
    print(f"\n📊 Regime Identificado: {resultado['quadrante']}")
    print(f"\n📈 Coordenadas (Inflação, Atividade): ({resultado['inflacao_score']:.3f}, {resultado['atividade_score']:.3f})")
    print(f"   • Inflação Score: {resultado['inflacao_score']:.3f}")
    print(f"   • Atividade Score: {resultado['atividade_score']:.3f}")
    print("="*60 + "\n")
    
    return resultado


if __name__ == "__main__":
    resultado = main()