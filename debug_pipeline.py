"""
Debug Pipeline - Verifica se mudanças estão sendo aplicadas
===========================================================

Testa toda a cadeia: Pesos → Classificador → Histórico → Backtest

OBJETIVO: Identificar onde o problema está ocorrendo
"""

import pandas as pd
import numpy as np
from Definicao_quadrante_3_CALIBRADO import ClassificadorQuadrantesCalibrado
from Definicao_quadrante_3 import ClassificadorQuadrantes


def teste_1_verificar_pesos():
    """Teste 1: Os pesos estão corretos no código?"""
    print("\n" + "="*70)
    print("TESTE 1: Verificação dos Pesos no Código")
    print("="*70)
    
    classificador = ClassificadorQuadrantesCalibrado()
    
    print("\n📊 PESOS DE INFLAÇÃO:")
    for ativo, peso in classificador.PESOS_INFLACAO.items():
        print(f"   {ativo:15} = {peso:+.2f}")
    
    print("\n📊 PESOS DE ATIVIDADE:")
    for ativo, peso in classificador.PESOS_ATIVIDADE.items():
        print(f"   {ativo:15} = {peso:+.2f}")
    
    # Verificar DXY especificamente
    dxy_atividade = classificador.PESOS_ATIVIDADE.get('DXY', None)
    
    print("\n🎯 VERIFICAÇÃO CRÍTICA:")
    if dxy_atividade is None:
        print("   ❌ DXY não encontrado em PESOS_ATIVIDADE!")
    elif dxy_atividade > 0:
        print(f"   ❌ DXY em ATIVIDADE = {dxy_atividade:+.2f} (POSITIVO - ERRADO!)")
        print("   💡 Deveria ser NEGATIVO (ex: -0.05 ou -0.10)")
        return False
    else:
        print(f"   ✅ DXY em ATIVIDADE = {dxy_atividade:+.2f} (negativo - correto)")
        return True


def teste_2_comparar_classificadores():
    """Teste 2: V1 e V2 dão resultados diferentes?"""
    print("\n" + "="*70)
    print("TESTE 2: Comparação V1 vs V2 (Score Mock)")
    print("="*70)
    
    # Criar scores mockados para teste
    dic_r_ativos = {
        'Oil_WTI': {'score': 0.5},
        'Gold': {'score': 0.3},
        'US_10Y': {'score': 0.2},
        'SP500': {'score': 0.6},
        'MSCI_EM': {'score': 0.4},
        'HighYield_ETF': {'score': 0.3},
        'DXY': {'score': 0.5}  # DXY alto (dólar forte)
    }
    
    # V1 (Original)
    v1 = ClassificadorQuadrantes()
    ativ_v1, infl_v1 = v1.calcular_proxies(dic_r_ativos)
    
    # V2 (Calibrado)
    v2 = ClassificadorQuadrantesCalibrado(usar_percentis=False)
    ativ_v2, infl_v2 = v2.calcular_proxies(dic_r_ativos)
    
    print("\n📊 SCORES COM DXY = 0.5 (Dólar Forte):")
    print(f"   V1 - Inflação:  {infl_v1:.4f}")
    print(f"   V1 - Atividade: {ativ_v1:.4f}")
    print(f"\n   V2 - Inflação:  {infl_v2:.4f}")
    print(f"   V2 - Atividade: {ativ_v2:.4f}")
    
    print(f"\n🔍 DIFERENÇAS:")
    diff_infl = abs(infl_v1 - infl_v2)
    diff_ativ = abs(ativ_v1 - ativ_v2)
    print(f"   Δ Inflação:  {diff_infl:.4f}")
    print(f"   Δ Atividade: {diff_ativ:.4f}")
    
    if diff_infl < 0.01 and diff_ativ < 0.01:
        print("\n   ⚠️  DIFERENÇAS MUITO PEQUENAS (< 0.01)")
        print("   💡 Pesos são quase idênticos → resultados iguais esperados")
        return False
    else:
        print("\n   ✅ Diferenças significativas → classificadores distintos")
        return True


def teste_3_verificar_historicos():
    """Teste 3: Os históricos V1 e V2 são diferentes?"""
    print("\n" + "="*70)
    print("TESTE 3: Comparação dos Históricos Gerados")
    print("="*70)
    
    try:
        h_v1 = pd.read_csv('historico_quadrantes_v1.csv', parse_dates=['data'])
        h_v2 = pd.read_csv('historico_quadrantes_v2.csv', parse_dates=['data'])
        
        print(f"\n✓ V1: {len(h_v1)} períodos")
        print(f"✓ V2: {len(h_v2)} períodos")
        
        # Comparar scores
        if len(h_v1) == len(h_v2):
            diff_infl = (h_v1['inflacao_score'] - h_v2['inflacao_score']).abs().mean()
            diff_ativ = (h_v1['atividade_score'] - h_v2['atividade_score']).abs().mean()
            
            print(f"\n📊 DIFERENÇA MÉDIA DOS SCORES:")
            print(f"   Inflação:  {diff_infl:.4f}")
            print(f"   Atividade: {diff_ativ:.4f}")
            
            if diff_infl < 0.01 and diff_ativ < 0.01:
                print("\n   ⚠️  SCORES QUASE IDÊNTICOS!")
                print("   💡 Classificadores estão gerando mesmos resultados")
                return False
            else:
                print("\n   ✅ Scores significativamente diferentes")
                
            # Comparar distribuição de quadrantes
            print(f"\n📊 DISTRIBUIÇÃO DE QUADRANTES:")
            
            def extrair_q(s):
                if 'Q1' in s: return 'Q1'
                elif 'Q2' in s: return 'Q2'
                elif 'Q3' in s: return 'Q3'
                else: return 'Q4'
            
            h_v1['Q'] = h_v1['quadrante'].apply(extrair_q)
            h_v2['Q'] = h_v2['quadrante'].apply(extrair_q)
            
            dist_v1 = h_v1['Q'].value_counts(normalize=True).sort_index()
            dist_v2 = h_v2['Q'].value_counts(normalize=True).sort_index()
            
            print("\n   V1:")
            for q, pct in dist_v1.items():
                print(f"      {q}: {pct*100:.1f}%")
            
            print("\n   V2:")
            for q, pct in dist_v2.items():
                print(f"      {q}: {pct*100:.1f}%")
            
            # Verificar se são idênticos
            quadrantes_iguais = (h_v1['Q'] == h_v2['Q']).sum()
            pct_iguais = quadrantes_iguais / len(h_v1) * 100
            
            print(f"\n   Períodos com MESMO quadrante: {quadrantes_iguais}/{len(h_v1)} ({pct_iguais:.1f}%)")
            
            if pct_iguais > 95:
                print("   ⚠️  CLASSIFICAÇÕES QUASE IDÊNTICAS (>95%)")
                print("   💡 Pesos muito similares → mesmo comportamento")
                return False
            else:
                print("   ✅ Classificações diferentes")
                return True
        else:
            print("\n   ⚠️ Tamanhos diferentes - não é possível comparar")
            return False
            
    except FileNotFoundError as e:
        print(f"\n   ❌ Arquivo não encontrado: {e}")
        print("   💡 Execute primeiro:")
        print("      python visualizacao_analise_historica/analise_historica_4.py --versao v1")
        print("      python visualizacao_analise_historica/analise_historica_4.py --versao v2")
        return None


def teste_4_verificar_backtest():
    """Teste 4: Backtest está usando arquivo correto?"""
    print("\n" + "="*70)
    print("TESTE 4: Verificação do Backtest")
    print("="*70)
    
    # Ler backtest_6.py para ver qual arquivo está sendo usado
    with open('backtest_6.py', 'r', encoding='utf-8') as f:
        conteudo = f.read()
    
    # Procurar arquivo de regimes
    import re
    match = re.search(r'arquivo_regimes\s*=\s*["\']([^"\']+)["\']', conteudo)
    
    if match:
        arquivo = match.group(1)
        print(f"\n📄 Arquivo de regimes usado: '{arquivo}'")
        
        try:
            df = pd.read_csv(arquivo, parse_dates=['data'])
            print(f"   ✓ Arquivo existe: {len(df)} períodos")
            print(f"   ✓ Período: {df['data'].min()} até {df['data'].max()}")
            
            # Verificar se tem coluna de intensidade
            if 'intensidade_12' in df.columns:
                print(f"   ✓ Tem intensidade K-Means")
                dist_int = df['intensidade_12'].value_counts()
                print(f"\n   Distribuição de intensidades:")
                for nome, count in dist_int.items():
                    print(f"      {nome}: {count} ({count/len(df)*100:.1f}%)")
            
            return True
            
        except FileNotFoundError:
            print(f"   ❌ ARQUIVO NÃO EXISTE!")
            print(f"   💡 Execute: python Analise_intensidade_5.py")
            return False
    else:
        print("\n   ❌ Não encontrou 'arquivo_regimes' no backtest_6.py")
        return False


def teste_5_sensibilidade_alocacao():
    """Teste 5: Problema está na alocação por regime?"""
    print("\n" + "="*70)
    print("TESTE 5: Análise da Estratégia de Alocação")
    print("="*70)
    
    # Ler alocação do backtest_6.py
    with open('backtest_6.py', 'r', encoding='utf-8') as f:
        linhas = f.readlines()
    
    print("\n📊 ALOCAÇÃO POR REGIME (do backtest_6.py):")
    dentro_alocacao = False
    for linha in linhas:
        if 'ALOCACAO_POR_REGIME' in linha and '=' in linha and not linha.strip().startswith('#'):
            dentro_alocacao = True
        if dentro_alocacao:
            print(linha.rstrip())
            if '}' in linha and dentro_alocacao:
                dentro_alocacao = False
                break
    
    print("\n💡 ANÁLISE:")
    print("   Se a estratégia de alocação for muito agressiva (ex: 100% short),")
    print("   pequenas mudanças no classificador não farão diferença no resultado.")
    print("\n   Se todos os quadrantes têm alocação similar, o backtest será flat")
    print("   independente da classificação de regime.")


def main():
    """Executa todos os testes."""
    print("\n" + "="*70)
    print(" "*20 + "DEBUG PIPELINE COMPLETO")
    print("="*70)
    print("\nObjetivo: Identificar por que mudanças não afetam resultados")
    print("="*70)
    
    resultados = {}
    
    # Executar testes
    resultados['pesos'] = teste_1_verificar_pesos()
    resultados['classificadores'] = teste_2_comparar_classificadores()
    resultados['historicos'] = teste_3_verificar_historicos()
    resultados['backtest'] = teste_4_verificar_backtest()
    teste_5_sensibilidade_alocacao()
    
    # Diagnóstico final
    print("\n" + "="*70)
    print("🎯 DIAGNÓSTICO FINAL")
    print("="*70)
    
    if resultados['pesos'] == False:
        print("\n❌ PROBLEMA: DXY ainda está POSITIVO em ATIVIDADE")
        print("   SOLUÇÃO: Edite Definicao_quadrante_3_CALIBRADO.py")
        print("            Mude 'DXY': 0.05 para 'DXY': -0.05")
    
    elif resultados['classificadores'] == False:
        print("\n❌ PROBLEMA: Pesos V1 e V2 são MUITO SIMILARES")
        print("   SOLUÇÃO: Os pesos 'calibrados' não mudam o suficiente")
        print("            Considere mudanças mais drásticas:")
        print("            - Oil: 0.40 → 0.30 (reduzir commodity bias)")
        print("            - Gold: 0.25 → 0.40 (aumentar safe-haven)")
        print("            - DXY: garantir que seja -0.10 (não -0.05)")
    
    elif resultados['historicos'] == False:
        print("\n❌ PROBLEMA: Históricos V1 e V2 são IDÊNTICOS")
        print("   SOLUÇÃO: Classificador não está diferenciando regimes")
        print("            Verifique se os thresholds estão corretos")
        print("            Teste com thresholds mais separados:")
        print("            - V1: limiar_inflacao=0.5, limiar_atividade=0.3")
        print("            - V2: limiar_inflacao=0.0, limiar_atividade=0.0")
    
    elif resultados['backtest'] == False:
        print("\n❌ PROBLEMA: Arquivo de regimes não existe")
        print("   SOLUÇÃO: Execute a pipeline completa:")
        print("            1. python visualizacao_analise_historica/analise_historica_4.py --versao v2")
        print("            2. python Analise_intensidade_5.py")
        print("            3. python backtest_6.py")
    
    else:
        print("\n⚠️  PROBLEMA MAIS COMPLEXO:")
        print("   Todos os componentes parecem corretos individualmente,")
        print("   mas o resultado final é o mesmo.")
        print("\n   POSSIBILIDADES:")
        print("   1. Estratégia de alocação não é sensível aos regimes")
        print("   2. Custos de transação muito altos mascaram diferenças")
        print("   3. Look-ahead bias ou erro de timing")
        print("   4. Período de teste é curto demais")
        print("\n   SOLUÇÃO: Teste com:")
        print("   - Reduzir custos: custo_transacao=0.0005 (5 bps)")
        print("   - Alocação mais agressiva nos regimes")
        print("   - Período mais longo de análise")
    
    print("\n" + "="*70)
    print("✅ Debug completo!")


if __name__ == '__main__':
    main()
