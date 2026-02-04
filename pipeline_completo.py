"""
Pipeline Completo Automatizado
==============================

Roda toda a cadeia de forma consistente:
1. Análise histórica (classificação de regimes)
2. K-Means (intensidades)
3. Backtest

Garante que arquivos usam mesma versão.

AUTOR: Matheus Mizrahi
DATA: Fevereiro 2026
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path


def executar_comando(cmd, descricao):
    """Executa comando e mostra progresso."""
    print(f"\n{'='*70}")
    print(f"🔄 {descricao}")
    print(f"{'='*70}")
    print(f"Comando: {cmd}\n")
    
    resultado = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if resultado.returncode != 0:
        print(f"\n❌ ERRO ao executar: {descricao}")
        return False
    else:
        print(f"\n✅ Concluído: {descricao}")
        return True


def main(versao='v2'):
    """Executa pipeline completa."""
    
    print("\n" + "="*70)
    print(" "*20 + "PIPELINE COMPLETO AUTOMATIZADA")
    print("="*70)
    print(f"\n🎯 Versão selecionada: {versao}")
    print("\nEsta pipeline irá:")
    print("  1. Gerar histórico de regimes (classificação)")
    print("  2. Calcular intensidades com K-Means")
    print("  3. Executar backtest completo")
    print("="*70)
    
    # Passo 1: Análise histórica
    if not executar_comando(
        f"python visualizacao_analise_historica/analise_historica_4.py --versao {versao}",
        f"PASSO 1/4: Classificação de Regimes ({versao})"
    ):
        return False
    
    # Verificar se arquivo foi gerado
    arquivo_historico = f'historico_quadrantes_{versao}.csv'
    if not Path(arquivo_historico).exists():
        print(f"\n❌ ERRO: Arquivo '{arquivo_historico}' não foi gerado!")
        return False
    
    print(f"\n✅ Arquivo gerado: {arquivo_historico}")
    
    # Passo 2: Copiar/renomear para nome que Analise_intensidade_5.py espera
    # (temporariamente, até consertarmos o script)
    print(f"\n🔄 Preparando arquivo para K-Means...")
    shutil.copy(arquivo_historico, 'historico_quadrantes_v3.csv')
    print(f"   Copiado: {arquivo_historico} → historico_quadrantes_v3.csv")
    
    # Passo 3: K-Means (intensidades)
    if not executar_comando(
        "python Analise_intensidade_5.py",
        "PASSO 2/4: Cálculo de Intensidades (K-Means)"
    ):
        return False
    
    # Verificar se arquivo foi gerado
    if not Path('historico_intensidade_12_simples_v3.csv').exists():
        print("\n❌ ERRO: 'historico_intensidade_12_simples_v3.csv' não foi gerado!")
        return False
    
    # Passo 4: Backtest
    if not executar_comando(
        "python backtest_6.py",
        "PASSO 3/4: Execução do Backtest"
    ):
        return False
    
    # Passo 5: Resumo
    print("\n" + "="*70)
    print("📊 RESUMO DOS RESULTADOS")
    print("="*70)
    
    print(f"\n✅ Pipeline completa executada com versão: {versao}")
    print("\n📁 Arquivos gerados:")
    print(f"   • {arquivo_historico}")
    print(f"   • historico_intensidade_12_simples_v3.csv")
    print(f"   • backtest_detalhado.csv")
    print(f"   • (gráficos e relatórios)")
    
    print("\n" + "="*70)
    print("💡 PRÓXIMOS PASSOS")
    print("="*70)
    print("""
1. Verifique o Sharpe Ratio no output do backtest
2. Compare com versões anteriores
3. Se quiser testar outra versão:
   python pipeline_completo.py --versao v1
   
4. Para comparar v1 vs v2:
   a) python pipeline_completo.py --versao v1 (anote Sharpe)
   b) python pipeline_completo.py --versao v2 (anote Sharpe)
   c) Compare os valores
    """)
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pipeline Completa: Classificação → K-Means → Backtest'
    )
    parser.add_argument(
        '--versao', 
        type=str, 
        default='v2',
        choices=['v1', 'v2', 'v3'],
        help='Versão do classificador (v1=Original, v2=Calibrado, v3=Calibrado+Percentis)'
    )
    
    args = parser.parse_args()
    
    sucesso = main(versao=args.versao)
    
    if sucesso:
        print("\n✅ Pipeline executada com SUCESSO!\n")
        sys.exit(0)
    else:
        print("\n❌ Pipeline FALHOU em algum ponto.\n")
        sys.exit(1)
