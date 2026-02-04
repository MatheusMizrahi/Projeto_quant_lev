# 🔧 SOLUÇÃO: Pipeline Quebrada

## 🚨 PROBLEMA IDENTIFICADO

O backtest usa `historico_intensidade_12_simples_v3.csv`, mas quando você:
1. Muda os pesos em `Definicao_quadrante_3_CALIBRADO.py`
2. Roda `analise_historica_4.py --versao v2` → gera `historico_quadrantes_v2.csv`
3. Roda `Analise_intensidade_5.py` → **PROBLEMA AQUI!**
4. Roda `backtest_6.py` → usa arquivo ANTIGO

**Analise_intensidade_5.py está hardcoded para usar `historico_quadrantes_v3.csv`!**

Mesmo mudando os pesos, o backtest continua usando os dados antigos.

---

## ✅ SOLUÇÃO COMPLETA

### OPÇÃO 1: Pipeline Manual (Rápido)

```bash
# 1. Gerar histórico V2 (com pesos novos)
python visualizacao_analise_historica/analise_historica_4.py --versao v2

# 2. Copiar para nome que Analise_intensidade_5.py espera
copy historico_quadrantes_v2.csv historico_quadrantes_v3.csv

# 3. Gerar intensidades (K-Means) 
python Analise_intensidade_5.py

# 4. Rodar backtest
python backtest_6.py
```

### OPÇÃO 2: Modificar Analise_intensidade_5.py (Correto)

Edite `Analise_intensidade_5.py` linha ~23:

```python
# ANTES (hardcoded):
def carregar_historico(path_csv='historico_quadrantes_v3.csv'):

# DEPOIS (aceita argumento):
def carregar_historico(path_csv='historico_quadrantes_v2.csv'):
```

E adicione argparse no final:
```python
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--versao', default='v2', choices=['v1','v2','v3'])
    args = parser.parse_args()
    
    # Modificar função main() para usar args.versao
    # ...
```

### OPÇÃO 3: Script Automático (Recomendado)

Execute `pipeline_completo.py` (vou criar agora) que:
- Pergunta qual versão testar
- Roda toda a cadeia automaticamente
- Garante consistência entre arquivos

---

## 🎯 TESTE DE VALIDAÇÃO

Depois de corrigir, teste se mudanças afetam resultado:

```bash
# Teste 1: V1 (pesos originais)
python pipeline_completo.py --versao v1
# Anote: Sharpe = X

# Teste 2: V2 (pesos calibrados)  
python pipeline_completo.py --versao v2
# Anote: Sharpe = Y

# Se X ≠ Y: SUCESSO! ✅
# Se X = Y: Problema persiste ❌
```

---

## 🔍 POR QUE ISSO ACONTECEU?

Os arquivos estão desconectados:

```
analise_historica_4.py (--versao v2) 
    ↓
historico_quadrantes_v2.csv  ← Você gera isso
    ↓
Analise_intensidade_5.py
    ↓
historico_intensidade_12_simples_v3.csv  ← Mas usa nome v3!
    ↓
backtest_6.py
```

**Solução**: Manter versionamento consistente em toda pipeline!

---

## 📋 PRÓXIMOS PASSOS

1. **Escolha Opção 1** (rápido para testar agora)
2. Depois rode `python pipeline_completo.py` (vou criar)
3. Compare sharpe ratios V1 vs V2
4. Se ainda iguais: problema está na ALOCAÇÃO do backtest
