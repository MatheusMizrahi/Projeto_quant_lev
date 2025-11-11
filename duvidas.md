# 🤔 Dúvidas e Questões Metodológicas

## 📋 Lista de Dúvidas

### 1. Determinação dos Limiares de Classificação
**Pergunta:** Como determinar os limiares ótimos de inflação e atividade econômica para separar os quadrantes?

**Opções consideradas:**
- Valores fixos (ex: 0.5 para inflação, 0.3 para atividade)
- Percentis históricos (ex: mediana = 50º percentil)
- Otimização via backtesting

**Status:** ⚠️ Usando valores fixos inicialmente, migrar para percentis históricos

---

### 2. Calibração dos Pesos das Proxies
**Pergunta:** Como determinar os pesos de cada ativo para calcular os scores compostos de inflação e atividade econômica?

**Fórmulas atuais:**
```python
# Inflação
Oil_WTI (40%) + Gold (30%) + US_10Y (20%) - DXY (10%)

# Atividade
SP500 (35%) + MSCI_EM (25%) + HYG (25%) + US_10Y (10%) - DXY (5%)
```

**Métodos de calibração:**
- Literatura acadêmica e prática de mercado
- Análise de componentes principais (PCA)
- Otimização via backtesting de performance
- Grid search de combinações de pesos

**Status:** ⚠️ Usando pesos baseados em literatura, validar com backtesting

---

### 3. Validação das Regressões Lineares
**Pergunta:** Como interpretar e validar as regressões lineares? Qual o papel do p-value e como relacionar diferentes ativos?

**Aspectos a verificar:**
- **P-value < 0.05**: Significância estatística da tendência
- **R²**: Qualidade do ajuste (0-1, quanto maior melhor)
- **Beta (β₁)**: Direção e magnitude da tendência
- **Score**: Métrica composta = sinal(β₁) × √R²

**Relações entre ativos:**
- Cada ativo é regredido **independentemente** vs. tempo
- Não há regressão de um ativo contra outro
- As relações emergem através das proxies compostas

**Testes de robustez:**
- [ ] Verificar multicolinearidade entre ativos
- [ ] Testar diferentes janelas temporais (30, 60, 90 dias)
- [ ] Validar estabilidade dos coeficientes ao longo do tempo
- [ ] Comparar com benchmarks (rolling sharpe, momentum simples)

**Status:** ⚠️ Regressões implementadas, falta validação de robustez