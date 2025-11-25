# 🔧 Integração dos Mappings de Target Encoding

## ❌ Problema Atual

As probabilidades estão extremamente baixas (~0.001) porque a API está usando **valores default (0.5)** para todas as 35 features de Target Encoding, em vez dos valores reais calculados durante o treinamento.

**Exemplo de saída atual:**
```json
{
  "ranking": [
    {"produto": "conta_terceiros", "prob": 0.0014},  // ❌ Deveria ser ~0.15-0.80
    {"produto": "conta_eletronica", "prob": 0.0013}  // ❌ Muito baixo
  ]
}
```

---

## ✅ Solução: Exportar Mappings do Notebook

### **PASSO 1: No Notebook de Treinamento**

Abra o notebook `product_recomendation (3).ipynb` e execute este código **DEPOIS** de treinar os modelos:

```python
# Copie e cole este código em uma nova célula do notebook
import joblib
import json

# Lista de colunas categóricas que receberam Target Encoding
te_categorical_cols = [
    "canal_aquisicao",
    "codigo_provincia", 
    "nome_provincia",
    "relacionamento_mes",
    "tipo_relacionamento_mes",
    "segmento_marketing",
    "segmento_idade"
]

# Lista de produtos-alvo
products = [
    'conta_corrente',
    'conta_eletronica',
    'conta_nominal',
    'conta_terceiros',
    'recebimento_recibos'
]

# Extrair os mappings
te_mappings = {}

for col in te_categorical_cols:
    te_mappings[col] = {}
    
    for product in products:
        te_feature_name = f"TE_{col}_{product}"
        
        if te_feature_name in train.columns:
            # Criar dicionário: categoria -> valor médio de TE
            mapping = train.groupby(col)[te_feature_name].mean().to_dict()
            global_mean = train[te_feature_name].mean()
            
            te_mappings[col][product] = {
                'mapping': mapping,
                'global_mean': global_mean
            }
            
            print(f"✓ {te_feature_name} ({len(mapping)} categorias)")

# Salvar como pickle
joblib.dump(te_mappings, './te_mappings.pkl')
print("\n✓ Mappings salvos em: ./te_mappings.pkl")
```

### **PASSO 2: Copiar o Arquivo Gerado**

Copie o arquivo `te_mappings.pkl` do notebook para a pasta `/models` do projeto da API:

```bash
cp /caminho/do/notebook/te_mappings.pkl /home/luisz/projects/products/models/
```

### **PASSO 3: Reiniciar a API**

```bash
cd /home/luisz/projects/products
source .venv/bin/activate
pkill -f uvicorn  # Parar servidor atual
uvicorn main:app --reload  # Reiniciar
```

Você verá esta mensagem no console:
```
✓ Target Encoding mappings carregados de ./models/te_mappings.pkl
  Total de XXXX mappings carregados
```

---

## 🎯 Resultado Esperado

Após carregar os mappings, as probabilidades devem ficar realistas:

```json
{
  "ranking": [
    {"produto": "conta_terceiros", "prob": 0.6523},    // ✓ 65.23%
    {"produto": "conta_eletronica", "prob": 0.4891},   // ✓ 48.91%
    {"produto": "conta_nominal", "prob": 0.3254},      // ✓ 32.54%
    {"produto": "recebimento_recibos", "prob": 0.1876},// ✓ 18.76%
    {"produto": "conta_corrente", "prob": 0.0932}      // ✓ 9.32%
  ]
}
```

---

## 📝 Arquivos Criados

- ✅ `export_te_mappings.py` - Script standalone (alternativa)
- ✅ `main.py` - Atualizado para carregar mappings automaticamente
- ✅ `README_TE_INTEGRATION.md` - Este arquivo

---

## 🔍 Verificação

Teste a API após carregar os mappings:

```bash
curl -X POST http://localhost:8000/recomendar \
  -H "Content-Type: application/json" \
  -d '{
    "genero": "H",
    "idade": 35,
    "antiguidade": 76,
    "renda": 120000,
    "segmento": "02 - PARTICULARES",
    "provincia": "MADRID",
    "canal": "KHL",
    "produtos": ["conta_corrente"]
  }'
```

**Saída esperada:** Probabilidades entre 0.05 - 0.80 (5% - 80%)

---

## ⚠️ Notas Importantes

1. **Versão do XGBoost**: O warning sobre serialização é normal e não afeta a precisão
2. **Categorias não vistas**: Quando uma categoria não existe nos mappings, usa-se a média global
3. **Ordem das features**: Os modelos esperam exatamente 62 features na ordem correta (já implementado)
4. **Performance**: Com os mappings reais, as predições serão precisas

---

## 🆘 Troubleshooting

**Problema:** "TE mappings não encontrados"
- ✅ Verifique se `te_mappings.pkl` está em `/models`
- ✅ Execute o código no notebook primeiro

**Problema:** "KeyError ao aplicar mapping"
- ✅ Verifique os nomes das colunas no `train` DataFrame
- ✅ Ajuste `te_categorical_cols` se necessário

**Problema:** Probabilidades ainda baixas
- ✅ Verifique se os mappings foram carregados (mensagem no console)
- ✅ Inspecione o arquivo: `pd.read_pickle('./models/te_mappings.pkl')`
