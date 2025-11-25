"""
Script para exportar os dicionários de Target Encoding do notebook de treinamento.
Execute este código no NOTEBOOK após treinar os modelos.
"""
import joblib
import json
import pandas as pd

# =============================================================================
# PASSO 1: Certifique-se que você tem o DataFrame 'train' com as colunas TE_*
# =============================================================================

# Lista de colunas categóricas que receberam Target Encoding
te_categorical_cols = [
    "canal_aquisicao",
    "codigo_provincia", 
    "nome_provincia",
    "relacionamento_mes",
    "tipo_relacionamento_mes",
    "segmento_marketing",
    "segmento_idade"  # Se você criou essa feature
]

# Lista de produtos-alvo
products = [
    'conta_corrente',
    'conta_eletronica',
    'conta_nominal',
    'conta_terceiros',
    'recebimento_recibos'
]

# =============================================================================
# PASSO 2: Extrair os mappings de cada coluna TE
# =============================================================================

te_mappings = {}

for col in te_categorical_cols:
    te_mappings[col] = {}
    
    for product in products:
        te_feature_name = f"TE_{col}_{product}"
        
        # Verificar se a feature existe no train
        if te_feature_name in train.columns:
            # Criar dicionário: categoria -> valor médio de TE
            mapping = train.groupby(col)[te_feature_name].mean().to_dict()
            
            # Salvar média global como fallback para categorias não vistas
            global_mean = train[te_feature_name].mean()
            
            te_mappings[col][product] = {
                'mapping': mapping,
                'global_mean': global_mean
            }
            
            print(f"✓ Exportado: {te_feature_name} ({len(mapping)} categorias, média global={global_mean:.4f})")
        else:
            print(f"⚠ Feature {te_feature_name} não encontrada no train")

# =============================================================================
# PASSO 3: Salvar como pickle (recomendado) e JSON (backup)
# =============================================================================

# Salvar como pickle (preserva tipos de dados)
output_path_pkl = './models/te_mappings.pkl'
joblib.dump(te_mappings, output_path_pkl)
print(f"\n✓ Mappings salvos em: {output_path_pkl}")

# Salvar como JSON (para inspeção manual)
# Converter keys para string para JSON
te_mappings_json = {}
for col, products_dict in te_mappings.items():
    te_mappings_json[col] = {}
    for product, data in products_dict.items():
        te_mappings_json[col][product] = {
            'mapping': {str(k): float(v) for k, v in data['mapping'].items()},
            'global_mean': float(data['global_mean'])
        }

output_path_json = './models/te_mappings.json'
with open(output_path_json, 'w') as f:
    json.dump(te_mappings_json, f, indent=2)
print(f"✓ Mappings salvos em: {output_path_json}")

# =============================================================================
# PASSO 4: Estatísticas
# =============================================================================

print("\n" + "="*60)
print("RESUMO DOS MAPPINGS EXPORTADOS")
print("="*60)

total_mappings = 0
for col, products_dict in te_mappings.items():
    print(f"\n{col}:")
    for product, data in products_dict.items():
        n_categories = len(data['mapping'])
        total_mappings += n_categories
        print(f"  - {product}: {n_categories} categorias (média={data['global_mean']:.4f})")

print(f"\nTotal de mappings exportados: {total_mappings}")
print("\n✓ Pronto! Copie os arquivos para a pasta /models da API.")
