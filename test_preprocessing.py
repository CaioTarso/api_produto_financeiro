"""
Script para testar se o preprocessamento gera as 62 features corretas.
"""
import pandas as pd
import sys
sys.path.insert(0, '/home/luisz/projects/products')

from main import build_raw_from_simulacao, preprocess_dataframe_for_api, encoder_idade, encoder_renda
from schemas import SimulacaoInput

# Dados de teste
test_payload = SimulacaoInput(
    genero="H",
    idade=35,
    antiguidade=76,
    renda=120000.0,
    segmento="02 - PARTICULARES",
    provincia="MADRID",
    canal="KHL",
    produtos=["conta_corrente", "recebimento_recibos"]
)

print("=" * 80)
print("TESTE DE PREPROCESSAMENTO")
print("=" * 80)

# 1. Construir DataFrame raw
print("\n1. Construindo DataFrame raw...")
raw_df = build_raw_from_simulacao(test_payload)
print(f"   Shape: {raw_df.shape}")
print(f"   Colunas: {list(raw_df.columns)}")
print(f"\n   Primeiras linhas:")
print(raw_df.head())

# 2. Preprocessar
print("\n2. Preprocessando...")
try:
    processed_df = preprocess_dataframe_for_api(raw_df)
    print(f"   ✓ Sucesso!")
    print(f"   Shape: {processed_df.shape}")
    print(f"   Número de features: {len(processed_df.columns)}")
    
    # Verificar se tem as 62 features esperadas
    expected_count = 62
    if len(processed_df.columns) == expected_count:
        print(f"   ✓ Número correto de features ({expected_count})")
    else:
        print(f"   ✗ ERRO: Esperado {expected_count} features, obteve {len(processed_df.columns)}")
    
    # Listar todas as features
    print("\n   Features geradas:")
    for i, col in enumerate(processed_df.columns, 1):
        val = processed_df[col].iloc[0]
        print(f"   {i:2d}. {col:45s} = {val}")
    
    # Verificar tipos
    print("\n3. Verificando tipos de dados...")
    non_numeric = processed_df.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric:
        print(f"   ✗ AVISO: Colunas não numéricas encontradas: {non_numeric}")
    else:
        print(f"   ✓ Todas as colunas são numéricas")
    
    # Verificar NaNs
    print("\n4. Verificando valores ausentes...")
    nan_cols = processed_df.columns[processed_df.isna().any()].tolist()
    if nan_cols:
        print(f"   ✗ AVISO: Colunas com NaN: {nan_cols}")
    else:
        print(f"   ✓ Nenhum valor ausente")
    
    # Estatísticas
    print("\n5. Estatísticas das features:")
    print(processed_df.describe().T)
    
except Exception as e:
    print(f"   ✗ ERRO: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("FIM DO TESTE")
print("=" * 80)
