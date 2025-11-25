"""
Script para testar os modelos XGBoost diretamente e comparar com a API.
"""
import pandas as pd
import numpy as np
import joblib
import pickle
import xgboost as xgb

# Carregar um modelo
model_path = './models/xgboost_conta_corrente_new.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f"Modelo carregado: {model_path}")
print(f"Tipo do modelo: {type(model)}")

# Criar dados sintéticos de teste (simular um cliente TOP)
# Com 62 features (valores aleatórios como exemplo)
test_data = pd.DataFrame({
    'ativo_mes_passado': [1.0],
    'conta_corrente': [0],
    'conta_nominal': [0],
    'conta_terceiros': [0],
    'conta_eletronica': [0],
    'conta_cobranca_recibos': [0],
    'recebimento_recibos': [0],
    'conta_corrente_prev': [0],
    'conta_eletronica_prev': [0],
    'conta_nominal_prev': [0],
    'conta_terceiros_prev': [0],
    'recebimento_recibos_prev': [0],
    'renda_log': [12.2],  # log1p(200000)
    'renda_variacao': [0.0],
    'possui_pacote_basico': [0],
    'conta_corrente_idade_interacao': [0],
    'conta_eletronica_idade_interacao': [0],
    'conta_nominal_idade_interacao': [0],
    'conta_terceiros_idade_interacao': [0],
    'recebimento_recibos_idade_interacao': [0],
    'conta_corrente_mes_anterior': [0],
    'conta_eletronica_mes_anterior': [0],
    'conta_nominal_mes_anterior': [0],
    'conta_terceiros_mes_anterior': [0],
    'recebimento_recibos_mes_anterior': [0],
    'idade_categoria_encoded': [3.0],  # Meia_Idade
    'renda_categoria_encoded': [3.0],  # Alta
})

# Adicionar 35 features de Target Encoding com valores típicos
te_features = [
    'TE_canal_aquisicao_conta_corrente', 'TE_canal_aquisicao_conta_eletronica',
    'TE_canal_aquisicao_conta_nominal', 'TE_canal_aquisicao_conta_terceiros',
    'TE_canal_aquisicao_recebimento_recibos',
    'TE_codigo_provincia_conta_corrente', 'TE_codigo_provincia_conta_eletronica',
    'TE_codigo_provincia_conta_nominal', 'TE_codigo_provincia_conta_terceiros',
    'TE_codigo_provincia_recebimento_recibos',
    'TE_nome_provincia_conta_corrente', 'TE_nome_provincia_conta_eletronica',
    'TE_nome_provincia_conta_nominal', 'TE_nome_provincia_conta_terceiros',
    'TE_nome_provincia_recebimento_recibos',
    'TE_relacionamento_mes_conta_corrente', 'TE_relacionamento_mes_conta_eletronica',
    'TE_relacionamento_mes_conta_nominal', 'TE_relacionamento_mes_conta_terceiros',
    'TE_relacionamento_mes_recebimento_recibos',
    'TE_tipo_relacionamento_mes_conta_corrente', 'TE_tipo_relacionamento_mes_conta_eletronica',
    'TE_tipo_relacionamento_mes_conta_nominal', 'TE_tipo_relacionamento_mes_conta_terceiros',
    'TE_tipo_relacionamento_mes_recebimento_recibos',
    'TE_segmento_marketing_conta_corrente', 'TE_segmento_marketing_conta_eletronica',
    'TE_segmento_marketing_conta_nominal', 'TE_segmento_marketing_conta_terceiros',
    'TE_segmento_marketing_recebimento_recibos',
    'TE_segmento_idade_conta_corrente', 'TE_segmento_idade_conta_eletronica',
    'TE_segmento_idade_conta_nominal', 'TE_segmento_idade_conta_terceiros',
    'TE_segmento_idade_recebimento_recibos'
]

# Valores típicos de TE (baseado nos mappings reais)
for te_feat in te_features:
    test_data[te_feat] = [0.5]  # Valor médio aproximado

print(f"\nShape do test_data: {test_data.shape}")
print(f"Colunas: {list(test_data.columns)[:10]}...")

# Fazer predição
dmatrix = xgb.DMatrix(test_data)
proba = model.predict(dmatrix)

print(f"\n✅ Predição (prob): {proba[0]:.6f}")
print(f"✅ Predição (%): {proba[0]*100:.4f}%")

# Testar com TE valores mais altos (cenário otimista)
print("\n--- Teste com TE valores altos (0.8) ---")
test_data_high_te = test_data.copy()
for te_feat in te_features:
    test_data_high_te[te_feat] = 0.8

dmatrix_high = xgb.DMatrix(test_data_high_te)
proba_high = model.predict(dmatrix_high)
print(f"✅ Predição com TE alto: {proba_high[0]:.6f} ({proba_high[0]*100:.4f}%)")

# Testar com valores baixos (cenário pessimista)
print("\n--- Teste com TE valores baixos (0.02) ---")
test_data_low_te = test_data.copy()
for te_feat in te_features:
    test_data_low_te[te_feat] = 0.02

dmatrix_low = xgb.DMatrix(test_data_low_te)
proba_low = model.predict(dmatrix_low)
print(f"✅ Predição com TE baixo: {proba_low[0]:.6f} ({proba_low[0]*100:.4f}%)")
