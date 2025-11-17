from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Session, select
from database import engine, get_session
from models import Simulacao # Seu modelo de banco de dados
from schemas import SimulacaoInput

import json
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import date as dt_date, date
import os

app = FastAPI()

# Middleware CORS para aceitar requisições de qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, substitua por lista restrita
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def create_db():
    SQLModel.metadata.create_all(engine)


# --- 1. Carregar Modelos e Parâmetros de Pré-processamento --- 

# Define o diretório onde os modelos e encoders foram salvos
MODELS_DIR = './ml_artifacts' # OU '.' se estiverem na mesma pasta do main.py

# Carregar encoders Ordinal
loaded_encoder_idade = None
loaded_encoder_renda = None
encoders_path = os.path.join(MODELS_DIR, 'preprocessing_encoders.joblib')
try:
    loaded_encoders = joblib.load(encoders_path)
    loaded_encoder_idade = loaded_encoders['encoder_idade']
    loaded_encoder_renda = loaded_encoders['encoder_renda']
    print("Encoders de idade e renda carregados com sucesso.")
except FileNotFoundError:
    print(f"ERRO: Arquivo de encoders não encontrado em {encoders_path}. Certifique-se de que o caminho está correto.")
    print("A API pode não funcionar corretamente sem os encoders.")
except Exception as e:
    print(f"ERRO ao carregar encoders: {e}")


# Definir os top 5 produtos (nomes do target)
top_5_products = [
    'conta_corrente_new', 'recebimento_recibos_new', 'conta_terceiros_new',
    'conta_nominal_new', 'conta_eletronica_new'
]

# Carregar os modelos XGBoost
top_5_xgb_models = {}
for product_name in top_5_products:
    model_path = os.path.join(MODELS_DIR, f'xgboost_{product_name}.json')
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        top_5_xgb_models[product_name] = model
        print(f"Modelo XGBoost para '{product_name}' carregado.")
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o modelo para '{product_name}' de {model_path}. Erro: {e}")
        print("A API pode não conseguir fazer previsões para este produto.")

# --- Parâmetros de Pré-processamento Derivados do Treino ---
# Estas variáveis são cruciais e DEVERIAM SER EXPORTADAS E CARREGADAS.
# Para simplificar aqui, elas são replicadas, mas em produção, use joblib.dump/load.

median_renda_by_segmento_api = pd.Series({
    'universitario': 60000.0,
    'particulares': 120000.0,
    'top': 250000.0,
    'Desconhecido': 90000.0
})

bins_age_train_api = [0, 25, 35, 45, 60, 100]
labels_age_train_api = ['Jovem', 'Jovem_Adulto', 'Adulto', 'Meia_Idade', 'Idoso']

bins_antiguedad_train_api = [0, 6, 12, 36, 60, 256.0] # Max de antiguidade do treino
labels_antiguedad_train_api = ['0-6m', '6-12m', '1-3y', '3-5y', '>5y']

bins_renda_train_api = [9452.97, 76049.52, 103219.62, 139311.675, 8807813.86] # Quartis do treino
labels_renda_train_api = ['Baixa', 'Media_Baixa', 'Media_Alta', 'Alta']

bins_renda_faixas_train_api = [9452.97, 85000.0, 150000.0, 8807813.86] # Quartis customizados
labels_renda_faixas_train_api = ['baixa', 'média', 'alta']

produtos_all_api = [
    "conta_corrente", "cartao_credito", "plano_pensao", "recebimento_recibos",
    "conta_nominal", "conta_maior_idade", "conta_terceiros", "conta_particular",
    "deposito_prazo_longo", "conta_eletronica", "fundo_investimento",
    "valores_mobiliarios", "deposito_salario", "deposito_pensao"
]

colunas_descartar_api = [
    "empregado_banco", "garantia_aval", "tipo_endereco",
    "ultima_data_cliente_trimestre", "falecido"
]

colunas_numericas_comuns_api = ["idade", "renda_estimativa", "antiguidade_meses"]

categoricas_api = ["sexo", "pais_residencia", "segmento_marketing", "canal_aquisicao",
                   "tipo_relacionamento", "tipo_relacionamento_mes"]

colunas_numericas_produtos_full_api = [
    "conta_corrente", "conta_poupanca", "cartao_credito", "deposito_prazo",
    "plano_pensao", "emprestimo_pessoal", "credito_habitacao", "recebimento_recibos",
    "conta_nominal", "conta_jovem", "conta_maior_idade", "conta_terceiros",
    "conta_particular", "fundo_investimento_corporativo", "deposito_mercado_monetario",
    "deposito_prazo_longo", "conta_eletronica", "fundo_investimento", "hipoteca",
    "valores_mobiliarios", "deposito_salario", "deposito_pensao", "recebimento_recibos"
]

translate_columns_api = {
    "fecha_dato": "data_referencia", "ncodpers": "id_cliente", "ind_empleado": "tipo_empregado",
    "pais_residencia": "pais_residencia", "sexo": "sexo", "age": "idade",
    "fecha_alta": "data_entrada_banco", "ind_nuevo": "cliente_novo", "antiguedad": "antiguidade_meses",
    "indrel": "tipo_relacionamento", "ult_fec_cli_1t": "ultima_data_cliente_trimestre",
    "indrel_1mes": "relacionamento_mes", "tiprel_1mes": "tipo_relacionamento_mes",
    "indresi": "residente", "indext": "estrangeiro", "conyuemp": "empregado_banco",
    "canal_entrada": "canal_aquisicao", "indfall": "falecido", "tipodom": "tipo_endereco",
    "cod_prov": "codigo_provincia", "nomprov": "nome_provincia",
    "ind_actividad_cliente": "ativo_mes_passado", "renta": "renda_estimativa", "segmento": "segmento_marketing",
    "ind_ahor_fin_ult1": "conta_poupanca", "ind_aval_fin_ult1": "garantia_aval",
    "ind_cco_fin_ult1": "conta_corrente", "ind_cder_fin_ult1": "deposito_prazo",
    "ind_cno_fin_ult1": "conta_nominal", "ind_ctju_fin_ult1": "conta_jovem",
    "ind_ctma_fin_ult1": "conta_maior_idade", "ind_ctop_fin_ult1": "conta_terceiros",
    "ind_ctpp_fin_ult1": "conta_particular", "ind_deco_fin_ult1": "fundo_investimento_corporativo",
    "ind_deme_fin_ult1": "deposito_mercado_monetario", "ind_dela_fin_ult1": "deposito_prazo_longo",
    "ind_ecue_fin_ult1": "conta_eletronica", "ind_fond_fin_ult1": "fundo_investimento",
    "ind_hip_fin_ult1": "hipoteca", "ind_plan_fin_ult1": "plano_pensao",
    "ind_pres_fin_ult1": "emprestimo_pessoal", "ind_reca_fin_ult1": "recebimento_recibos",
    "ind_tjcr_fin_ult1": "cartao_credito", "ind_valo_fin_ult1": "valores_mobiliarios",
    "ind_viv_fin_ult1": "credito_habitacao", "ind_nomina_ult1": "deposito_salario",
    "ind_nom_pens_ult1": "deposito_pensao", "ind_recibo_ult1": "recebimento_recibos"
}

# Esta é a lista de colunas que seu X_train FINAL tinha.
# É CRÍTICA para garantir que o DataFrame de inferência tenha as mesmas colunas e ordem.
# Você DEVE CARREGAR isso de um arquivo salvo durante o treinamento (ex: joblib.dump(X_train.columns.tolist(), 'X_train_cols.joblib'))
X_train_columns = [
    'ativo_mes_passado', 'conta_corrente', 'cartao_credito', 'plano_pensao', 'recebimento_recibos', 'conta_nominal',
    'conta_maior_idade', 'conta_terceiros', 'conta_particular', 'deposito_prazo_longo', 'conta_eletronica',
    'fundo_investimento', 'valores_mobiliarios', 'deposito_salario', 'deposito_pensao',
    'renda_estimativa_faltante', 'tempo_desde_alta', 'renda_log', 'renda_variacao', 'possui_pacote_basico',
    'conta_corrente_idade_interacao', 'cartao_credito_idade_interacao', 'plano_pensao_idade_interacao',
    'recebimento_recibos_idade_interacao', 'conta_nominal_idade_interacao', 'conta_maior_idade_idade_interacao',
    'conta_terceiros_idade_interacao', 'conta_particular_idade_interacao', 'deposito_prazo_longo_idade_interacao',
    'conta_eletronica_idade_interacao', 'fundo_investimento_idade_interacao', 'valores_mobiliarios_idade_interacao',
    'deposito_salario_idade_interacao', 'deposito_pensao_idade_interacao',
    'conta_corrente_mes_anterior', 'cartao_credito_mes_anterior', 'plano_pensao_mes_anterior',
    'recebimento_recibos_mes_anterior', 'conta_nominal_mes_anterior', 'conta_maior_idade_mes_anterior',
    'conta_terceiros_mes_anterior', 'conta_particular_mes_anterior', 'deposito_prazo_longo_mes_anterior',
    'conta_eletronica_mes_anterior', 'fundo_investimento_mes_anterior', 'valores_mobiliarios_mes_anterior',
    'deposito_salario_mes_anterior', 'deposito_pensao_mes_anterior',
    'idade_categoria_encoded', 'renda_categoria_encoded',
    'TE_canal_aquisicao_conta_corrente', 'TE_canal_aquisicao_recebimento_recibos',
    'TE_canal_aquisicao_conta_terceiros', 'TE_canal_aquisicao_conta_nominal', 'TE_canal_aquisicao_conta_eletronica',
    'TE_codigo_provincia_conta_corrente', 'TE_codigo_provincia_recebimento_recibos',
    'TE_codigo_provincia_conta_terceiros', 'TE_codigo_provincia_conta_nominal', 'TE_codigo_provincia_conta_eletronica',
    'TE_nome_provincia_conta_corrente', 'TE_nome_provincia_recebimento_recibos',
    'TE_nome_provincia_conta_terceiros', 'TE_nome_provincia_conta_nominal', 'TE_nome_provincia_conta_eletronica',
    'TE_relacionamento_mes_conta_corrente', 'TE_relacionamento_mes_recebimento_recibos',
    'TE_relacionamento_mes_conta_terceiros', 'TE_relacionamento_mes_conta_nominal', 'TE_relacionamento_mes_conta_eletronica',
    'TE_tipo_relacionamento_mes_conta_corrente', 'TE_tipo_relacionamento_mes_recebimento_recibos',
    'TE_tipo_relacionamento_mes_conta_terceiros', 'TE_tipo_relacionamento_mes_conta_nominal', 'TE_tipo_relacionamento_mes_conta_eletronica',
    'TE_segmento_marketing_conta_corrente', 'TE_segmento_marketing_recebimento_recibos',
    'TE_segmento_marketing_conta_terceiros', 'TE_segmento_marketing_conta_nominal', 'TE_segmento_marketing_conta_eletronica',
    'TE_segmento_idade_conta_corrente', 'TE_segmento_idade_recebimento_recibos',
    'TE_segmento_idade_conta_terceiros', 'TE_segmento_idade_conta_nominal', 'TE_segmento_idade_conta_eletronica'
] # ESTA LISTA DEVE SER OBTIDA DINAMICAMENTE


# Carregar Target Encoding mappings (opcional, recomendado)
loaded_te_mappings = None
te_path = os.path.join(MODELS_DIR, 'te_mappings.joblib')
try:
    if os.path.exists(te_path):
        loaded_te_mappings = joblib.load(te_path)
        print("Target Encoding mappings carregados com sucesso.")
    else:
        print(f"AVISO: TE mappings não encontrados em {te_path}. Usando 0.0 como fallback.")
except Exception as e:
    print(f"ERRO ao carregar TE mappings: {e}")


# --- 2. Função de Pré-processamento para a API ---

def preprocess_dataframe_for_api(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_processed = df_raw.copy()

    # Aplicar renomeações (conforme translate_columns_api)
    df_processed.rename(columns=translate_columns_api, inplace=True)

    # 1. Remover Colunas Descartáveis
    df_processed = df_processed.drop(columns=colunas_descartar_api, errors="ignore")

    # 2. Substituir valores de sexo
    if 'sexo' in df_processed.columns:
        df_processed["sexo"] = df_processed["sexo"].replace({"H": "homem", "V": "mulher"})

    # 3. Converter colunas numéricas comuns
    for col in colunas_numericas_comuns_api:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")

    # 4. Criar flag para renda faltante e preencher NaNs na renda
    if 'renda_estimativa' in df_processed.columns and 'segmento_marketing' in df_processed.columns:
        df_processed['renda_estimativa_faltante'] = df_processed['renda_estimativa'].isna().astype(int)
        df_processed["renda_estimativa"] = df_processed.apply(
            lambda row: row["renda_estimativa"] if pd.notna(row["renda_estimativa"])
            else median_renda_by_segmento_api.get(row["segmento_marketing"], median_renda_by_segmento_api.median()), axis=1
        )
    elif 'renda_estimativa' in df_processed.columns:
        df_processed['renda_estimativa_faltante'] = df_processed['renda_estimativa'].isna().astype(int)
        df_processed["renda_estimativa"] = df_processed["renda_estimativa"].fillna(median_renda_by_segmento_api.median())

    # 5. Converter colunas de data
    if 'data_entrada_banco' in df_processed.columns:
        df_processed["data_entrada_banco"] = pd.to_datetime(df_processed["data_entrada_banco"], errors="coerce")
    if 'data_referencia' in df_processed.columns:
        df_processed["data_referencia"] = pd.to_datetime(df_processed["data_referencia"], errors="coerce")
    # 5.1 tempo_desde_alta em meses aproximados
    if 'data_entrada_banco' in df_processed.columns and 'data_referencia' in df_processed.columns:
        diff_days = (df_processed['data_referencia'] - df_processed['data_entrada_banco']).dt.days
        df_processed['tempo_desde_alta'] = (diff_days.fillna(0) / 30.0).astype(float)

    # 6. Preencher NaNs em colunas categóricas
    for col in categoricas_api:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna("Desconhecido")

    # 7. Garantir idade válida e dentro da faixa; evitar remoção do único registro
    if 'idade' in df_processed.columns:
        df_processed['idade'] = pd.to_numeric(df_processed['idade'], errors='coerce').fillna(35)
        df_processed['idade'] = df_processed['idade'].clip(lower=18, upper=95)

    # 8. Evitar remoção por NaN: preencher defaults mínimos para inferência única
    # codigo_provincia: 28 (MADRID), relacionamento_mes: 1.0, renda_estimativa já tratada acima
    if 'codigo_provincia' in df_processed.columns:
        df_processed['codigo_provincia'] = pd.to_numeric(df_processed['codigo_provincia'], errors='coerce').fillna(28.0)
    else:
        df_processed['codigo_provincia'] = 28.0
    if 'relacionamento_mes' in df_processed.columns:
        df_processed['relacionamento_mes'] = pd.to_numeric(df_processed['relacionamento_mes'], errors='coerce').fillna(1.0)
    else:
        df_processed['relacionamento_mes'] = 1.0

    # 9. Preencher NaNs em 'deposito_salario' e 'deposito_pensao'
    if 'deposito_salario' in df_processed.columns:
        df_processed["deposito_salario"] = df_processed["deposito_salario"].fillna(0)
    if 'deposito_pensao' in df_processed.columns:
        df_processed["deposito_pensao"] = df_processed["deposito_pensao"].fillna(0)

    # 10. Remover duplicatas
    df_processed = df_processed.drop_duplicates()

    # 11. Substituir valores em 'segmento_marketing'
    if 'segmento_marketing' in df_processed.columns:
        df_processed['segmento_marketing'] = df_processed['segmento_marketing'].replace({
            '03 - UNIVERSITARIO': 'universitario',
            '02 - PARTICULARES': 'particulares',
            '01 - TOP': 'top'
        })

    # 12. Remover produtos não relevantes
    if 'colunas_numericas_produtos_full_api' in globals() and 'produtos_all_api' in globals():
        current_product_cols_to_drop = [col for col in colunas_numericas_produtos_full_api if col not in produtos_all_api]
        df_processed = df_processed.drop(columns=current_product_cols_to_drop, errors="ignore")

    # 13. Criar categorias de idade, antiguidade e renda (com fallback para valores fora dos bins)
    if 'idade' in df_processed.columns and loaded_encoder_idade:
        df_processed['idade_categoria'] = pd.cut(
            df_processed['idade'], bins=bins_age_train_api, labels=labels_age_train_api, right=False
        )
        # Converter para string e substituir 'nan' por primeira faixa conhecida para evitar erro de categoria desconhecida
        df_processed['idade_categoria_str'] = df_processed['idade_categoria'].astype(str).replace('nan', labels_age_train_api[0])
        try:
            df_processed['idade_categoria_encoded'] = loaded_encoder_idade.transform(
                df_processed[['idade_categoria_str']]
            ).ravel()
        except Exception:
            # Se ainda falhar, aplica -1 como código desconhecido
            df_processed['idade_categoria_encoded'] = -1
        # Garantir tipo numérico
        if 'idade_categoria_encoded' in df_processed.columns:
            df_processed['idade_categoria_encoded'] = pd.to_numeric(df_processed['idade_categoria_encoded'], errors='coerce').fillna(-1)

    if 'antiguidade_meses' in df_processed.columns:
        df_processed['antiguedade_tempo'] = pd.cut(df_processed['antiguidade_meses'], bins=bins_antiguedad_train_api, labels=labels_antiguedad_train_api, right=False)

    if 'renda_estimativa' in df_processed.columns and loaded_encoder_renda:
        df_processed['renda_categoria'] = pd.cut(
            df_processed['renda_estimativa'], bins=bins_renda_train_api, labels=labels_renda_train_api,
            include_lowest=True, right=False, duplicates='drop'
        )
        df_processed['renda_categoria_str'] = df_processed['renda_categoria'].astype(str).replace('nan', labels_renda_train_api[0])
        try:
            df_processed['renda_categoria_encoded'] = loaded_encoder_renda.transform(
                df_processed[['renda_categoria_str']]
            ).ravel()
        except Exception:
            df_processed['renda_categoria_encoded'] = -1
        if 'renda_categoria_encoded' in df_processed.columns:
            df_processed['renda_categoria_encoded'] = pd.to_numeric(df_processed['renda_categoria_encoded'], errors='coerce').fillna(-1)

        df_processed['renda_log'] = np.log1p(df_processed['renda_estimativa'])
        # 'renda_variacao' para um único ponto no tempo será 0 ou NaN. Precisa de histórico.
        df_processed['renda_variacao'] = df_processed.groupby('id_cliente')['renda_estimativa'].pct_change().fillna(0)
        df_processed['renda_faixas'] = pd.cut(df_processed['renda_estimativa'], bins=bins_renda_faixas_train_api, labels=labels_renda_faixas_train_api, include_lowest=True, duplicates='drop')

    # Features de Interação e Co-ocorrência
    if 'conta_corrente' in df_processed.columns and 'recebimento_recibos' in df_processed.columns:
        df_processed['possui_pacote_basico'] = ((df_processed['conta_corrente'] == 1) & (df_processed['recebimento_recibos'] == 1)).astype(int)

    if 'segmento_marketing' in df_processed.columns and 'idade_categoria' in df_processed.columns:
        df_processed['segmento_idade'] = df_processed['segmento_marketing'].astype(str) + '_' + df_processed['idade_categoria'].astype(str)

    # Features de Posse de Produtos Anteriores.
    # Para inferência de um único cliente sem histórico na requisição, assumimos 0.
    for prod in produtos_all_api:
        if prod in df_processed.columns:
            df_processed[f'{prod}_mes_anterior'] = df_processed[prod]
        else:
            df_processed[f'{prod}_mes_anterior'] = 0

    # Target Encoding (IMPORTANTE: Mapeamentos TE devem ser carregados, não criados)
    # Aqui, estamos preenchendo com 0 como placeholder. Em produção, use um dicionário/DataFrame pré-calculado.
    categorical_for_encoding_api = [
        "canal_aquisicao", "codigo_provincia", "nome_provincia",
        "relacionamento_mes", "tipo_relacionamento_mes", "segmento_marketing", "segmento_idade"
    ]
    for col in categorical_for_encoding_api:
        cat_val = None
        if col in df_processed.columns and len(df_processed.index) > 0:
            cat_val = str(df_processed[col].iloc[0])
        for product_name_te in top_5_products:
            base_prod = product_name_te.replace('_new', '')
            te_feature_name = f"TE_{col}_{base_prod}"
            te_value = 0.0
            if loaded_te_mappings and isinstance(loaded_te_mappings, dict):
                te_col_map = loaded_te_mappings.get(col, {})
                te_prod_map = te_col_map.get(base_prod, {})
                te_value = float(te_prod_map.get(cat_val, te_prod_map.get('__global__', 0.0)))
            df_processed[te_feature_name] = te_value

    # Converter colunas 'object' ou 'category' para numérico para XGBoost (cat.codes)
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].astype('category').cat.codes
        elif isinstance(df_processed[col].dtype, pd.CategoricalDtype):
            df_processed[col] = df_processed[col].cat.codes

    df_processed = df_processed.fillna(0) # Preencher quaisquer NaNs restantes

    # Reindexar para ter exatamente as mesmas colunas que X_train e na mesma ordem
    # ESSENCIAL para a previsão correta!
    df_processed = df_processed.reindex(columns=X_train_columns, fill_value=0)

    return df_processed


# --- 3. Definir o Esquema de Entrada da API (CustomerFeatures) ---
# Esta classe Pydantic define o formato esperado para a entrada da API.
# Ela deve conter *todas as colunas originais* que o preprocessamento espera.
# Os nomes devem ser os originais do CSV, antes de qualquer renomeação.
class CustomerFeatures(BaseModel):
    fecha_dato: Optional[dt_date] = None
    ncodpers: Optional[int] = None
    ind_empleado: Optional[str] = None
    pais_residencia: Optional[str] = None
    sexo: Optional[str] = None
    age: Optional[int] = None
    fecha_alta: Optional[dt_date] = None
    ind_nuevo: Optional[int] = None
    antiguedad: Optional[float] = None
    indrel: Optional[float] = None
    ult_fec_cli_1t: Optional[str] = None
    indrel_1mes: Optional[float] = None
    tiprel_1mes: Optional[str] = None
    indresi: Optional[str] = None
    indext: Optional[str] = None
    conyuemp: Optional[str] = None
    canal_entrada: Optional[str] = None
    indfall: Optional[str] = None
    tipodom: Optional[int] = None
    cod_prov: Optional[float] = None
    nomprov: Optional[str] = None
    ind_actividad_cliente: Optional[float] = None
    renta: Optional[float] = None
    segmento: Optional[str] = None
    ind_ahor_fin_ult1: Optional[int] = None
    ind_aval_fin_ult1: Optional[int] = None
    ind_cco_fin_ult1: Optional[int] = None
    ind_cder_fin_ult1: Optional[int] = None
    ind_cno_fin_ult1: Optional[int] = None
    ind_ctju_fin_ult1: Optional[int] = None
    ind_ctma_fin_ult1: Optional[int] = None
    ind_ctop_fin_ult1: Optional[int] = None
    ind_ctpp_fin_ult1: Optional[int] = None
    ind_deco_fin_ult1: Optional[int] = None
    ind_deme_fin_ult1: Optional[int] = None
    ind_dela_fin_ult1: Optional[int] = None
    ind_ecue_fin_ult1: Optional[int] = None
    ind_fond_fin_ult1: Optional[int] = None
    ind_hip_fin_ult1: Optional[int] = None
    ind_plan_fin_ult1: Optional[int] = None
    ind_pres_fin_ult1: Optional[int] = None
    ind_reca_fin_ult1: Optional[int] = None
    ind_tjcr_fin_ult1: Optional[int] = None
    ind_valo_fin_ult1: Optional[int] = None
    ind_viv_fin_ult1: Optional[int] = None
    ind_nomina_ult1: Optional[int] = None
    ind_nom_pens_ult1: Optional[int] = None
    ind_recibo_ult1: Optional[int] = None


def _apply_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    # Valores padrão mínimos para permitir o pré-processamento
    defaults = {
        "fecha_dato": date.today().strftime("%Y-%m-%d"),
        "ncodpers": 1,
        "ind_empleado": "N",
        "pais_residencia": "ES",
        "sexo": "H",
        "age": 35,
        "fecha_alta": date.today().strftime("%Y-%m-%d"),
        "ind_nuevo": 0,
        "antiguedad": 12.0,
        "indrel": 1.0,
        "indrel_1mes": 1.0,
        "tiprel_1mes": "A",
        "indresi": "S",
        "indext": "N",
        "conyuemp": "N",
        "canal_entrada": "KHL",
        "indfall": "N",
        "tipodom": 1,
        "cod_prov": 28.0,
        "nomprov": "MADRID",
        "ind_actividad_cliente": 1.0,
        "renta": 120000.0,
        "segmento": "02 - PARTICULARES",
        # Indicadores de produtos (prefixo ind_*)
        "ind_cco_fin_ult1": 1,
        "ind_recibo_ult1": 1,
        "ind_nomina_ult1": 1
    }
    # Preenche faltantes
    for k, v in defaults.items():
        if data.get(k) in (None, ""):
            data[k] = v
    # Qualquer indicador não fornecido vira 0
    for k in list(data.keys()):
        if k.startswith("ind_") and data[k] is None:
            data[k] = 0
    return data


# --- Seus Endpoints Existentes (Mantidos) ---

@app.post("/simular")
def criar_simulacao(data: dict, session: Session = Depends(get_session)):

    simulacao = Simulacao(
        genero=data["genero"],
        idade=data["idade"],
        antiguidade=data["antiguidade"],
        renda=data["renda"],
        segmento=data["segmento"],
        provincia=data["provincia"],
        canal=data["canal"],
        produtos=json.dumps(data["produtos"])
    )

    session.add(simulacao)
    session.commit()
    session.refresh(simulacao)

    return {"id": simulacao.id}


@app.get("/simular/{id}")
def obter_simulacao(id: int, session: Session = Depends(get_session)):
    simulacao = session.get(Simulacao, id)

    if not simulacao:
        return {"erro": "simulação não encontrada"}

    # O resultado aqui ainda é mockado. Se você quiser que o resultado venha do modelo,
    # precisaria de todos os campos de CustomerFeatures na sua tabela Simulacao,
    # ou então buscar esses dados de outro lugar para preencher o CustomerFeatures.
    resultado = {
        "ranking": [
            {"pos": 1, "produto": "Cartão de Crédito", "prob": 0.9821},
            {"pos": 2, "produto": "Seguro Residencial", "prob": 0.9755},
        ]
    }

    return {
        "simulacao": {
            **simulacao.model_dump(),
            "produtos": json.loads(simulacao.produtos)
        },
        "resultado": resultado
    }


# --- NOVO Endpoint de Predição com os Modelos --- 
@app.post("/predict_products", response_model=Dict[str, float])
async def predict_products(customer_data: CustomerFeatures):
    data_dict = customer_data.model_dump()
    data_dict = _apply_defaults(data_dict)

    # Datas para string
    for k in ["fecha_dato", "fecha_alta"]:
        if isinstance(data_dict.get(k), dt_date):
            data_dict[k] = data_dict[k].strftime("%Y-%m-%d")

    input_df = pd.DataFrame([data_dict])

    try:
        processed_df = preprocess_dataframe_for_api(input_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro durante o pré-processamento: {e}")

    if processed_df.empty:
        raise HTTPException(status_code=400, detail="Cliente removido após pré-processamento.")

    if not top_5_xgb_models:
        raise HTTPException(status_code=500, detail="Modelos de ML não carregados.")

    predictions = {}
    for product_name, model in top_5_xgb_models.items():
        feature_names = getattr(model, "feature_names", None)
        if feature_names:
            df_for_model = processed_df.reindex(columns=list(feature_names), fill_value=0)
        else:
            df_for_model = processed_df
        dtest = xgb.DMatrix(df_for_model)
        proba = model.predict(dtest)[0]
        predictions[product_name.replace("_new", "")] = float(proba)

    return predictions


# --- NOVO: Utilitário para construir linha bruta a partir do SimulacaoInput ---
def _map_segmento_to_original(seg: Optional[str]) -> str:
    if not seg:
        return "02 - PARTICULARES"
    s = seg.strip().lower()
    if s.startswith("top"):
        return "01 - TOP"
    if s.startswith("part"):
        return "02 - PARTICULARES"
    if s.startswith("univ"):
        return "03 - UNIVERSITARIO"
    return "02 - PARTICULARES"


def _map_genero_to_original(gen: Optional[str]) -> str:
    if not gen:
        return "H"
    g = gen.strip().upper()
    if g in {"H", "M"}:  # H/M -> homem (dataset)
        return "H"
    if g in {"V", "F"}:  # V/F -> mulher (dataset)
        return "V"
    return "H"


def _provincia_codigo(nome_prov: Optional[str]) -> float:
    if not nome_prov:
        return 28.0  # MADRID por padrão
    nome = str(nome_prov).upper()
    # Mapa simples: hash estável em uma faixa, mas determinístico
    # Preferível ter um dicionário real; aqui garantimos não-NaN
    return float(abs(hash(nome)) % 50 + 1)


def build_raw_from_simulacao(payload: SimulacaoInput) -> pd.DataFrame:
    today_str = date.today().strftime("%Y-%m-%d")

    # Preparar todos os 24 indicadores originais como 0
    product_indicator_original_cols = {k: v for k, v in translate_columns_api.items() if k.startswith("ind_")}
    produto_pt_to_original = {v: k for k, v in product_indicator_original_cols.items()}
    indicadores = {col: 0 for col in produto_pt_to_original.values()}

    # Marcar produtos que o usuário já possui
    if payload.produtos:
        for p in payload.produtos:
            pt = str(p).strip()
            if pt in produto_pt_to_original:
                indicadores[produto_pt_to_original[pt]] = 1

    raw = {
        # Campos de data/identificação
        "fecha_dato": payload and today_str,
        "ncodpers": 1,
        "ind_empleado": "N",
        # Localização e demografia
        "pais_residencia": "ES",
        "sexo": _map_genero_to_original(getattr(payload, "genero", None)),
        "age": getattr(payload, "idade", None) or 35,
        "fecha_alta": today_str,
        "ind_nuevo": 0,
        "antiguedad": float(getattr(payload, "antiguidade", None) or 12),
        "indrel": 1.0,
        "ult_fec_cli_1t": None,
        "indrel_1mes": 1.0,  # evitar drop por NaN
        "tiprel_1mes": "A",
        "indresi": "S",
        "indext": "N",
        "conyuemp": "N",
        "canal_entrada": getattr(payload, "canal", None) or "KHL",
        "indfall": "N",
        "tipodom": 1,
        "cod_prov": _provincia_codigo(getattr(payload, "provincia", None)),
        "nomprov": (getattr(payload, "provincia", None) or "MADRID").upper(),
        "ind_actividad_cliente": 1.0,
        "renta": float(getattr(payload, "renda", None) or 120000.0),
        "segmento": _map_segmento_to_original(getattr(payload, "segmento", None)),
        # Indicadores de produtos atuais
        **indicadores,
    }

    return pd.DataFrame([raw])


# --- NOVO: Endpoint amigável para recomendar top-5 ---
@app.post("/recomendar")
async def recomendar(simulacao: SimulacaoInput):
    try:
        raw_df = build_raw_from_simulacao(simulacao)
        processed_df = preprocess_dataframe_for_api(raw_df)
        if processed_df.empty:
            raise HTTPException(status_code=400, detail="Dados insuficientes após pré-processamento.")

        if not top_5_xgb_models:
            raise HTTPException(status_code=500, detail="Modelos não carregados.")

        preds = {}
        for product_name, model in top_5_xgb_models.items():
            feature_names = getattr(model, "feature_names", None)
            if feature_names:
                df_for_model = processed_df.reindex(columns=list(feature_names), fill_value=0)
            else:
                df_for_model = processed_df
            dtest = xgb.DMatrix(df_for_model)
            proba = float(model.predict(dtest)[0])
            base_name = product_name.replace("_new", "")
            preds[base_name] = proba

        # Não recomendar produtos já possuídos
        possui = set(simulacao.produtos or [])
        ranking = [
            {"produto": k, "prob": v}
            for k, v in sorted(preds.items(), key=lambda x: x[1], reverse=True)
            if k not in possui
        ][:5]

        return {"ranking": ranking}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao recomendar: {e}")
