from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Session, select
from database import engine, get_session
from models import Simulacao
from schemas import SimulacaoInput

import json
import pandas as pd
import numpy as np
import joblib
import pickle
import xgboost as xgb
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import date as dt_date, date
import os
from sklearn.preprocessing import OrdinalEncoder

app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def create_db():
    SQLModel.metadata.create_all(engine)


# --- 1. Preparar Encoders (mesma lógica do notebook) ---
idade_order = ['Jovem', 'Jovem_Adulto', 'Adulto', 'Meia_Idade', 'Idoso']
renda_order = ['Baixa', 'Media_Baixa', 'Media_Alta', 'Alta']

encoder_idade = OrdinalEncoder(categories=[idade_order], handle_unknown='use_encoded_value', unknown_value=-1)
encoder_renda = OrdinalEncoder(categories=[renda_order], handle_unknown='use_encoded_value', unknown_value=-1)

# Fit com dados de exemplo (necessário para o encoder funcionar)
encoder_idade.fit(pd.DataFrame({'idade_categoria': idade_order}))
encoder_renda.fit(pd.DataFrame({'renda_categoria': renda_order}))

print("Encoders configurados.")


# --- 2. Carregar Modelos do Novo Diretório /models ---
MODELS_DIR = './models'

# Top 5 produtos
top_5_products = [
    'conta_corrente_new', 'recebimento_recibos_new', 'conta_terceiros_new',
    'conta_nominal_new', 'conta_eletronica_new'
]

# Carregar modelos XGBoost em formato .pkl
top_5_xgb_models = {}
for product_name in top_5_products:
    model_path = os.path.join(MODELS_DIR, f'xgboost_{product_name}.pkl')
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        top_5_xgb_models[product_name] = model
        # Verificar feature_names
        try:
            feature_names = model.feature_names if hasattr(model, 'feature_names') else model.get_booster().feature_names if hasattr(model.get_booster(), 'feature_names') else None
            if feature_names:
                print(f"Modelo '{product_name}' espera {len(feature_names)} features.")
        except:
            pass
        print(f"Modelo XGBoost para '{product_name}' carregado de {model_path}.")
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o modelo para '{product_name}' de {model_path}. Erro: {e}")
        print("A API pode não conseguir fazer previsões para este produto.")


# --- 2. Dicionário de Tradução de Colunas ---
translate_columns_api = {
    # Parte 1 – Dados do cliente
    "fecha_dato": "data_referencia",
    "ncodpers": "id_cliente",
    "ind_empleado": "tipo_empregado",
    "pais_residencia": "pais_residencia",
    "sexo": "sexo",
    "age": "idade",
    "fecha_alta": "data_entrada_banco",
    "ind_nuevo": "cliente_novo",
    "antiguedad": "antiguidade_meses",
    "indrel": "tipo_relacionamento",
    "ult_fec_cli_1t": "ultima_data_cliente_trimestre",
    "indrel_1mes": "relacionamento_mes",
    "tiprel_1mes": "tipo_relacionamento_mes",
    "indresi": "residente",
    "indext": "estrangeiro",
    "conyuemp": "empregado_banco",
    "canal_entrada": "canal_aquisicao",
    "indfall": "falecido",
    "tipodom": "tipo_endereco",
    "cod_prov": "codigo_provincia",
    "nomprov": "nome_provincia",
    "ind_actividad_cliente": "ativo_mes_passado",
    "renta": "renda_estimativa",
    "segmento": "segmento_marketing",

    # Parte 2 – Produtos financeiros
    "ind_ahor_fin_ult1": "conta_poupanca",
    "ind_aval_fin_ult1": "garantia_aval",
    "ind_cco_fin_ult1": "conta_corrente",
    "ind_cder_fin_ult1": "deposito_prazo",
    "ind_cno_fin_ult1": "conta_nominal",
    "ind_ctju_fin_ult1": "conta_jovem",
    "ind_ctma_fin_ult1": "conta_maior_idade",
    "ind_ctop_fin_ult1": "conta_terceiros",
    "ind_ctpp_fin_ult1": "conta_particular",
    "ind_deco_fin_ult1": "fundo_investimento_corporativo",
    "ind_deme_fin_ult1": "deposito_mercado_monetario",
    "ind_dela_fin_ult1": "deposito_prazo_longo",
    "ind_ecue_fin_ult1": "conta_eletronica",
    "ind_fond_fin_ult1": "fundo_investimento",
    "ind_hip_fin_ult1": "hipoteca",
    "ind_plan_fin_ult1": "plano_pensao",
    "ind_pres_fin_ult1": "emprestimo_pessoal",
    "ind_reca_fin_ult1": "conta_cobranca_recibos",
    "ind_tjcr_fin_ult1": "cartao_credito",
    "ind_valo_fin_ult1": "valores_mobiliarios",
    "ind_viv_fin_ult1": "credito_habitacao",
    "ind_nomina_ult1": "deposito_salario",
    "ind_nom_pens_ult1": "deposito_pensao",
    "ind_recibo_ult1": "recebimento_recibos"
}


# --- 3. Função de Pré-processamento (BASEADA NO NOTEBOOK) ---
def preprocess_dataframe_for_api(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pré-processamento usando a mesma lógica do notebook de treinamento.
    Gera as 62 features esperadas pelos modelos XGBoost.
    """
    print(f"[DEBUG] preprocess - Input shape: {df_raw.shape}, columns: {list(df_raw.columns)[:10]}...")
    
    df = df_raw.copy()

    # 1. Converter tipos básicos
    df['idade'] = pd.to_numeric(df['idade'], errors='coerce').fillna(35)
    df['renda_estimativa'] = pd.to_numeric(df['renda_estimativa'], errors='coerce').fillna(120000.0)
    df['antiguidade_meses'] = pd.to_numeric(df['antiguidade_meses'], errors='coerce').fillna(12.0)
    
    # 2. Preencher categoricas
    categorical_cols = ["sexo", "pais_residencia", "segmento_marketing", "canal_aquisicao",
                       "tipo_relacionamento", "tipo_relacionamento_mes", "codigo_provincia", "nome_provincia"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Desconhecido" if df[col].dtype == 'object' else 0)

    # 3. Features de engenharia: renda_log e renda_variacao
    df['renda_log'] = np.log1p(df['renda_estimativa'])
    df['renda_variacao'] = 0.0  # Para novos clientes, assume 0
    
    # 4. Features _prev (produtos do mês anterior)
    product_cols_base = ['conta_corrente', 'conta_eletronica', 'conta_nominal', 'conta_terceiros', 'recebimento_recibos']
    for prod_base in product_cols_base:
        # Se não vier no input, assume 0
        df[f'{prod_base}_prev'] = df.get(f'{prod_base}_prev', 0)
    
    # 5. Faixas de Idade
    bins_age = [0, 25, 35, 45, 60, 100]
    labels_age = ['Jovem', 'Jovem_Adulto', 'Adulto', 'Meia_Idade', 'Idoso']
    df['idade_categoria'] = pd.cut(df['idade'], bins=bins_age, labels=labels_age, right=False)
    df['idade_categoria'] = df['idade_categoria'].fillna('Adulto')  # Fallback
    
    # 6. Faixas de Renda (simplificado - idealmente usar os mesmos quantis do treino)
    # Usando quantis fixos como aproximação
    if df['renda_estimativa'].nunique() > 1:
        bins_renda = [0, 75000, 120000, df['renda_estimativa'].max() + 1]
    else:
        bins_renda = [0, 75000, 120000, 300000]
    labels_renda = ['Baixa', 'Media_Baixa', 'Media_Alta']
    
    # Garantir pelo menos 3 bins
    if len(bins_renda) < 4:
        bins_renda = [0, 75000, 120000, 300000]
    
    df['renda_categoria'] = pd.cut(df['renda_estimativa'], bins=bins_renda, labels=labels_renda[:len(bins_renda)-1], include_lowest=True)
    df['renda_categoria'] = df['renda_categoria'].fillna('Media_Baixa')  # Fallback
    
    # Adicionar 'Alta' se necessário para match com encoder
    if 'Alta' not in df['renda_categoria'].cat.categories:
        df['renda_categoria'] = df['renda_categoria'].cat.add_categories(['Alta'])
    
    # 7. Outras features criadas
    df['possui_pacote_basico'] = ((df.get('conta_corrente', 0) == 1) & (df.get('recebimento_recibos', 0) == 1)).astype(int)
    
    # 8. Features de Interação produto-idade
    for prod_base in product_cols_base:
        df[f'{prod_base}_idade_interacao'] = df.get(prod_base, 0) * df['idade']
    
    # 9. Features _mes_anterior (cópia dos produtos atuais se não fornecido)
    for prod_base in product_cols_base:
        if f'{prod_base}_mes_anterior' not in df.columns:
            df[f'{prod_base}_mes_anterior'] = df.get(prod_base, 0)
    
    # 10. Aplicar Encoders
    try:
        df['idade_categoria_encoded'] = encoder_idade.transform(df[['idade_categoria']]).flatten()
    except Exception as e:
        print(f"Erro ao encodar idade_categoria: {e}")
        df['idade_categoria_encoded'] = 2  # Default 'Adulto'
    
    try:
        df['renda_categoria_encoded'] = encoder_renda.transform(df[['renda_categoria']]).flatten()
    except Exception as e:
        print(f"Erro ao encodar renda_categoria: {e}")
        df['renda_categoria_encoded'] = 1  # Default 'Media_Baixa'
    
    # 11. Target Encoding (usando valores default - IDEALMENTE carregar mappings salvos)
    # Sem os mappings reais, usamos 0.5 como média global para todas as TE features
    te_cols = [
        "canal_aquisicao", "codigo_provincia", "nome_provincia",
        "relacionamento_mes", "tipo_relacionamento_mes", "segmento_marketing"
    ]
    
    # Criar feature segmento_idade para TE
    if 'segmento_marketing' in df.columns:
        df['segmento_idade'] = df['segmento_marketing'].astype(str) + '_' + df['idade_categoria'].astype(str)
    else:
        df['segmento_idade'] = 'Desconhecido_Adulto'
    
    te_cols.append('segmento_idade')
    
    products_te = ['conta_corrente', 'conta_eletronica', 'conta_nominal', 'conta_terceiros', 'recebimento_recibos']
    
    # Preencher todas as features TE com valor default 0.5 (média global)
    # TODO: Substituir por mappings reais salvos durante o treinamento
    for col in te_cols:
        for prod_te in products_te:
            feature_name = f"TE_{col}_{prod_te}"
            df[feature_name] = 0.5  # Valor default sem os mappings reais
    
    # 12. Selecionar apenas as 62 features esperadas pelo modelo (na ordem correta)
    expected_features = [
        'ativo_mes_passado', 'conta_corrente', 'conta_nominal', 'conta_terceiros',
        'conta_eletronica', 'conta_cobranca_recibos', 'recebimento_recibos',
        'conta_corrente_prev', 'conta_eletronica_prev', 'conta_nominal_prev',
        'conta_terceiros_prev', 'recebimento_recibos_prev',
        'renda_log', 'renda_variacao', 'possui_pacote_basico',
        'conta_corrente_idade_interacao', 'conta_eletronica_idade_interacao',
        'conta_nominal_idade_interacao', 'conta_terceiros_idade_interacao',
        'recebimento_recibos_idade_interacao',
        'conta_corrente_mes_anterior', 'conta_eletronica_mes_anterior',
        'conta_nominal_mes_anterior', 'conta_terceiros_mes_anterior',
        'recebimento_recibos_mes_anterior',
        'idade_categoria_encoded', 'renda_categoria_encoded',
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
    
    # Garantir que todas as features existem (fill com 0 se ausente)
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Retornar apenas as 62 features na ordem correta
    df_final = df[expected_features]
    
    print(f"[DEBUG] preprocess - Output shape: {df_final.shape}, columns: {list(df_final.columns)[:10]}...")
    
    return df_final
    
    return df_final


# --- 4. Esquema de Entrada (CustomerFeatures) ---
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
    """Aplica valores padrão mínimos."""
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
        "ind_cco_fin_ult1": 0,
        "ind_recibo_ult1": 0,
        "ind_nomina_ult1": 0
    }
    for k, v in defaults.items():
        if data.get(k) in (None, ""):
            data[k] = v
    for k in list(data.keys()):
        if k.startswith("ind_") and data[k] is None:
            data[k] = 0
    return data


# --- 5. Endpoints Existentes (Mantidos) ---
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


# --- 6. Endpoint de Predição com os Novos Modelos ---
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
        try:
            dmatrix = xgb.DMatrix(processed_df)
            proba = float(model.predict(dmatrix)[0])
            predictions[product_name.replace("_new", "")] = proba
            print(f"Predição '{product_name}': {proba} (shape: {processed_df.shape})")
        except Exception as e:
            print(f"Erro ao prever '{product_name}': {e}")
            print(f"  - DataFrame shape: {processed_df.shape}")
            print(f"  - DataFrame columns: {list(processed_df.columns)[:10]}...")
            predictions[product_name.replace("_new", "")] = 0.0

    return predictions


# --- 7. Utilitários para /recomendar ---
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
    if g in {"H", "M"}:
        return "H"
    if g in {"V", "F"}:
        return "V"
    return "H"


def _provincia_codigo(nome_prov: Optional[str]) -> float:
    if not nome_prov:
        return 28.0
    return float(abs(hash(nome_prov.upper())) % 50 + 1)


def build_raw_from_simulacao(payload: SimulacaoInput) -> pd.DataFrame:
    """
    Constrói um DataFrame com os dados brutos de entrada da simulação.
    Inclui campos necessários para o preprocessamento baseado no notebook.
    """
    today_str = date.today().strftime("%Y-%m-%d")

    # Mapear produtos do payload para os indicadores originais
    product_indicator_original_cols = {k: v for k, v in translate_columns_api.items() if k.startswith("ind_")}
    produto_pt_to_original = {v: k for k, v in product_indicator_original_cols.items()}
    indicadores = {col: 0 for col in produto_pt_to_original.values()}

    if payload.produtos:
        for p in payload.produtos:
            pt = str(p).strip()
            if pt in produto_pt_to_original:
                indicadores[produto_pt_to_original[pt]] = 1

    # Construir dicionário raw com campos renomeados para PT-BR
    raw = {
        # Campos básicos
        "idade": getattr(payload, "idade", None) or 35,
        "renda_estimativa": float(getattr(payload, "renda", None) or 120000.0),
        "antiguidade_meses": float(getattr(payload, "antiguidade", None) or 12),
        "sexo": _map_genero_to_original(getattr(payload, "genero", None)),
        "segmento_marketing": _map_segmento_to_original(getattr(payload, "segmento", None)),
        "codigo_provincia": _provincia_codigo(getattr(payload, "provincia", None)),
        "nome_provincia": (getattr(payload, "provincia", None) or "MADRID").upper(),
        "canal_aquisicao": getattr(payload, "canal", None) or "KHL",
        "pais_residencia": "ES",
        "tipo_relacionamento": "1",
        "tipo_relacionamento_mes": "A",
        "relacionamento_mes": 1.0,
        "ativo_mes_passado": 1.0,
        
        # Produtos atuais (traduzidos)
        "conta_corrente": indicadores.get("ind_cco_fin_ult1", 0),
        "conta_eletronica": indicadores.get("ind_ecue_fin_ult1", 0),
        "conta_nominal": indicadores.get("ind_cno_fin_ult1", 0),
        "conta_terceiros": indicadores.get("ind_ctop_fin_ult1", 0),
        "recebimento_recibos": indicadores.get("ind_recibo_ult1", 0),
        "conta_cobranca_recibos": indicadores.get("ind_reca_fin_ult1", 0),
        
        # Produtos do mês anterior (para novos clientes, assume mesmos valores)
        "conta_corrente_prev": indicadores.get("ind_cco_fin_ult1", 0),
        "conta_eletronica_prev": indicadores.get("ind_ecue_fin_ult1", 0),
        "conta_nominal_prev": indicadores.get("ind_cno_fin_ult1", 0),
        "conta_terceiros_prev": indicadores.get("ind_ctop_fin_ult1", 0),
        "recebimento_recibos_prev": indicadores.get("ind_recibo_ult1", 0),
    }

    return pd.DataFrame([raw])


# --- 8. Endpoint /recomendar ---
@app.post("/recomendar")
async def recomendar(simulacao: SimulacaoInput, debug: bool = False):
    try:
        raw_df = build_raw_from_simulacao(simulacao)
        processed_df = preprocess_dataframe_for_api(raw_df)
        if processed_df.empty:
            raise HTTPException(status_code=400, detail="Dados insuficientes após pré-processamento.")

        if not top_5_xgb_models:
            raise HTTPException(status_code=500, detail="Modelos não carregados.")

        preds = {}
        print(f"DataFrame processado: shape={processed_df.shape}, colunas={list(processed_df.columns)[:10]}...")
        for product_name, model in top_5_xgb_models.items():
            try:
                dmatrix = xgb.DMatrix(processed_df)
                proba = float(model.predict(dmatrix)[0])
                base_name = product_name.replace("_new", "")
                preds[base_name] = proba
                print(f"Predição '{base_name}': {proba}")
            except Exception as e:
                print(f"Erro ao prever '{product_name}': {e}")
                import traceback
                traceback.print_exc()
                preds[product_name.replace("_new", "")] = 0.0

        # Não recomendar produtos já possuídos
        possui = set(simulacao.produtos or [])
        ranking = [
            {"produto": k, "prob": v}
            for k, v in sorted(preds.items(), key=lambda x: x[1], reverse=True)
            if k not in possui
        ][:5]

        if not debug:
            return {"ranking": ranking}

        # Modo debug
        feature_row = processed_df.iloc[0].to_dict()
        debug_info = {
            "processed_features_sample": {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in list(feature_row.items())[:20]},
            "processed_shape": processed_df.shape
        }

        return {"ranking": ranking, "debug": debug_info}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao recomendar: {e}")


# --- 9. Endpoint de Diagnóstico ---
@app.get("/_artifacts")
def artifacts_status():
    models_loaded = list(top_5_xgb_models.keys())
    return {
        "models_directory": MODELS_DIR,
        "models_loaded": models_loaded,
        "models_count": len(models_loaded)
    }
