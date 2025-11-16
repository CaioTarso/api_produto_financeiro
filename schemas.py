from pydantic import BaseModel
from typing import List, Optional

SEGMENTOS = ["top", "particulares", "universitario"]

CANAL_LIST = ['KAT','KFC','KFA','007','013','KCG','KAF','KAW','KCH','KGX','RED','KAS','KHN','KAY','KBW','KAG','KBZ','KHE','Desconhecido','KAH','KAA','KAU','KHK','KHL','KHM','KAZ','KAR','KCF','KEG','KEY','KCN','KCC','KBH','KDX','KBF','KBD','KCB','KBU','KES','KBO','KCI','KAB','KCU','KAC','KDR','KDC','KEF','KCM','KDY','KCA','KDS','KDU','KEA','KAE','KCL','KAD','KCD','KFD','KDO','KBQ','KFP','KEI','KHC','KHO','KEW','KDM','KEZ','KEL','KAP','KAI','KAJ','KEJ','KBJ','KED','KAO','KAK','KAL','KBR','KEN','KBB','KBS','K00','KAQ','KFK','KAM','KHQ','KBM','KGY','KFU','KFM','KFN','KFH','KFG','KFT','KFJ','KFS','KGV','KFF','KBG','KHD','KEH','KHF','KHP']

PROVINCIAS = ['MADRID','GRANADA','MALAGA','BARCELONA','ALICANTE','ALMERIA','VALLADOLID','SEVILLA','ZAMORA','GIRONA','VALENCIA','HUELVA','GIPUZKOA','ASTURIAS','BALEARS, ILLES','CANTABRIA','JAEN','SANTA CRUZ DE TENERIFE','MURCIA','LERIDA','CUENCA','CIUDAD REAL','BIZKAIA','CADIZ','ALBACETE','TARRAGONA','CORUÑA, A','BURGOS','BADAJOZ','ALAVA','PALMAS, LAS','RIOJA, LA','MELILLA','OURENSE','ZARAGOZA','NAVARRA','GUADALAJARA','CASTELLON','PONTEVEDRA','SALAMANCA','CEUTA','TOLEDO','CORDOBA','HUESCA','SORIA','CACERES','LUGO','LEON','PALENCIA','AVILA','TERUEL','SEGOVIA']

PRODUTOS_PERMITIDOS = ["conta_corrente","cartao_credito","plano_pensao","recebimento_recibos","conta_nominal","conta_maior_idade","conta_terceiros","conta_particular","deposito_prazo_longo","conta_eletronica","fundo_investimento","valores_mobiliarios","deposito_salario","deposito_pensao"]

class SimulacaoInput(BaseModel):
    genero: Optional[str]
    idade: Optional[int]
    antiguidade: Optional[int]
    renda: Optional[int]
    segmento: Optional[str]
    provincia: Optional[str]
    canal: Optional[str]
    produtos: Optional[List[str]] = []
