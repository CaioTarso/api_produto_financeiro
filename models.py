from sqlmodel import SQLModel, Field
from typing import Optional

class Simulacao(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    genero: str
    idade: int
    antiguidade: int
    renda: int
    segmento: str
    provincia: str
    canal: str

    produtos: str  
