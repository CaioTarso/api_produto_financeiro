from fastapi import FastAPI, Depends
from sqlmodel import SQLModel, Session, select
from database import engine, get_session
from models import Simulacao
import json

app = FastAPI()

@app.on_event("startup")
def create_db():
    SQLModel.metadata.create_all(engine)


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
            **simulacao.dict(),
            "produtos": json.loads(simulacao.produtos)
        },
        "resultado": resultado
    }
