from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.agent import ask

api = FastAPI(title="AgentOps Data Quality Assistant")

class Q(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

@api.get("/health")
def health():
    return {"status": "ok"}

@api.post("/ask")
def ask_route(q: Q):
    out = ask(q.question)
    return {"route": out["route"], "answer": out["answer"]}
