from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import load_rag_pipeline

app = FastAPI()
qa_chain = load_rag_pipeline()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "RAG API is running"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    response = qa_chain.run(request.query)
    return {"response": response}