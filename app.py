from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag_chain import retriever_multi_vector_img , multi_modal_rag_chain

# Initialize FastAPI app
app = FastAPI()

# Create the chain once
rag_chain = multi_modal_rag_chain(retriever_multi_vector_img)

# Request schema
class QueryRequest(BaseModel):
    question: str

# Endpoint
@app.post("/ask")
async def ask_question(req: QueryRequest):
    try:
        result = rag_chain.invoke(req.question)
        return {"response": result}
    except Exception as e:
        return {"error": str(e)}

