# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
from bs4 import BeautifulSoup
from langchain_mistralai.embeddings import MistralAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
EMBED_MODEL = "mistral-embed"

# Initialize components
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
embedder = MistralAIEmbeddings(model=EMBED_MODEL)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    text: str
    max_results: int = 10

def format_assessment(metadata):
    return {
        "assessment_name": metadata.get("title", ""),
        "url": metadata.get("link", ""),
        "remote_testing": metadata.get("remote_testing", "No"),
        "adaptive_irt": metadata.get("adaptive_irt", "No"),
        "details": metadata.get("details", ""),
        "test_type": metadata.get("test_type", "")
    }

@app.post("/recommendations")
async def get_recommendations(request: QueryRequest):
    try:
        # Generate embedding
        query_embedding = embedder.embed_query(request.text)
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=request.max_results,
            include_metadata=True
        )
        
        # Format response
        assessments = [format_assessment(match.metadata) for match in results.matches]
        return {"recommendations": assessments[:request.max_results]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)