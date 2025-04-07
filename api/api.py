# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
from bs4 import BeautifulSoup
from langchain_mistralai.embeddings import MistralAIEmbeddings
from pinecone import Pinecone
from fastapi.middleware.cors import CORSMiddleware


# ── Configuration ──────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
EMBED_MODEL = "mistral-embed"


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

# Initialize components
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
embedder = MistralAIEmbeddings(model="mistral-embed")
llm = ChatGroq(
    temperature=0,
    model_name="deepseek-r1-distill-llama-70b",
    api_key=GROQ_API_KEY
)

def format_recommendations(context, query):
    prompt = ChatPromptTemplate.from_template("""
    You are an SHL assessment expert. Recommend assessments based on the job requirements.
    
    Job Description: {query}
    
    Available Assessments:
    {context}
    
    Format recommendations as a JSON array containing objects with these fields:
    - assessment_name (with URL as markdown link)
    - remote_testing (Yes/No)
    - adaptive_irt (Yes/No)
    - details
    - test_type
    
    Include only assessments matching the job requirements. Max 10 recommendations.
    Return only the JSON array.
    """)
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": query}).content
    return json.loads(response.strip("```json\n").strip("```"))

@app.post("/recommendations")
async def get_recommendations(request: QueryRequest):
    try:
        # Generate embedding and query Pinecone
        results = index.query(
            vector=embedder.embed_query(request.text),
            top_k=request.max_results,
            include_metadata=True
        )

        # Format context
        context = "\n".join([
            f"Name: {m['metadata']['title']}\nURL: {m['metadata']['link']}\n"
            f"Remote: {m['metadata']['remote_testing']}\nAdaptive: {m['metadata']['adaptive_irt']}\n"
            f"Details: {m['metadata']['details']}\nType: {m['metadata']['test_type']}"
            for m in results['matches']
        ])

        # Generate and parse recommendations
        recommendations = format_recommendations(context, request.text)
        return {"recommendations": recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)