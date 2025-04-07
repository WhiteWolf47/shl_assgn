# chat.py
import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_mistralai.embeddings import MistralAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv

# ── Streamlit Page Config MUST BE FIRST ───────────────────────────────────────
st.set_page_config(page_title="SHL Assessment Recommender")
st.title("SHL Assessment Recommendation System")

# ── Configuration ──────────────────────────────────────────────────────────────

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX = st.secrets["PINECONE_INDEX"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]

EMBED_MODEL = "mistral-embed"

# Initialize components
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
embedder = MistralAIEmbeddings(model=EMBED_MODEL)
llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)

# ── Helper Functions ────────────────────────────────────────────────────────────
def scrape_job_description(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join(soup.get_text().split()[:1000])  # Return first 1000 words
    except Exception as e:
        st.error(f"Error scraping URL: {str(e)}")
        return None

def rag_query(query_text, top_k=10):
    # Generate embedding for query
    query_embedding = embedder.embed_query(query_text)
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return [match.metadata for match in results.matches]

def format_recommendations(context, query):
    prompt = ChatPromptTemplate.from_template("""
    You are an SHL assessment expert. Recommend assessments based on the job requirements.
    
    Job Description: {query}
    
    Available Assessments:
    {context}
    
    Format recommendations as a markdown table with columns:
    - Assessment Name (as link)
    - Remote Testing
    - Adaptive/IRT
    - Details
    - Test Type
    
    Include only assessments matching the job requirements. Max 10 recommendations.
    """)
    
    chain = prompt | llm
    return chain.invoke({"context": context, "query": query}).content


input_method = st.radio("Input Method:", ["Text", "URL"])

query_text = ""
if input_method == "Text":
    query_text = st.text_area("Paste Job Description:", height=200)
else:
    url = st.text_input("Enter Job Posting URL:")
    '''if url:
        with st.spinner("Scraping job description..."):
            query_text = scrape_job_description(url)
            if query_text:
                st.text_area("Scraped Content:", value=query_text, height=200)'''

if (st.button("Get Recommendations") and input_method=="Text" and query_text) or (st.button("Get Recommendations") and input_method=="URL"):
    with st.spinner("Searching assessments..."):

        if input_method == "URL":
            with st.spinner("Scraping job description..."):
                query_text = scrape_job_description(url)
                if query_text:
                    st.text_area("Scraped Content:", value=query_text, height=200)

        # Retrieve context from Pinecone
        results = rag_query(query_text)
        
        # Format context for LLM
        context = "\n".join([
            f"Name: {res['title']}\nURL: {res['link']}\n"
            f"Remote: {res['remote_testing']}\nAdaptive: {res['adaptive_irt']}\n"
            f"Details: {res['details']}\nType: {res['test_type']}"
            for res in results
        ])
        
        # Generate and display recommendations
        response = format_recommendations(context, query_text)
        st.markdown(response)