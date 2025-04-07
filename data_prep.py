# data_prep.py

import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_mistralai.embeddings import MistralAIEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "shl‑assessments")

EMBED_MODEL      = "mistral-embed"
CSV_PATH         = "shl_catalog.csv"
BATCH_SIZE       = 32


# ── Initialize Pinecone Client ───────────────────────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check / create index
existing = pc.list_indexes().names()
if PINECONE_INDEX not in existing:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(PINECONE_INDEX)

# ── Load & prepare data ───────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.fillna("", inplace=True)

def make_document(row: pd.Series) -> str:
    return (
        f"Assessment: {row['title']}\n"
        f"Link:\n{row['link']}"
        f"Remote Testing: {row['remote_testing']}\n"
        f"Adaptive/IRT: {row['adaptive_irt']}\n"
        f"Test Type: {row['test_type']}\n\n"
        f"Details:\n{row['details']}"
    )

docs = [make_document(r) for _, r in df.iterrows()]
ids  = df["title"].tolist()

# ── Initialize embedder ──────────────────────────────────────────────────────
embedder = MistralAIEmbeddings(
    model=EMBED_MODEL,
    api_key=MISTRAL_API_KEY
)

# ── Embed & upsert ────────────────────────────────────────────────────────────
for i in range(0, len(docs), BATCH_SIZE):
    batch_docs = docs[i : i + BATCH_SIZE]
    batch_ids  = ids[i : i + BATCH_SIZE]
    metas      = df.iloc[i : i + BATCH_SIZE].to_dict(orient="records")

    # generate embeddings in batch
    embeddings = [embedder.embed_query(doc) for doc in batch_docs]

    # prepare upsert tuples
    to_upsert = [
        (doc_id, emb, meta)
        for doc_id, emb, meta in zip(batch_ids, embeddings, metas)
    ]

    index.upsert(vectors=to_upsert)
    print(f"Upserted {i + len(to_upsert)}/{len(docs)} rows")

print("✅ All rows embedded and upserted with Mistral.")
