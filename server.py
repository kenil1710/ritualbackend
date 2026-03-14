import contextlib
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from chatbot import (
    get_answer,
    load_and_ingest_knowledge,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL_NAME,
    GROQ_API_KEY,
    MAX_QUERY_LENGTH,
)
import os

# --- FastAPI Setup ---
# Note: initialized below with lifespan

# --- Request / Response schemas ---

class ChatRequest(BaseModel):
    message: str

class Source(BaseModel):
    title: str
    url: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]

# --- Global references set during startup ---

embedding_model = None
collection = None
groq_client = None

# --- Lifespan: load models once on startup ---

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, collection, groq_client

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Connectinfg to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Re-create the collection to ensure fresh data on every startup
    try:
        chroma_client.delete_collection(name="ritual_knowledge")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="ritual_knowledge",
        metadata={"hnsw:space": "cosine"},
    )

    print("Ingesting knowledge...")
    success = load_and_ingest_knowledge(collection, embedding_model)
    if not success:
        raise RuntimeError("Knowledge ingestion failed. Check knowledge.json.")

    print("Initializing Groq client...")
    groq_client = Groq(api_key=GROQ_API_KEY)

    print("Server is ready!")
    yield  # app runs here
    print("Shutting down.")

# --- FastAPI app ---

app = FastAPI(
    title="Ritual Knowledge Chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Single endpoint ---

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if len(request.message) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Message is too long (max {MAX_QUERY_LENGTH} characters)."
        )

    try:
        answer, sources = get_answer(
            request.message, collection, embedding_model, groq_client
        )
        return ChatResponse(
            answer=answer,
            sources=[Source(title=s["title"], url=s["url"]) for s in sources],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Respect the PORT environment variable for deployment (Render/Heroku/etc)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
