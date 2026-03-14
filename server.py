import os
import contextlib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from chatbot import (
    get_answer,
    load_knowledge,
    GROQ_API_KEY,
    MAX_QUERY_LENGTH,
)

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

bm25_index = None
chunks = None
metadatas = None
groq_client = None

# --- Lifespan: load knowledge once on startup ---

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global bm25_index, chunks, metadatas, groq_client

    print("Building BM25 knowledge index...")
    bm25_index, chunks, metadatas = load_knowledge()

    print("Initializing Groq client...")
    groq_client = Groq(api_key=GROQ_API_KEY)

    print("Server is ready!")
    yield
    print("Shutting down.")

# --- FastAPI app ---

app = FastAPI(
    title="Ritual Knowledge Chatbot API",
    version="2.0.0",
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
            request.message, bm25_index, chunks, metadatas, groq_client
        )
        return ChatResponse(
            answer=answer,
            sources=[Source(title=s["title"], url=s["url"]) for s in sources],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
