# main.py

import os
import sqlite3
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------
# Config (env vars)
# ---------------------------

LLAMACLOUD_API_KEY = os.getenv("LLAMACLOUD_API_KEY", "")
LLAMACLOUD_PROJECT = os.getenv("LLAMACLOUD_PROJECT", "default-project")
LLAMACLOUD_ORG = os.getenv("LLAMACLOUD_ORG", "default-org")

DB_PATH = os.getenv("INDEX_DB_PATH", "user_indexes.db")


# ---------------------------
# SQLite helpers
# ---------------------------

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_index_map (
                user_id TEXT PRIMARY KEY,
                index_name TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def get_or_create_index_name(user_id: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT index_name FROM user_index_map WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        if row:
            return row[0]

        index_name = f"tm_user_{user_id}_index"

        cur.execute(
            "INSERT INTO user_index_map (user_id, index_name) VALUES (?, ?)",
            (user_id, index_name),
        )
        conn.commit()
        return index_name
    finally:
        conn.close()


# ---------------------------
# LlamaCloud query stub
# ---------------------------

def query_llamacloud(index_name: str, query: str) -> str:
    """
    TODO: replace with real LlamaCloud/LlamaIndex call using:
      - LLAMACLOUD_API_KEY
      - LLAMACLOUD_PROJECT
      - LLAMACLOUD_ORG
    """
    return (
        f"[DEMO ANSWER] Index={index_name}, "
        f"Project={LLAMACLOUD_PROJECT}, Org={LLAMACLOUD_ORG}, "
        f"Query={query[:200]}"
    )


# ---------------------------
# Pydantic models
# ---------------------------

class RagRequest(BaseModel):
    user_id: str
    query: str


class RagResponse(BaseModel):
    answer: str
    index_name: str


# ---------------------------
# Lifespan
# ---------------------------

@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncIterator[None]:
    print("FastAPI startup: init DB")
    init_db()
    yield
    print("FastAPI shutdown")


# ---------------------------
# FastAPI app
# ---------------------------

app = FastAPI(
    title="TypingMind LlamaCloud Bridge (HTTP)",
    lifespan=app_lifespan,
)


@app.post("/plugin/rag", response_model=RagResponse)
async def rag_tool(payload: RagRequest) -> RagResponse:
    """
    Per-user RAG endpoint.

    Expect JSON:
    {
      "user_id": "...",
      "query": "..."
    }
    """
    if not payload.user_id or not payload.query:
        raise HTTPException(status_code=400, detail="user_id and query are required")

    index_name = get_or_create_index_name(payload.user_id)
    answer = query_llamacloud(index_name=index_name, query=payload.query)

    return RagResponse(answer=answer, index_name=index_name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
