# main.py

import os
import sqlite3
from llama_cloud_services import LlamaCloudIndex
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
    Create or connect to a LlamaCloud index by name and run the query.
    """
    if not LLAMACLOUD_API_KEY:
        raise RuntimeError("LLAMACLOUD_API_KEY is not set")

    # Connect to or create the managed index in LlamaCloud
    index = LlamaCloudIndex(
        index_name,
        project_name=LLAMACLOUD_PROJECT,
        api_key=LLAMACLOUD_API_KEY,
        # optionally: base_url=EU_BASE_URL or similar if you use a non-default region
    )

    # Get a query engine and run the query
    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    # Convert response to plain text
    return str(response)

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
