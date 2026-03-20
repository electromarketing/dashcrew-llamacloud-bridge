# main.py

import os
import sqlite3
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel

# ---------------------------
# Config (from environment)
# ---------------------------

LLAMACLOUD_API_KEY = os.getenv("LLAMACLOUD_API_KEY", "")
LLAMACLOUD_PROJECT = os.getenv("LLAMACLOUD_PROJECT", "default-project")
LLAMACLOUD_ORG = os.getenv("LLAMACLOUD_ORG", "default-org")

DB_PATH = os.getenv("INDEX_DB_PATH", "user_indexes.db")

PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "https://typingmind-llamacloud-bridge.example.com",  # update after deploy
)


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
    Stub: replace with real LlamaCloud/LlamaIndex query.

    Here you will:
    - Use LLAMACLOUD_API_KEY / PROJECT / ORG
    - Attach to or create the LlamaCloud index "index_name"
    - Run the query and return the answer text
    """
    # TODO: implement using LlamaIndex Cloud SDK or HTTP.
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
# Lifespans
# ---------------------------

@asynccontextmanager
async def fastapi_lifespan(app: FastAPI) -> AsyncIterator[None]:
    print("FastAPI startup: init DB")
    init_db()
    yield
    print("FastAPI shutdown")


@asynccontextmanager
async def mcp_lifespan() -> AsyncIterator[None]:
    print("MCP startup")
    yield
    print("MCP shutdown")


# ---------------------------
# FastAPI app
# ---------------------------

app = FastAPI(
    title="TypingMind LlamaCloud Bridge",
    lifespan=fastapi_lifespan,
)


# ---------------------------
# REST endpoint (optional)
# ---------------------------

@app.post("/plugin/rag", response_model=RagResponse)
async def rag_tool(payload: RagRequest) -> RagResponse:
    """
    Per-user RAG endpoint:
    - Auto-creates logical index per user_id
    - Queries that index with 'query'
    """
    if not payload.user_id or not payload.query:
        raise HTTPException(status_code=400, detail="user_id and query are required")

    index_name = get_or_create_index_name(payload.user_id)
    answer = query_llamacloud(index_name=index_name, query=payload.query)

    return RagResponse(answer=answer, index_name=index_name)


# ---------------------------
# MCP server mounted on /mcp
# ---------------------------

mcp = FastApiMCP(
    app,
    name="LlamaCloud RAG MCP",
    description=(
        "Per-user LlamaCloud RAG tool. Accepts user_id + query, "
        "auto-creates a dedicated index per TypingMind user, and queries it."
    ),
    base_url=PUBLIC_BASE_URL,
    lifespan=mcp_lifespan,
)

mcp.mount()


# ---------------------------
# Dev entrypoint
# ---------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
