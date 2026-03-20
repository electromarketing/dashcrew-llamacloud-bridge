# main.py

import os
import sqlite3
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastmcp import FastMCP  # NEW

# ---------------------------
# Config (from environment)
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
# FastAPI app (your API)
# ---------------------------

api = FastAPI(
    title="TypingMind LlamaCloud Bridge (HTTP + MCP)",
    lifespan=app_lifespan,
)


@api.post("/plugin/rag", response_model=RagResponse)
async def rag_tool(payload: RagRequest) -> RagResponse:
    if not payload.user_id or not payload.query:
        raise HTTPException(status_code=400, detail="user_id and query are required")

    index_name = get_or_create_index_name(payload.user_id)
    answer = query_llamacloud(index_name=index_name, query=payload.query)
    return RagResponse(answer=answer, index_name=index_name)


# ---------------------------
# MCP server using FastMCP
# ---------------------------

mcp = FastMCP("LlamaCloud RAG MCP")  # Name shown to MCP clients


@mcp.tool
def llamacloud_rag(user_id: str, query: str) -> dict:
    """
    Per-user LlamaCloud RAG.
    Auto-creates an index for this user_id and queries it.
    """
    index_name = get_or_create_index_name(user_id)
    answer = query_llamacloud(index_name=index_name, query=query)
    return {
        "answer": answer,
        "index_name": index_name,
    }


# Create the MCP ASGI app and mount it at /mcp
mcp_app = mcp.http_app(path="/")  # FastMCP HTTP app root[web:40]

# Mount MCP at /mcp on the same FastAPI app
api.mount("/mcp", mcp_app)  # MCP endpoint: /mcp


# ---------------------------
# ASGI app entrypoint
# ---------------------------

app = api  # This is what uvicorn runs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
