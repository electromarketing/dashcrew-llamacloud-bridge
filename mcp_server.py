# mcp_server.py

import os
import sqlite3

from fastmcp import FastMCP

LLAMACLOUD_API_KEY = os.getenv("LLAMACLOUD_API_KEY", "")
LLAMACLOUD_PROJECT = os.getenv("LLAMACLOUD_PROJECT", "default-project")
LLAMACLOUD_ORG = os.getenv("LLAMACLOUD_ORG", "default-org")
DB_PATH = os.getenv("INDEX_DB_PATH", "user_indexes.db")


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


def query_llamacloud(index_name: str, query: str) -> str:
    return (
        f"[DEMO ANSWER] Index={index_name}, "
        f"Project={LLAMACLOUD_PROJECT}, Org={LLAMACLOUD_ORG}, "
        f"Query={query[:200]}"
    )


mcp = FastMCP("LlamaCloud RAG MCP")


@mcp.tool()
def llamacloud_rag(user_id: str, query: str) -> dict:
    """
    Per-user LlamaCloud RAG.
    Auto-creates an index for this user_id and queries it.
    """
    index_name = get_or_create_index_name(user_id)
    answer = query_llamacloud(index_name=index_name, query=query)
    return {"answer": answer, "index_name": index_name}


if __name__ == "__main__":
    # Run as a standalone HTTP MCP server on port 8081, path /mcp
    mcp.run(transport="http", host="0.0.0.0", port=8081, path="/mcp")
