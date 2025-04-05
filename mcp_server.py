import asyncio
from dotenv import load_dotenv
from rag import RAGWorkflow
from mcp.server.fastmcp import FastMCP

# Load .env (if used for config later)
load_dotenv()

# Initialize MCP server
mcp = FastMCP('rag-server')
rag_workflow = RAGWorkflow()

# 🔧 Tool 1: RAG Q&A Tool
@mcp.tool()
async def rag(query: str) -> str:
    """
    Use RAG to answer questions from your local docs.
    """
    print(f"🧠 Received query: {query}")
    response = await rag_workflow.query(query)

    # Stream result fully before returning to Cursor
    chunks = []
    async for chunk in response.async_response_gen():
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    return "".join(chunks)


# 🔧 Tool 2: Re-ingest KB dynamically
@mcp.tool()
async def reload_docs(path: str = "data") -> str:
    """
    Re-ingest your local knowledge base from given folder path.
    """
    print(f"📂 Re-indexing documents in: {path}")
    await rag_workflow.ingest_documents(path)
    return f"✅ Re-ingested docs from '{path}'"


# 🔧 Startup: Pre-load default KB
if __name__ == "__main__":
    async def main():
        await rag_workflow.ingest_documents("data")
        response = await rag_workflow.query("How do I use RISC Zero's Bonsai to write zk guest programs? Please explain the process, tools, and steps involved.")
        print("\nRESPONSE:")
        print(response.result)
        print("\n")
    
    asyncio.run(main())
