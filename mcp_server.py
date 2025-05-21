import asyncio
from dotenv import load_dotenv
from rag import RAGWorkflow
from mcp.server.fastmcp import FastMCP

# Load .env for configuration
load_dotenv(override=True)

import os

# Initialize MCP server
mcp = FastMCP('rag-server')

# Get configuration from environment variables (with defaults)
provider = os.environ.get("LLM_PROVIDER", "ollama")
embed_provider = os.environ.get("EMBED_PROVIDER", "huggingface")
model_name = os.environ.get("MODEL_NAME", "mistral")
openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o")
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_base_url = os.environ.get("OPENAI_BASE_URL")
data_dir = os.environ.get("DATA_DIR", "data")

# Embedding model depends on the provider
if embed_provider == "ollama":
    embedding_model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
else:
    embedding_model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

print(f"🔌 Using LLM provider: {provider}")
print(f"🧠 Using embedding provider: {embed_provider}")

if provider.lower() == "openai":
    # Using OpenAI
    rag_workflow = RAGWorkflow(
        llm_provider="openai",
        model_name=openai_model,
        embed_provider=embed_provider,
        embedding_model=embedding_model,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url
    )
else:
    # Using Ollama
    rag_workflow = RAGWorkflow(
        llm_provider="ollama", 
        model_name=model_name,
        embed_provider=embed_provider,
        embedding_model=embedding_model
    )

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
async def reload_docs(path: str = None) -> str:
    """
    Re-ingest your local knowledge base from given folder path.
    """
    doc_path = path or data_dir
    print(f"📂 Re-indexing documents in: {doc_path}")
    await rag_workflow.ingest_documents(doc_path)
    return f"✅ Re-ingested docs from '{doc_path}'"


# 🔧 Startup: Pre-load default KB
if __name__ == "__main__":
    async def interactive_chat():
        """Run an interactive chat session with the RAG system"""
        print("🚀 Starting RAG Chat - Type 'exit' to quit\n")
        
        # Ingest documents at startup
        print(f"📚 Loading knowledge base from '{data_dir}'...")
        await rag_workflow.ingest_documents(data_dir)
        print("✅ Knowledge base loaded. You can start chatting!\n")
        
        while True:
            # Get user input
            user_query = input("🧑‍💻 You: ")
            
            # Check for exit command
            if user_query.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
                
            # Check for reload command
            if user_query.lower().startswith('reload '):
                path = user_query[7:].strip() or data_dir
                print(f"📂 Reloading knowledge base from {path}...")
                await rag_workflow.ingest_documents(path)
                print("✅ Knowledge base reloaded.")
                continue
                
            # Process normal query
            print("\n🤖 Assistant: ", end="")
            response = await rag_workflow.query(user_query)

            print(response.result)
            print("\n")
            
    
    # Run the interactive chat
    asyncio.run(interactive_chat())
