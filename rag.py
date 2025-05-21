import os
import nest_asyncio
from dotenv import load_dotenv

# Load .env for configuration
load_dotenv(override=True)

nest_asyncio.apply()

from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.settings import Settings
from llama_index.core.workflow import (
    Event, Context, Workflow,
    StartEvent, StopEvent, step
)
from llama_index.core.schema import NodeWithScore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.node_parser import SimpleNodeParser


class RetrieverEvent(Event):
    nodes: list[NodeWithScore]


class RAGWorkflow(Workflow):
    def __init__(self, 
                 llm_provider="openai",
                 model_name="mistral",
                 embed_provider="huggingface", 
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 openai_api_key=None,
                 openai_base_url=None):
        super().__init__()

        print("🛠️ Initializing LLM and embedding model...")
        if llm_provider == "openai":
            print(f"🔤 Using OpenAI LLM with model: {model_name}")
            # print api_key and base_url for debugging
            print(f"🔑 OpenAI API Key: {openai_api_key if openai_api_key else '✗ Not set'}")
            print(f"🌐 OpenAI Base URL: {openai_base_url or 'Default'}")
        else:
            print(f"🔤 Using Ollama LLM with model: {model_name}")
        
        # Initialize LLM based on provider choice
        if llm_provider == "openai":
            if not openai_api_key:
                openai_api_key = os.environ.get("OPENAI_API_KEY")
            
            
            self.llm = OpenAI(
                model=model_name,
                api_key=openai_api_key,
                api_base=openai_base_url
            )
        else:  # default to ollama
            self.llm = Ollama(model=model_name)
        
        # Initialize embedding model based on provider choice
        if embed_provider == "ollama":
            # For Ollama embeddings, use the specified model or default to nomic-embed
            embed_model_name = embedding_model if embedding_model else "nomic-embed-text"
            print(f"🔤 Using Ollama embeddings with model: {embed_model_name}")
            try:
                from llama_index.embeddings.ollama import OllamaEmbedding
                self.embed_model = OllamaEmbedding(model_name=embed_model_name)
            except ImportError:
                print("⚠️ OllamaEmbedding not available. Falling back to HuggingFace.")
                self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        else:  # default to HuggingFace
            print(f"🔤 Using HuggingFace embeddings with model: {embedding_model}")
            self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.index = None
        print("✅ Models initialized.")

    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        dirname = ev.get("dirname")
        if not dirname or not os.path.exists(dirname):
            print(f"❌ Directory '{dirname}' does not exist.")
            return None

        print(f"📂 Ingesting documents from: {dirname}")
        documents = SimpleDirectoryReader(dirname).load_data()
        parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
        nodes = parser.get_nodes_from_documents(documents)
        self.index = VectorStoreIndex(nodes)
        print(f"✅ Ingested {len(nodes)} nodes.")
        return StopEvent(result=self.index)

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        """Retrieve relevant nodes based on a user query."""
        query = ev.get("query")
        index = ev.get("index") or self.index

        if not query:
            return None
        if index is None:
            print("Index is missing. Please ingest documents first.")
            return None

        retriever = index.as_retriever(similarity_top_k=3)
        nodes = await retriever.aretrieve(query)
        await ctx.set("query", query)
        return RetrieverEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        """Generate a final answer from retrieved nodes."""
        summarizer = CompactAndRefine(streaming=False, verbose=False)
        query = await ctx.get("query", default=None)
        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)

    async def ingest_documents(self, directory: str):
        print("🚀 Running ingest_documents()...")
        ctx = Context(workflow=self)
        result = await self.ingest(ctx, StartEvent(dirname=directory))
        self.index = result.result if isinstance(result, StopEvent) else None
        return self.index

    async def query(self, query_text: str):
        print("🤖 Running query()...")
        if self.index is None:
            raise ValueError("Please ingest documents first.")

        ctx = Context(workflow=self)

        print("🟡 Running retrieve step...")
        retrieve_event = StartEvent(query=query_text, index=self.index)
        retrieve_result = await self.retrieve(ctx, retrieve_event)

        if not isinstance(retrieve_result, RetrieverEvent):
            raise ValueError("Retrieval failed.")

        print("🟡 Running synthesize step...")
        synth_result = await self.synthesize(ctx, retrieve_result)
        return synth_result


# 🔧 CLI Testing
async def main():
    # Get configuration from .env file (already loaded at the top of the file)
    # with fallbacks to default values if not set
    provider = os.environ.get("LLM_PROVIDER", "ollama")
    embed_provider = os.environ.get("EMBED_PROVIDER", "huggingface")
    model_name = os.environ.get("MODEL_NAME", "mistral")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_base_url = os.environ.get("OPENAI_BASE_URL")
    
    # Embedding model depends on the provider
    if embed_provider == "ollama":
        embedding_model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    else:
        embedding_model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    print(f"🔌 Using LLM provider: {provider}")
    print(f"🧠 Using embedding provider: {embed_provider}")
    
    if provider.lower() == "openai":
        # Using OpenAI
        rag = RAGWorkflow(
            llm_provider="openai",
            model_name=openai_model,
            embed_provider=embed_provider,
            embedding_model=embedding_model,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url
        )
    else:
        # Using Ollama
        rag = RAGWorkflow(
            llm_provider="ollama", 
            model_name=model_name,
            embed_provider=embed_provider,
            embedding_model=embedding_model
        )

    # Get data directory and query from environment variables
    data_dir = os.environ.get("DATA_DIR", "data")
    query = os.environ.get("QUERY", "What is EIP-8514?")
    
    # Print configuration info
    print(f"📂 Using data directory: {data_dir}")
    print(f"❓ Default query: {query}")
    
    await rag.ingest_documents(data_dir)
    result = await rag.query(query)

    print("⏳ Waiting for LLM response...\n")
    async for chunk in result.async_response_gen():
        print(chunk, end="", flush=True)
    print("\n✅ Done!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
