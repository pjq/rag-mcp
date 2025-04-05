import os
import nest_asyncio
nest_asyncio.apply()

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
    def __init__(self, model_name="mistral", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()

        print("🛠️ Initializing LLM and embedding model...")
        self.llm = Ollama(model=model_name)
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
    rag = RAGWorkflow()

    await rag.ingest_documents("data")
    result = await rag.query("What is EIP-8514?")

    print("⏳ Waiting for LLM response...\n")
    async for chunk in result.async_response_gen():
        print(chunk, end="", flush=True)
    print("\n✅ Done!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
