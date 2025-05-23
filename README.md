# RAG-MCP Server

A general-purpose Retrieval-Augmented Generation (RAG) server using the Model Control Protocol (MCP), designed to be tested with RISC Zero's Bonsai documentation.

## Overview

This project implements a RAG server that:
- Uses MCP (Model Control Protocol) for standardized communication
- Implements RAG (Retrieval-Augmented Generation) workflow for document querying
- Can be tested with RISC Zero's Bonsai documentation
- Supports local LLM integration through Ollama

## Features

- Document ingestion and indexing
- Semantic search capabilities
- Flexible LLM integration (OpenAI or Ollama)
- Flexible embedding models (HuggingFace or Ollama)
- Environment-based configuration
- MCP protocol compliance
- RISC Zero Bonsai documentation support

## Prerequisites

- Python 3.12+
- Ollama (for local LLM support, if using local models)
- Poetry (for dependency management)
- OpenAI API key (if using OpenAI models)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment configuration:
```bash
# Create your .env file from the example
cp .env.example .env
# Edit the .env file with your preferred settings and API keys
```

3. Install and start Ollama (if using local models):
```bash
# Install Ollama
brew install ollama  # for macOS
# or
curl -fsSL https://ollama.com/install.sh | sh  # for Linux

# Start Ollama service
ollama serve
```

4. Pull the required models (if using Ollama):
```bash
# Pull LLM model
ollama pull mistral

# Pull embedding model (if using Ollama embeddings)
ollama pull nomic-embed-text
```

## Configuration

The application uses environment variables for configuration, which can be set in the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider to use (`openai` or `ollama`) | `ollama` |
| `EMBED_PROVIDER` | Embedding provider to use (`huggingface` or `ollama`) | `huggingface` |
| `MODEL_NAME` | Model name for Ollama | `mistral` |
| `OPENAI_MODEL` | Model name for OpenAI | `gpt-4o` |
| `OPENAI_API_KEY` | Your OpenAI API key | - |
| `OPENAI_BASE_URL` | OpenAI API base URL | `https://api.openai.com/v1` |
| `EMBEDDING_MODEL` | Model used for embeddings (depends on provider) | `sentence-transformers/all-MiniLM-L6-v2` or `nomic-embed-text` |
| `DATA_DIR` | Directory for document ingestion | `data` |
| `QUERY` | Default query for testing | - |

### Embedding Model Configuration

The application supports two embedding providers:

1. **HuggingFace** (default): Uses sentence-transformer models for creating embeddings
   ```
   EMBED_PROVIDER=huggingface
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```

2. **Ollama**: Uses Ollama's embedding models (requires Ollama to be installed)
   ```
   EMBED_PROVIDER=ollama
   EMBEDDING_MODEL=nomic-embed-text
   ```

Choose the embedding provider based on your needs:
- HuggingFace embeddings work offline and don't require additional services
- Ollama embeddings may provide better performance for certain use cases

## Usage

1. Start the MCP server:
```bash
poetry run python mcp_server.py
```

2. The server will:
   - Initialize the LLM based on your chosen provider (OpenAI or Ollama)
   - Initialize the embedding model based on your chosen provider (HuggingFace or Ollama)
   - Ingest documents from the data directory
   - Process queries using the RAG workflow

3. Test with RISC Zero Bonsai docs:
   - Place RISC Zero Bonsai documentation in the `data/` directory
   - Query the server about Bonsai features and implementation

## Project Structure

- `mcp_server.py`: Main server implementation
- `rag.py`: RAG workflow implementation
- `data/`: Directory for document ingestion
- `storage/`: Vector store and document storage
- `.env.example`: Example environment configuration
- `.env`: Your local environment configuration (create from example)
- `start_ollama.sh`: Script to start Ollama service

## Testing with RISC Zero Bonsai

The server is configured to work with RISC Zero's Bonsai documentation. You can:
1. Add Bonsai documentation to the `data/` directory
2. Query about Bonsai features, implementation details, and usage
3. Test the RAG workflow with Bonsai-specific questions

## Made with ❤️ by [proofofsid](https://github.com/proofofsid)