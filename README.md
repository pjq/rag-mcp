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
- Local LLM integration
- MCP protocol compliance
- RISC Zero Bonsai documentation support

## Prerequisites

- Python 3.12+
- Ollama (for local LLM support)
- Poetry (for dependency management)

## Installation

1. Install Python dependencies:
```bash
poetry install
```

2. Install and start Ollama:
```bash
# Install Ollama
brew install ollama  # for macOS
# or
curl -fsSL https://ollama.com/install.sh | sh  # for Linux

# Start Ollama service
ollama serve
```

3. Pull the required model:
```bash
ollama pull llama2
```

## Usage

1. Start the MCP server:
```bash
poetry run python mcp_server.py
```

2. The server will:
   - Initialize the LLM and embedding model
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
- `start_ollama.sh`: Script to start Ollama service

## Testing with RISC Zero Bonsai

The server is configured to work with RISC Zero's Bonsai documentation. You can:
1. Add Bonsai documentation to the `data/` directory
2. Query about Bonsai features, implementation details, and usage
3. Test the RAG workflow with Bonsai-specific questions

## Made with ❤️ by [proofofsid](https://github.com/proofofsid)