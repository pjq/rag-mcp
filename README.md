# ZK-MCP (Zero Knowledge - MCP)

A Zero Knowledge proof system using RISC Zero's Bonsai for guest program execution.

## Overview

This project implements a Zero Knowledge proof system that allows for private computation using RISC Zero's Bonsai service. It includes:

- MCP (Model Control Protocol) server implementation
- RAG (Retrieval Augmented Generation) workflow
- Integration with RISC Zero's Bonsai for zkVM guest program execution

## Prerequisites

- Python 3.12+
- Rust and Cargo
- RISC Zero tools:
  - cargo-risczero (v0.20.0)
  - bonsai-cli

## Installation

1. Install Python dependencies:
```bash
poetry install
```

2. Install RISC Zero tools:
```bash
cargo install cargo-risczero --version 0.20.0
cargo install bonsai-cli
```

3. Configure Bonsai:
```bash
bonsai config set api_key <your-api-key>
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

## Project Structure

- `mcp_server.py`: Main server implementation
- `rag.py`: RAG workflow implementation
- `data/`: Directory for document ingestion
- `guest/`: RISC Zero guest program implementation
- `host/`: RISC Zero host program implementation

## License

MIT