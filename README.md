# Study Companion

A local AI agent that answers questions from my own study notes, with
citations, built step by step to learn RAG, LangGraph, and MCP.

## Stack
- Python, Ollama (qwen2.5:7b, nomic-embed-text)
- LangChain, LangGraph, Chroma

## Progress
- [x] Project setup and PDF loader
- [x] Chunking and embeddings
- [ ] Basic RAG with citations
- [ ] LangGraph agent
- [ ] MCP tools
- [ ] Quiz mode and memory

## Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/notes   # add your own PDFs here