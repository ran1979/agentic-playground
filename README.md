# agentic-playground

Small, self-contained experiments with local LLM agents using [Ollama](https://ollama.com), LangChain and LangGraph. Everything runs locally — no API keys needed.

## Files

| File | What it shows |
|------|---------------|
| `translate.py` | Minimal LangChain chain: prompt → `ChatOllama` → string parser (English → Italian). |
| `rag_example.py` | Basic RAG: load a text file, chunk it, embed with `nomic-embed-text`, store in Chroma, answer from retrieved context. |
| `native_agent.py` | Tool-calling agent with a raw loop using the `ollama` client directly. Tool: live weather from wttr.in. |
| `native_agent.md` | Walkthrough + sequence diagram of the native agent loop. |
| `native_agent_with_tracker_and_guardrails.py` | Native agent plus token/cost tracking, input-prompt and tool-arg guardrails, and a max-iteration cap. |
| `langgraph_agent.py` | Same weather agent built with LangGraph (`StateGraph`, `ToolNode`, `tools_condition`). |
| `langgraph_vs_raw_loop.md` | Sequence diagram of the LangGraph flow, for comparison with the raw loop. |

## Setup

1. Install and start [Ollama](https://ollama.com), then pull the models:
   ```bash
   ollama pull llama3
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
2. Install Python deps:
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

| Package | What it does | Why it's needed |
|---------|--------------|-----------------|
| `ollama` | Official Python client for the local Ollama server. | Used directly by the native agents (`ollama.chat` with tool calling). |
| `langchain-core` | LangChain base abstractions: prompts, output parsers, runnables, `@tool`, messages. | Foundation for every LangChain/LangGraph example (the `\|` pipe chains). |
| `langchain-community` | Community integrations for LangChain. | Provides `TextLoader` for loading the RAG document. |
| `langchain-text-splitters` | Text chunking utilities. | `RecursiveCharacterTextSplitter` splits the document before embedding in `rag_example.py`. |
| `langchain-ollama` | LangChain wrappers for Ollama. | `ChatOllama` (chat model) and `OllamaEmbeddings` (vectors) in the chain, RAG and LangGraph examples. |
| `langchain-chroma` | LangChain wrapper for the Chroma vector store. | Stores and searches embeddings for retrieval in `rag_example.py`. |
| `chromadb` | The Chroma vector database itself. | Backend that `langchain-chroma` runs on. |
| `langgraph` | Graph-based agent orchestration (`StateGraph`, `ToolNode`). | Builds the agent loop in `langgraph_agent.py`. |

Each example is standalone on purpose, so the weather tool is repeated across the agent files.

## Run

```bash
python translate.py
python rag_example.py
python native_agent.py
python native_agent_with_tracker_and_guardrails.py
python langgraph_agent.py
```
