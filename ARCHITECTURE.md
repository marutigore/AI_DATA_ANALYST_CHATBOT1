# 🏛️ Luminary AI Data Analyst — Enterprise System Architecture & Engineering Reference

## Executive Overview
**Luminary AI Data Analyst** is an enterprise-grade autonomous data science platform designed to combine natural language query processing, RAG (Retrieval-Augmented Generation) tabular document indexing, restricted code execution, multi-provider LLM fallback orchestration, and interactive data visualization.

---

## 🏗️ High-Level System Architecture

```
                                  ┌─────────────────────────────────────────┐
                                  │      Streamlit Frontend (app.py)        │
                                  └────────────────────┬────────────────────┘
                                                       │
                                            HTTP REST / JSON API (Port 8000)
                                                       │
                                  ┌────────────────────▼────────────────────┐
                                  │         FastAPI Backend (api.py)        │
                                  └─┬──────────────────┬──────────────────┬─┘
                                    │                  │                  │
               ┌────────────────────▼──┐   ┌───────────▼──────────┐   ┌───▼──────────────────┐
               │ Multi-Provider LLM    │   │ AST Security Sandbox │   │ In-Memory DuckDB     │
               │ Fallback Engine       │   │ Code Guard           │   │ Query Engine         │
               │ (utils/llm_factory.py)│   │ (utils/sandbox.py)   │   │(utils/query_engine.py│
               └───────────────────────┘   └──────────────────────┘   └──────────────────────┘
                                    │                  │                  │
               ┌────────────────────▼──┐   ┌───────────▼──────────┐   ┌───▼──────────────────┐
               │ Hybrid BM25 + FAISS   │   │ Rate Limiting & TTL  │   │ Executive PDF & Data │
               │ Search Engine         │   │ Middleware           │   │ Exporter Engine      │
               │(utils/hybrid_search.py│   │(utils/rate_limiter)  │   │(utils/report_generator│
               └───────────────────────┘   └──────────────────────┘   └──────────────────────┘
```

---

## 📊 Summary of 30 Enterprise Implementations

| # | Enterprise Feature Module | Primary File(s) | Description |
|---|---------------------------|-----------------|-------------|
| 1 | Restricted AST Code Sandbox Guard | `utils/sandbox.py` | Validates agent code against forbidden modules/builtins using Python AST visitor. |
| 2 | Multi-Provider LLM Fallback Engine | `utils/llm_factory.py` | Orchestrates Gemini 2.5 Flash Lite with automatic fallbacks to Flash & OpenAI. |
| 3 | Per-IP & Per-Session Rate Limiter | `utils/rate_limiter.py` | Fixed-window token bucket middleware protecting endpoints against DoS & quota abuse. |
| 4 | Strict Pydantic v2 API Schemas | `api.py` | Rich request/response validation, Field constraints, and OpenAPI specifications. |
| 5 | Docker Containerization | `Dockerfile`, `docker-compose.yml` | Multi-stage slim Docker image and docker-compose service orchestration. |
| 6 | DuckDB High-Performance Query Engine | `utils/query_engine.py` | Vectorized C++ SQL query execution directly over in-memory Pandas DataFrames. |
| 7 | Automated Data Profiling & Cleaning | `utils/data_cleaner.py` | Currency parsing, string trimming, missingness metrics, and data standardization. |
| 8 | Executive PDF Report Generator | `utils/report_generator.py` | Formatted downloadable executive PDF reports using ReportLab templates. |
| 9 | SQL Database Connector Interface | `utils/db_connector.py` | Native SQLite database connection, table schema discovery, and query loading. |
| 10 | Statistical Hypothesis Testing Engine | `utils/stats_engine.py` | Correlation matrices, two-sample Student's t-tests, and One-way ANOVA calculations. |
| 11 | Hybrid BM25 + FAISS Search Engine | `utils/hybrid_search.py` | Fuses exact keyword BM25 ranks with FAISS dense vector search via RRF. |
| 12 | Hierarchical Parent-Child Chunking | `utils/chunker.py` | Generates high-level dataset parent summary chunks alongside child row chunks. |
| 13 | Hash-Based Vector Store Disk Cache | `utils/vector_cache.py` | Caches pre-computed FAISS embeddings on disk using MD5 file hashes. |
| 14 | Query Rewriting & HyDE Generator | `utils/query_rewriter.py` | Expands user prompts and abbreviations into rich domain-specific search queries. |
| 15 | Dynamic Top-K Retrieval Optimization | `utils/retriever.py` | Dynamically tunes top-k retrieval bounds based on prompt complexity and corpus size. |
| 16 | Multi-Turn Summary Buffer Memory | `utils/memory.py` | Sliding window chat memory with rolling summary compression for long sessions. |
| 17 | Multi-Agent Delegation Pipeline | `utils/agent_team.py` | Routes queries to specialized agent roles (Analyst, Visualization, Executive). |
| 18 | Autonomous One-Click EDA Generator | `utils/eda_engine.py` | Instant distribution summaries, missingness matrices, and duplicate checks. |
| 19 | Guardrails & Prompt Injection Classifier | `utils/guardrails.py` | Regex pattern classifier protecting against jailbreaks and system prompt leaks. |
| 20 | Natural Language Data Export Engine | `utils/exporter.py` | Converts query results to downloadable CSV, Excel (.xlsx), and JSON file streams. |
| 21 | Interactive Dataset Explorer | `utils/ui_components.py` | Streamlit interactive feature explorer with distribution histograms and stats. |
| 22 | Full Chat Session Export & Import | `app.py` | Session JSON export and import restore capabilities for chat history persistence. |
| 23 | Real-Time Server Telemetry Monitor | `utils/ui_components.py` | Live backend API status monitor displaying online status and active session counts. |
| 24 | Custom Theme Switcher Engine | `.streamlit/config.toml`, `utils/ui_components.py` | Dynamic accent theme switcher (Indigo, Emerald Cyber, Midnight Purple). |
| 25 | Step-by-Step Status Streamer | `app.py` | Real-time progress status streamer during agent reasoning and code execution. |
| 26 | GitHub Actions CI/CD Pipeline | `.github/workflows/ci.yml` | Automated build, linting, pytest suite, and Docker container verification. |
| 27 | Structured JSON Logging & Correlation | `config.py` | ELK / Datadog compatible JSON log formatter with correlation timestamps. |
| 28 | Synthetic Benchmark & Load Generator | `tests/benchmark_generator.py` | Synthetic dataset generator for latency stress testing and load benchmarking. |
| 29 | Comprehensive Test Suite Expansion | `test_api.py`, `test_validation.py` | 37 automated unit tests covering all core modules, API routes, and edge cases. |
| 30 | Enterprise Architecture Documentation | `ARCHITECTURE.md` | Comprehensive system design document, endpoint reference, and component sitemap. |

---

## 📡 REST API Endpoint Reference

### 1. `POST /api/upload`
Uploads a CSV or Excel dataset, initializes a session, builds vector embeddings, and returns preview metrics.
- **Request:** `multipart/form-data` with `file`.
- **Response:** `UploadResponse` schema with `session_id`, `metrics`, `preview_json`, and `suggestions`.

### 2. `POST /api/chat`
Submits a natural language analytical prompt for the active session.
- **Request Body:** `{"session_id": "...", "prompt": "..."}`
- **Response:** `ChatResponse` schema with `response` text, optional `plot_json` Plotly figure, and RAG `citations`.

### 3. `GET /api/health`
Returns backend API health status and timestamp.
- **Response:** `{"status": "healthy", "timestamp": 1234567890.0}`

### 4. `GET /api/sessions`
Returns telemetry for all currently active sessions and TTL countdowns.
- **Response:** `{"active_sessions_count": N, "sessions": [...]}`
