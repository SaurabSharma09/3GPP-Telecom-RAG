# 📡 3GPP Telecom RAG

> A grounded Retrieval-Augmented Generation (RAG) system for querying 3GPP 5G specifications using hybrid retrieval, cross-encoder reranking, evidence verification, and Groq-powered LLM generation.

---

## 🚀 Overview

3GPP Telecom RAG is an end-to-end RAG system designed to answer technical questions about 3GPP 5G specifications.

The system works with structured 3GPP specifications and combines:

- DOCX document ingestion
- Structure-aware intelligent chunking
- FAISS semantic retrieval
- BM25 lexical retrieval
- Reciprocal Rank Fusion (RRF)
- Query intent and entity analysis
- Technical entity-aware ranking
- Cross-encoder reranking
- Evidence gating
- Evidence verification
- Groq LLM generation
- Source-aware answers
- Streamlit conversational interface

The primary goal is to generate answers that are **grounded in retrieved 3GPP evidence rather than relying solely on the LLM's parametric knowledge**.

---

## 🎯 Problem

Telecom specifications contain thousands of pages of highly technical information.

A simple semantic search system can fail when:

- exact technical terms matter
- interface names such as `N11`, `N2`, or `N4` are important
- similar concepts occur across multiple sections
- the correct evidence is not the highest semantic match
- an LLM generates an answer without sufficient supporting evidence

This project addresses these problems through a multi-stage retrieval and verification pipeline.

---

# 🏗️ Architecture

```text
                    3GPP DOCX
                        │
                        ▼
              ┌──────────────────┐
              │  DOCX Ingestion  │
              └────────┬─────────┘
                       │
                       ▼
              Structured JSON
                       │
                       ▼
              Intelligent Chunking
                       │
                       ▼
              ┌──────────────────┐
              │   Chunk Store    │
              └────────┬─────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
           FAISS               BM25
       Semantic Search     Keyword Search
              │                 │
              └────────┬────────┘
                       ▼
                RRF Fusion
                       │
                       ▼
              Query Analyzer
                       │
                       ▼
          Entity / Intent / Concept
                       │
                       ▼
              Cross-Encoder
                Reranking
                       │
                       ▼
               Evidence Gate
                       │
                       ▼
             Evidence Verifier
                       │
                       ▼
              Verified Evidence
                       │
                       ▼
                  Groq LLM
                       │
                       ▼
             Grounded Answer
                       │
                       ▼
                Streamlit UI
