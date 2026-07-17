"""
Minimal RAG layer: a tiny local knowledge base of "how to close this skill
gap" notes, embedded and searched with FAISS. This is what lets you
honestly put "vector database" and "RAG" on your resume.

To extend: replace KNOWLEDGE_BASE with real scraped docs (LangChain docs,
HuggingFace course pages, etc.), chunk them, and re-embed.
"""
import os
import numpy as np
import faiss
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"

# Tiny seed knowledge base -- expand this with real notes/docs over time.
KNOWLEDGE_BASE = [
    {"skill": "LangChain", "note": "LangChain lets you chain LLM calls, tools, and memory into agents. Start with a simple LLMChain, then move to an AgentExecutor with tools."},
    {"skill": "HuggingFace Transformers", "note": "Use the `pipeline` API for quick inference (sentiment, summarization). Fine-tuning uses the `Trainer` class with a labeled dataset."},
    {"skill": "Prompt Engineering", "note": "Use clear role instructions, few-shot examples, and explicit output format constraints (e.g. 'respond only in JSON') to get reliable outputs."},
    {"skill": "Vector Databases", "note": "FAISS is a free, local vector index good for small projects. Pinecone/Weaviate are managed, better for production scale."},
    {"skill": "RAG", "note": "Retrieval-Augmented Generation: embed documents, store in a vector DB, retrieve top-k relevant chunks at query time, and pass them into the LLM prompt as context."},
    {"skill": "Fine-tuning", "note": "Fine-tuning adapts a pretrained model to a narrow task using labeled examples; for small projects, prefer prompt engineering or RAG first -- fine-tuning is usually the last resort."},
    {"skill": "Agent frameworks", "note": "LangGraph models an agent as a graph of steps with conditional edges, better suited than plain LangChain chains for loops and branching decisions."},
]

_index = None
_vectors = None


def _embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype="float32")


def _build_index():
    global _index, _vectors
    notes = [item["note"] for item in KNOWLEDGE_BASE]
    _vectors = _embed(notes)
    _index = faiss.IndexFlatL2(_vectors.shape[1])
    _index.add(_vectors)


def retrieve_context(missing_skills: list[str], top_k: int = 3) -> list[dict]:
    """Given missing skill names, retrieve the most relevant knowledge-base notes."""
    if not missing_skills:
        return []
    if _index is None:
        _build_index()

    query = "Skills to explain: " + ", ".join(missing_skills)
    query_vec = _embed([query])
    distances, indices = _index.search(query_vec, top_k)

    return [KNOWLEDGE_BASE[i] for i in indices[0] if i < len(KNOWLEDGE_BASE)]
