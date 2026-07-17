# Resume ⇄ Job Description Fit Agent

A multi-step LLM agent that evaluates how well a resume matches a job description,
using chained reasoning steps (extract → extract → compare → suggest) and a small
RAG layer to ground its improvement suggestions.

## Why this project (for your resume)

This mirrors what an AI hiring platform (like the one you're applying to) actually
builds: an agent that reasons over unstructured text, uses tools/retrieval, and
returns structured, actionable output. It legitimately demonstrates:

- **LLM API usage** (OpenAI)
- **Prompt engineering** (strict JSON-output prompts per step)
- **Multi-step / chained reasoning** (4 distinct steps, each feeding the next)
- **RAG + vector database** (FAISS-backed retrieval in step 4)
- **Agentic tool-use pattern** (step 4 decides what to retrieve based on step 3's output)

## Project structure

```
resume_fit_agent/
├── app.py         # Flask API entrypoint (POST /analyze)
├── chain.py        # The 4-step agent chain
├── prompts.py       # Prompt templates, one per step
├── retriever.py      # Tiny FAISS-based RAG knowledge base
└── requirements.txt
```

## The chain (this is the "agent" part)

1. **Extract resume skills** — LLM parses the resume into structured JSON (skills, projects, certs).
2. **Extract JD requirements** — LLM parses the job description into structured JSON (required/bonus skills).
3. **Compare fit** — LLM compares the two structured outputs, scores match %, lists missing skills.
4. **Generate suggestions** — Given the missing skills, the agent *retrieves* relevant notes from a
   FAISS vector index, then asks the LLM to generate a grounded improvement plan.

Each step is a separate, narrow LLM call rather than one giant prompt. That's what
makes this "multi-step reasoning" instead of just prompting — and it's also what
makes debugging and demoing it much easier (you can show/print each intermediate
JSON output).

## Setup

```bash
cd resume_fit_agent
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."       # or use a free alternative, see below
python app.py
```

Test it:

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "...", "jd_text": "..."}'
```

## No OpenAI budget? Free alternatives

Swap the `OpenAI` client in `chain.py`/`retriever.py` for any OpenAI-compatible
free-tier API — the code barely changes since they share the same SDK interface:

- **Groq** (fast, generous free tier, Llama/Mixtral models) — just change `base_url`
  to `https://api.groq.com/openai/v1` and the model name.
- **Google Gemini** — free tier via `google-generativeai` SDK (slightly different
  interface, small adapter needed).
- **Local models via Ollama** — fully free, run a small model like `llama3.1:8b`
  locally, point `base_url` to `http://localhost:11434/v1`.

## Extension ideas (for a stronger resume bullet)

- **Swap in LangChain**: rewrite `chain.py` using `LLMChain` / `RunnableSequence`
  and `AgentExecutor` with tools — same logic, but now you can *also* say
  "LangChain" on your resume, since that's explicitly named in most AI-agent JDs.
- **Swap in LangGraph**: model the 4 steps as a graph with conditional edges (e.g.,
  re-run step 1 if resume parsing confidence is low) — demonstrates branching/looping
  agent behavior, not just a linear pipeline.
- **Add a real tool call**: let step 4 call a live web search tool when it needs to
  explain an unfamiliar skill, instead of only using the local FAISS knowledge base.
- **Frontend**: wrap this in a simple React form (resume paste box + JD paste box +
  results view) — you already have this stack from your Campus Notes project.

## Suggested resume bullet

```
AI Resume-Job Fit Agent | Python, OpenAI API, FAISS, Flask
Built a multi-step LLM agent that evaluates resume-job fit through chained
reasoning (skill extraction, requirement extraction, gap analysis, suggestion
generation), using FAISS-based retrieval to ground recommendations and reduce
hallucination.
```
