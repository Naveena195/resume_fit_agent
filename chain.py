"""
The agent "chain": a sequence of narrow LLM calls, each feeding the next.

This is the piece that turns "a prompt" into "an agent with multi-step
reasoning" -- each step has a single responsibility and passes structured
output forward, and step 4 optionally pulls in retrieved context (RAG).

Swap `call_llm` internals for LangChain's LCEL / a LangGraph graph later
if you want to demonstrate that framework specifically -- the prompt
design and step boundaries stay the same either way.
"""
import json
import os
from openai import OpenAI

from prompts import (
    EXTRACT_RESUME_SKILLS_PROMPT,
    EXTRACT_JD_REQUIREMENTS_PROMPT,
    COMPARE_FIT_PROMPT,
    GENERATE_SUGGESTIONS_PROMPT,
)
from retriever import retrieve_context  # simple FAISS RAG lookup, see retriever.py

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = os.environ.get("AGENT_MODEL", "gpt-4o-mini")


def call_llm(prompt: str) -> dict:
    """Single LLM call that enforces JSON-only output and parses it."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You respond with strict JSON only. No prose, no markdown fences."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    # defensive cleanup in case the model wraps output in ```json fences anyway
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def step1_extract_resume(resume_text: str) -> dict:
    prompt = EXTRACT_RESUME_SKILLS_PROMPT.format(resume_text=resume_text)
    return call_llm(prompt)


def step2_extract_jd(jd_text: str) -> dict:
    prompt = EXTRACT_JD_REQUIREMENTS_PROMPT.format(jd_text=jd_text)
    return call_llm(prompt)


def step3_compare_fit(resume_json: dict, jd_json: dict) -> dict:
    prompt = COMPARE_FIT_PROMPT.format(
        resume_json=json.dumps(resume_json),
        jd_json=json.dumps(jd_json),
    )
    return call_llm(prompt)


def step4_generate_suggestions(fit_json: dict) -> dict:
    # "Tool use" step: the agent decides what to retrieve based on the
    # missing skills found in step 3, then grounds its suggestions in that.
    missing_skill_names = [m["skill"] for m in fit_json.get("missing_skills", [])]
    retrieved_context = retrieve_context(missing_skill_names)

    prompt = GENERATE_SUGGESTIONS_PROMPT.format(
        fit_json=json.dumps(fit_json),
        retrieved_context=json.dumps(retrieved_context),
    )
    return call_llm(prompt)


def run_agent(resume_text: str, jd_text: str) -> dict:
    """Runs the full 4-step chain and returns a combined result."""
    resume_json = step1_extract_resume(resume_text)
    jd_json = step2_extract_jd(jd_text)
    fit_json = step3_compare_fit(resume_json, jd_json)
    suggestions_json = step4_generate_suggestions(fit_json)

    return {
        "resume_analysis": resume_json,
        "jd_analysis": jd_json,
        "fit_analysis": fit_json,
        "suggestions": suggestions_json,
    }
