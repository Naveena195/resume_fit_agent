"""
Prompt templates for the multi-step Resume <-> JD Fit Agent.

Each step has ONE job and returns STRICT JSON. Keeping steps narrow is
what makes this "multi-step reasoning" rather than one giant prompt --
it's also what makes the output reliable enough to parse programmatically.
"""

EXTRACT_RESUME_SKILLS_PROMPT = """You are a resume parser. Extract structured data from the resume below.

Return ONLY valid JSON (no markdown fences, no commentary) in this exact shape:
{{
  "skills": ["skill1", "skill2", ...],
  "languages": ["..."],
  "frameworks": ["..."],
  "tools": ["..."],
  "projects": [
    {{"name": "...", "tech": ["..."], "summary": "..."}}
  ],
  "experience_years": <number, estimate if not explicit>,
  "certifications": ["..."]
}}

RESUME:
{resume_text}
"""

EXTRACT_JD_REQUIREMENTS_PROMPT = """You are a job description parser. Extract structured requirements.

Return ONLY valid JSON in this exact shape:
{{
  "required_skills": [{{"skill": "...", "importance": "High|Medium|Low"}}],
  "bonus_skills": ["..."],
  "responsibilities": ["..."],
  "min_experience": "<string, e.g. 'entry-level' or '2+ years'>"
}}

JOB DESCRIPTION:
{jd_text}
"""

COMPARE_FIT_PROMPT = """You are an ATS + technical recruiter. Compare the candidate's extracted
skills against the job's extracted requirements. Be strict and evidence-based --
do not assume skills that are not explicitly listed.

CANDIDATE_SKILLS_JSON:
{resume_json}

JOB_REQUIREMENTS_JSON:
{jd_json}

Return ONLY valid JSON in this exact shape:
{{
  "overall_match_percent": <0-100>,
  "matched_skills": ["..."],
  "missing_skills": [{{"skill": "...", "importance": "High|Medium|Low"}}],
  "strengths": ["..."],
  "gaps": ["..."]
}}
"""

GENERATE_SUGGESTIONS_PROMPT = """You are a career coach. Using the fit analysis below, and the
retrieved context (if any) about how to close specific skill gaps, generate a short,
prioritized improvement plan.

FIT_ANALYSIS_JSON:
{fit_json}

RETRIEVED_CONTEXT (may be empty):
{retrieved_context}

Return ONLY valid JSON in this exact shape:
{{
  "top_improvements": [
    {{"action": "...", "why_it_matters": "...", "difficulty": "Easy|Medium|Hard"}}
  ],
  "recommended_projects": ["..."]
}}
"""
