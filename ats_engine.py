"""
ats_engine.py

Consolidated ATS Resume Analyzer engine, built across Phases 2-9 of the
AI Resume Analyzer & ATS Score Predictor project.

This module contains NO Streamlit/UI code -- it is pure logic, so it can
be imported by app.py (Streamlit) or used directly in a notebook/script.

Contents:
    - PDF text extraction (Phase 2)
    - Text cleaning (Phase 3)
    - Section splitting (Phase 2/6)
    - Sentence Transformer loading + semantic matching (Phase 4-6)
    - Multi-field skill taxonomy + extraction (Phase 7, expanded)
    - ATS scoring with adaptive weighting (Phase 8, expanded)
    - Recommendations engine (Phase 9)
"""

import re
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.matcher import PhraseMatcher


# ============================================================
# PHASE 2: PDF Extraction
# ============================================================

def extract_text_pymupdf(pdf_path) -> str:
    """Extract raw text from a PDF using PyMuPDF's block-based reading order."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_text_from_pdf(pdf_path) -> str:
    """
    Generic entry point for PDF -> cleaned text.
    Accepts a file path OR a file-like object (e.g. Streamlit's UploadedFile,
    since PyMuPDF's fitz.open() accepts a stream via `stream=` + `filetype=`).
    """
    raw_text = extract_text_pymupdf(pdf_path)
    lines = [line.strip() for line in raw_text.split("\n")]
    lines = [line for line in lines if line != ""]
    return "\n".join(lines)


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """
    Variant of extract_text_from_pdf() for in-memory PDF bytes --
    this is what we'll use in Streamlit, since uploaded files arrive
    as bytes rather than a filesystem path.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line != ""]
    return "\n".join(lines)


# ============================================================
# PHASE 3: Safe Text Cleaning (NOT classic NLP preprocessing --
# see Phase 3 theory: transformers need natural, minimally-processed text)
# ============================================================

def safe_clean_text(text: str) -> str:
    """
    Minimal cleaning appropriate for transformer embeddings.
    Deliberately does NOT lowercase, remove stopwords, or lemmatize.
    """
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line != "")
    return text.strip()


def preprocess_for_embedding(text: str) -> str:
    """Official preprocessing step for all embedding-based operations."""
    return safe_clean_text(text)


# ============================================================
# PHASE 2/6: Section Splitting (rule-based header detection)
# ============================================================

SECTION_PATTERNS_V2 = {
    "experience": re.compile(
        r"^(work\s+)?experience\s*:?\s*$|^professional\s+experience\s*:?\s*$|^employment\s+history\s*:?\s*$",
        re.IGNORECASE,
    ),
    "education": re.compile(r"^education\s*:?\s*$|^academic\s+background\s*:?\s*$", re.IGNORECASE),
    "skills": re.compile(
        r"^(technical\s+)?skills\s*:?(.*)$|^core\s+competencies\s*:?(.*)$", re.IGNORECASE
    ),
    "projects": re.compile(r"^projects\s*:?\s*$|^personal\s+projects\s*:?\s*$", re.IGNORECASE),
    "certifications": re.compile(
        r"^certifications?\s*:?\s*$|^licenses?\s*(and|&)?\s*certifications?\s*:?\s*$", re.IGNORECASE
    ),
}


def split_into_sections_v2(resume_text: str) -> dict:
    """
    Splits resume text into {section_name: section_text}, tolerating
    trailing colons and inline content on the header line
    (e.g. 'Skills: Python, SQL').
    """
    sections = {"header": []}
    current_section = "header"

    for line in resume_text.split("\n"):
        stripped = line.strip()
        matched_section = None
        inline_content = None

        for section_name, pattern in SECTION_PATTERNS_V2.items():
            match = pattern.match(stripped)
            if match:
                matched_section = section_name
                groups = [g for g in match.groups() if g]
                inline_content = groups[0].strip() if groups else None
                break

        if matched_section:
            current_section = matched_section
            sections.setdefault(current_section, [])
            if inline_content:
                sections[current_section].append(inline_content)
        else:
            sections[current_section].append(stripped)

    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


# ============================================================
# PHASE 4: Sentence Transformer Model (lazy-loaded singleton)
# ============================================================

_model = None


def get_model() -> SentenceTransformer:
    """
    Lazily loads and caches the Sentence Transformer model so it's only
    loaded once per process, not once per function call (loading is slow).
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ============================================================
# PHASE 6: Resume Matching Engine
# ============================================================

def compute_match_score(resume_text: str, job_description_text: str) -> dict:
    """
    Computes semantic match between ANY resume text and ANY job description
    text, using section-level embedding + cosine similarity.
    """
    model = get_model()
    sections = split_into_sections_v2(resume_text)
    sections = {name: text for name, text in sections.items() if text.strip() != ""}

    if not sections:
        raise ValueError("No usable text found in resume after section splitting.")

    jd_embedding = model.encode(job_description_text)

    section_scores = {}
    for section_name, section_text in sections.items():
        section_embedding = model.encode(section_text)
        score = cosine_similarity(
            section_embedding.reshape(1, -1), jd_embedding.reshape(1, -1)
        )[0][0]
        section_scores[section_name] = float(score)

    max_section_score = max(section_scores.values())
    avg_section_score = float(np.mean(list(section_scores.values())))

    return {
        "max_section_score": max_section_score,
        "avg_section_score": avg_section_score,
        "section_scores": section_scores,
    }


# ============================================================
# PHASE 7 (EXPANDED): Multi-Field Skill Taxonomy
# ============================================================

# Every field's taxonomy is a dict of category -> list of skill names.
# "General" skills apply regardless of field and are always included.
SKILL_TAXONOMY_BY_FIELD = {
    "General": {
        "soft_skills": [
            "Communication", "Leadership", "Cross-functional Collaboration",
            "Problem Solving", "Teamwork", "Time Management", "Project Management",
            "Critical Thinking", "Adaptability", "Negotiation",
        ],
    },
    "Technology / ML": {
        "programming_languages": [
            "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "R", "SQL", "Scala",
        ],
        "frameworks_libraries": [
            "PyTorch", "TensorFlow", "Scikit-learn", "Keras", "Hugging Face Transformers",
            "React", "Django", "Flask", "FastAPI", "Spring Boot",
        ],
        "cloud": ["AWS", "Azure", "GCP", "Google Cloud"],
        "databases": ["PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch"],
        "tools_devops": ["Docker", "Kubernetes", "Git", "Jenkins", "Terraform", "CI/CD"],
        "ml_nlp_concepts": [
            "Machine Learning", "Deep Learning", "NLP", "Natural Language Processing",
            "Computer Vision", "Sentence Transformers", "BERT", "Data Science",
        ],
    },
    "Healthcare": {
        "clinical_skills": [
            "Patient Care", "Clinical Assessment", "Vital Signs Monitoring", "Phlebotomy",
            "Wound Care", "Medication Administration", "CPR", "Triage",
        ],
        "certifications": [
            "RN", "BLS", "ACLS", "CPR Certified", "CNA", "EMT",
        ],
        "systems": ["EHR", "Epic", "Cerner", "HIPAA Compliance"],
        "specialties": [
            "Pediatrics", "Oncology", "Cardiology", "ICU", "Emergency Medicine", "Geriatrics",
        ],
    },
    "Marketing": {
        "digital_marketing": [
            "SEO", "SEM", "Google Ads", "Facebook Ads", "Content Marketing", "Email Marketing",
            "Social Media Marketing", "Marketing Automation",
        ],
        "analytics_tools": [
            "Google Analytics", "HubSpot", "Mailchimp", "Hootsuite", "Tableau", "SEMrush",
        ],
        "skills": ["Brand Strategy", "Copywriting", "Market Research", "A/B Testing", "Campaign Management"],
    },
    "Finance": {
        "core_skills": [
            "Financial Modeling", "Financial Analysis", "Forecasting", "Budgeting",
            "Risk Management", "Valuation", "Bookkeeping", "Auditing",
        ],
        "tools": ["Excel", "QuickBooks", "SAP", "Bloomberg Terminal", "Power BI"],
        "certifications": ["CFA", "CPA", "FRM"],
        "domains": ["Investment Banking", "Equity Research", "Portfolio Management", "Tax Compliance"],
    },
    "Sales": {
        "core_skills": [
            "Lead Generation", "Cold Calling", "Negotiation", "Account Management",
            "Client Relationship Management", "Sales Forecasting", "Closing", "Upselling",
        ],
        "tools": ["Salesforce", "HubSpot CRM", "Zoho CRM", "Outreach", "LinkedIn Sales Navigator"],
    },
    "Legal": {
        "core_skills": [
            "Legal Research", "Contract Drafting", "Litigation", "Due Diligence",
            "Compliance", "Legal Writing", "Case Management", "Negotiation",
        ],
        "certifications": ["JD", "Bar Admission", "Paralegal Certification"],
        "domains": ["Corporate Law", "Intellectual Property", "Employment Law", "Real Estate Law"],
    },
}

# Manual synonym normalization -- maps alternate spellings/abbreviations
# to their canonical taxonomy form.
SKILL_SYNONYMS = {
    "js": "JavaScript",
    "nlp": "NLP",
    "natural language processing": "NLP",
    "google cloud": "GCP",
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "rn": "RN",
    "cpr certified": "CPR Certified",
    "seo": "SEO",
    "sem": "SEM",
    "crm": "Salesforce",
}

AVAILABLE_FIELDS = [f for f in SKILL_TAXONOMY_BY_FIELD.keys() if f != "General"]


def get_skill_list_for_field(field: str) -> list:
    """
    Returns the flattened skill list for a given field, always including
    the General/soft-skills category regardless of field chosen.
    """
    taxonomy = SKILL_TAXONOMY_BY_FIELD.get(field, {})
    general = SKILL_TAXONOMY_BY_FIELD["General"]

    skills = []
    for category_skills in taxonomy.values():
        skills.extend(category_skills)
    for category_skills in general.values():
        skills.extend(category_skills)

    return list(dict.fromkeys(skills))  # de-duplicate while preserving order


_nlp = None


def get_spacy_nlp():
    """Lazily loads and caches the spaCy pipeline."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def build_matcher_for_field(field: str):
    """
    Builds a spaCy PhraseMatcher for the given field's skill list.
    Rebuilt per field/request since taxonomies are small -- this is cheap.
    """
    nlp = get_spacy_nlp()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    skill_list = get_skill_list_for_field(field)
    patterns = [nlp.make_doc(skill) for skill in skill_list]
    matcher.add("SKILL", patterns)

    lowercase_to_canonical = {skill.lower(): skill for skill in skill_list}
    lowercase_to_canonical.update(SKILL_SYNONYMS)

    return matcher, lowercase_to_canonical


def extract_skills(text: str, field: str = "Technology / ML") -> set:
    """
    Extracts a deduplicated set of canonical skill names found in `text`,
    using the skill taxonomy for the specified field (plus General skills).
    Works on ANY text -- resume or job description, ANY supported field.
    """
    nlp = get_spacy_nlp()
    matcher, lowercase_to_canonical = build_matcher_for_field(field)

    doc = nlp(text)
    matches = matcher(doc)

    found_skills = set()
    for match_id, start, end in matches:
        span_text = doc[start:end].text
        canonical = lowercase_to_canonical.get(span_text.lower(), span_text)
        found_skills.add(canonical)

    return found_skills


def suggest_field(resume_text: str, job_description_text: str) -> str:
    """
    Heuristic auto-detection of the best-fitting field: tries every field's
    taxonomy against the JD text and picks whichever field found the most
    skill matches. Falls back to 'Technology / ML' if nothing matches well
    (reasonable default given this project's original scope).
    """
    best_field = "Technology / ML"
    best_count = 0

    for field in AVAILABLE_FIELDS:
        jd_skills = extract_skills(job_description_text, field=field)
        if len(jd_skills) > best_count:
            best_count = len(jd_skills)
            best_field = field

    return best_field


# ============================================================
# PHASE 8: Experience / Education / Formatting Helpers
# ============================================================

def extract_years_experience(text: str) -> int:
    """Finds the maximum stated 'years of experience' mentioned in text."""
    pattern = re.compile(r"(\d+)\+?\s*-?\s*(?:\d+)?\s*years?", re.IGNORECASE)
    matches = pattern.findall(text)
    if not matches:
        return 0
    return max(int(m) for m in matches)


EDUCATION_LEVELS = {
    "phd": 4,
    "doctorate": 4,
    "md": 4,
    "master": 3,
    "m.s.": 3,
    "mba": 3,
    "bachelor": 2,
    "b.s.": 2,
    "b.a.": 2,
    "associate": 1,
    "diploma": 1,
}


def extract_education_level(text: str) -> int:
    """Returns the highest education level found in text. 0 if none recognized."""
    text_lower = text.lower()
    found_levels = [level for keyword, level in EDUCATION_LEVELS.items() if keyword in text_lower]
    return max(found_levels) if found_levels else 0


def compute_formatting_score(resume_text: str) -> float:
    """
    Proxy for 'is this resume ATS-parseable': checks how many expected
    sections were successfully detected. Returns a score from 0.0 to 1.0.
    """
    sections = split_into_sections_v2(resume_text)
    detected = [name for name, content in sections.items() if name != "header" and content.strip() != ""]
    expected_section_count = 3
    return min(len(detected) / expected_section_count, 1.0)


# ============================================================
# PHASE 8 (EXPANDED): ATS Scoring with Adaptive Weighting
# ============================================================

BASE_WEIGHTS = {
    "semantic": 0.40,
    "skills": 0.30,
    "experience": 0.15,
    "education": 0.10,
    "formatting": 0.05,
}

# If the taxonomy finds fewer than this many skills combined (resume + JD),
# we treat the skills signal as unreliable for this field/text and
# redistribute its weight into semantic similarity instead.
LOW_SKILL_SIGNAL_THRESHOLD = 2


def compute_adaptive_weights(resume_skills: set, jd_skills: set) -> dict:
    """
    Returns a weight dict. If very few skills were detected on either side
    (a sign the taxonomy doesn't cover this resume's domain well), shifts
    the 'skills' weight into 'semantic' so under-covered fields aren't
    unfairly penalized just because our dictionary is incomplete for them.
    """
    weights = dict(BASE_WEIGHTS)
    total_detected = len(resume_skills) + len(jd_skills)

    if total_detected < LOW_SKILL_SIGNAL_THRESHOLD:
        shifted = weights["skills"]
        weights["skills"] = 0.05  # keep a small nonzero weight rather than dropping entirely
        weights["semantic"] += shifted - 0.05

    return weights


def compute_ats_score(resume_text: str, job_description_text: str, field: str = "Technology / ML") -> dict:
    """
    Computes the final weighted ATS Match Score (0-100) for ANY resume
    and ANY job description in the specified field.
    """
    match_result = compute_match_score(resume_text, job_description_text)
    semantic_score = max(match_result["avg_section_score"], 0)

    resume_skills = extract_skills(resume_text, field=field)
    jd_skills = extract_skills(job_description_text, field=field)

    weights = compute_adaptive_weights(resume_skills, jd_skills)

    if jd_skills:
        skills_score = len(resume_skills & jd_skills) / len(jd_skills)
    else:
        skills_score = 1.0

    resume_years = extract_years_experience(resume_text)
    jd_years_required = extract_years_experience(job_description_text)
    experience_score = 1.0 if jd_years_required == 0 else min(resume_years / jd_years_required, 1.0)

    resume_edu = extract_education_level(resume_text)
    jd_edu_required = extract_education_level(job_description_text)
    if jd_edu_required == 0:
        education_score = 1.0
    else:
        education_score = 1.0 if resume_edu >= jd_edu_required else resume_edu / jd_edu_required

    formatting_score = compute_formatting_score(resume_text)

    final_score = (
        semantic_score * weights["semantic"]
        + skills_score * weights["skills"]
        + experience_score * weights["experience"]
        + education_score * weights["education"]
        + formatting_score * weights["formatting"]
    )

    return {
        "field_used": field,
        "weights_used": {k: round(v, 3) for k, v in weights.items()},
        "final_ats_score": round(final_score * 100, 1),
        "breakdown": {
            "semantic_score": round(semantic_score * 100, 1),
            "skills_score": round(skills_score * 100, 1),
            "experience_score": round(experience_score * 100, 1),
            "education_score": round(education_score * 100, 1),
            "formatting_score": round(formatting_score * 100, 1),
        },
        "details": {
            "resume_years": resume_years,
            "jd_years_required": jd_years_required,
            "resume_education_level": resume_edu,
            "jd_education_level_required": jd_edu_required,
            "matched_skills": sorted(resume_skills & jd_skills),
            "missing_skills": sorted(jd_skills - resume_skills),
        },
    }


# ============================================================
# PHASE 9: Recommendations Engine
# ============================================================

THRESHOLDS = {
    "section_score_low": 0.35,
    "section_score_very_low": 0.20,
    "section_score_strong": 0.55,
    "skills_score_low": 0.5,
    "formatting_score_low": 0.7,
}


def recommend_missing_skills(ats_result: dict) -> list:
    missing = ats_result["details"]["missing_skills"]
    recommendations = []

    if not missing:
        recommendations.append("No missing skills detected -- great alignment with this job description.")
        return recommendations

    skills_list = ", ".join(sorted(missing))
    recommendations.append(
        f"Consider adding or highlighting the following skills if you have experience with them: {skills_list}."
    )

    if ats_result["breakdown"]["skills_score"] < THRESHOLDS["skills_score_low"] * 100:
        recommendations.append(
            "Your skills overlap with this job description is significantly low. "
            "Prioritize tailoring your resume to explicitly mention the required tools and technologies."
        )

    return recommendations


def recommend_section_improvements(match_result: dict) -> list:
    """
    Distinguishes moderate gaps (likely phrasing) from very low gaps
    (likely a genuine experience/substance gap) -- see Phase 9 discussion.
    """
    recommendations = []
    section_scores = match_result["section_scores"]

    for section_name, score in section_scores.items():
        if score < THRESHOLDS["section_score_very_low"]:
            recommendations.append(
                f"Your '{section_name}' section shows very limited alignment with this role's core "
                f"requirements (score: {score:.2f}). This likely reflects a genuine experience gap rather "
                f"than a wording issue -- consider whether this role is a strong fit, or focus on gaining "
                f"relevant experience before applying."
            )
        elif score < THRESHOLDS["section_score_low"]:
            recommendations.append(
                f"Your '{section_name}' section has moderate relevance to this job description "
                f"(score: {score:.2f}). Consider rewriting it to more directly reflect the language "
                f"and priorities used in the job posting."
            )

    if not recommendations:
        recommendations.append("All resume sections show reasonable relevance to this job description.")

    return recommendations


def identify_strengths(ats_result: dict, match_result: dict) -> list:
    strengths = []

    matched = ats_result["details"]["matched_skills"]
    if matched:
        strengths.append(f"Strong alignment on key skills: {', '.join(sorted(matched))}.")

    for section_name, score in match_result["section_scores"].items():
        if score >= THRESHOLDS["section_score_strong"]:
            strengths.append(f"Your '{section_name}' section is a strong match for this role (score: {score:.2f}).")

    if ats_result["details"]["resume_years"] >= ats_result["details"]["jd_years_required"]:
        strengths.append(
            f"You meet or exceed the required experience "
            f"({ats_result['details']['resume_years']} years vs {ats_result['details']['jd_years_required']} required)."
        )

    if not strengths:
        strengths.append("No standout strengths detected relative to this specific job description.")

    return strengths


def identify_weaknesses(ats_result: dict) -> list:
    weaknesses = []
    details = ats_result["details"]
    breakdown = ats_result["breakdown"]

    if details["resume_years"] < details["jd_years_required"]:
        gap = details["jd_years_required"] - details["resume_years"]
        weaknesses.append(
            f"You have {details['resume_years']} years of experience; "
            f"this role requests {details['jd_years_required']}+ years ({gap} year(s) short)."
        )

    if details["resume_education_level"] < details["jd_education_level_required"]:
        weaknesses.append("Your stated education level is below what this job description specifies.")

    if breakdown["formatting_score"] < THRESHOLDS["formatting_score_low"] * 100:
        weaknesses.append(
            "Your resume's section structure wasn't fully detected -- consider using clearer, "
            "standard section headers (e.g., 'Experience', 'Education', 'Skills') to improve ATS parseability."
        )

    if details["missing_skills"]:
        weaknesses.append(
            f"Missing {len(details['missing_skills'])} skill(s) mentioned in the job description: "
            f"{', '.join(sorted(details['missing_skills']))}."
        )

    if not weaknesses:
        weaknesses.append("No significant weaknesses detected relative to this job description.")

    return weaknesses


def generate_recommendations(resume_text: str, job_description_text: str, field: str = "Technology / ML") -> dict:
    """
    Runs the full pipeline and returns a complete recommendations report
    for ANY resume, ANY job description, and ANY supported field.
    """
    ats_result = compute_ats_score(resume_text, job_description_text, field=field)
    match_result = compute_match_score(resume_text, job_description_text)

    return {
        "field_used": ats_result["field_used"],
        "final_ats_score": ats_result["final_ats_score"],
        "breakdown": ats_result["breakdown"],
        "details": ats_result["details"],
        "strengths": identify_strengths(ats_result, match_result),
        "weaknesses": identify_weaknesses(ats_result),
        "missing_skills_recommendations": recommend_missing_skills(ats_result),
        "section_recommendations": recommend_section_improvements(match_result),
    }
