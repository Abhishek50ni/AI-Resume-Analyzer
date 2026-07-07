# AI Resume Analyzer & ATS Score Predictor

An AI-powered resume analyzer that goes beyond keyword matching. It parses resume PDFs, computes **semantic similarity** against a job description using Sentence Transformer embeddings, and produces an explainable ATS compatibility score — along with skill-gap analysis and personalized improvement recommendations.

**Live Demo:** [Add your Streamlit Community Cloud URL here]
**Repository:** https://github.com/Abhishek50ni/AI-Resume-Analyzer

---

## Why This Project Exists

Most "ATS checkers" rely on literal keyword matching — if your resume says "ML" but the job description says "machine learning," a naive system scores that as zero overlap, even though they mean the same thing. This project uses **transformer-based sentence embeddings** to understand *meaning*, not just word overlap, so paraphrased or differently-worded experience is still recognized as relevant.

---

## Features

- 📄 **PDF Resume Upload** — supports single or **batch upload** (multiple resumes ranked against one job description)
- 🧠 **Semantic Matching** — section-level embeddings via Sentence Transformers (`all-MiniLM-L6-v2`), compared using cosine similarity
- 🎯 **Explainable ATS Score** — a weighted combination of semantic similarity, skill overlap, experience match, education match, and resume formatting/parseability
- 🏷️ **Multi-Field Skill Taxonomy** — Technology/ML, Healthcare, Marketing, Finance, Sales, and Legal, each with its own curated skill dictionary
- ⚖️ **Adaptive Weighting** — automatically reduces reliance on skill-matching when the taxonomy doesn't confidently cover the resume's domain, so the semantic score (which works regardless of field) carries more weight instead
- 🏆 **Candidate Ranking** — upload multiple resumes against one job description and get a ranked Top-N shortlist
- 📊 **Interactive Dashboard** — score gauge, radar chart breakdown, skill coverage bars, and a word cloud
- 💡 **Personalized Recommendations** — missing skills, section-level rewrite suggestions, strengths, and weaknesses, each grounded in an actual computed number (not generic advice)
- 📥 **Downloadable Report** — full analysis exportable as a text report

---

## How It Works

```
Resume PDF ──► PyMuPDF text extraction ──► Section splitting (regex heuristics)
                                                    │
Job Description ──────────────────────┐            │
                                       ▼            ▼
                          Sentence Transformer embeddings (per section / JD)
                                       │
                                       ▼
                       Cosine similarity → semantic_score
                                       │
        ┌──────────────────────────────┼───────────────────────────┐
        ▼                              ▼                           ▼
  Skill extraction              Experience/Education           Formatting
  (spaCy PhraseMatcher,          (regex heuristics)          (section detection
   multi-field taxonomy)                                       completeness)
        │                              │                           │
        └──────────────┬───────────────┴───────────────────────────┘
                        ▼
         Adaptive weighted combination → Final ATS Score (0–100)
                        │
                        ▼
          Recommendations Engine (rule-based, threshold-driven)
```

### The ATS Score Formula

```
Final Score = (semantic × 0.40) + (skills × 0.30) + (experience × 0.15)
            + (education × 0.10) + (formatting × 0.05)
```

Weights are **adaptively rebalanced** at runtime: if the skill taxonomy detects little or nothing in the job description (a sign the selected field doesn't match the content well), its weight is reduced and shifted to the semantic score, which doesn't depend on any hand-curated dictionary.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Semantic embeddings | Sentence Transformers (`all-MiniLM-L6-v2`), PyTorch, Hugging Face Transformers |
| Skill extraction | spaCy (`PhraseMatcher`) |
| PDF parsing | PyMuPDF |
| Similarity computation | scikit-learn (cosine similarity) |
| Visualization | Plotly, Matplotlib, WordCloud |
| Web app | Streamlit |
| Data handling | NumPy, Pandas |

---

## Project Structure

```
ats_project/
├── app.py                  # Streamlit UI and orchestration
├── ats_engine.py            # Core logic: parsing, embeddings, scoring, recommendations
├── evaluate.py               # Evaluation script (multi-field test suite)
├── requirements.txt
└── .streamlit/
    └── config.toml           # Dark theme configuration
```

`ats_engine.py` contains no UI code — it's a pure, importable module, so the same logic that powers the Streamlit app is directly testable via `evaluate.py`.

---

## Running Locally

```bash
git clone https://github.com/Abhishek50ni/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

The first run downloads the `all-MiniLM-L6-v2` model (~90MB, one-time, requires internet) and caches it locally.

### Running the Evaluation Suite

```bash
python evaluate.py
```

Runs the engine against multiple resume/job-description pairs spanning different fields, including a deliberate field-mismatch test, and prints a pass/fail validation summary.

---

## Known Limitations

- **Semantic scores are relative, not a literal "percent match."** Because it compares dense, multi-topic resume sections against dense, multi-topic job descriptions, even strong matches typically score in the 0.4–0.7 range rather than near 1.0 — the number should be read comparatively, not as an absolute percentage.
- **Skill taxonomies are hand-curated and finite.** A resume using terminology outside the selected field's dictionary may show an artificially low skill-match score even if genuinely qualified. Adaptive weighting reduces this risk but does not eliminate it.
- **Field selection is manual.** If the wrong field is selected, skill-matching accuracy drops (the app does not currently auto-detect or validate the chosen field against the content).
- **Experience/education extraction uses regex heuristics.** Unusual phrasing (e.g., "half a decade of experience" instead of "5 years") won't be recognized and will default to 0.
- **Formatting score is a lightweight proxy.** It checks whether standard sections were detected, but doesn't catch every real ATS-breaking issue (e.g., text embedded in images, complex multi-column layouts).
- **Scanned/image-based PDFs are not supported.** Text extraction requires a text-based PDF; OCR is not currently implemented.

---

## Possible Future Improvements

- OCR support for scanned/image-based resumes
- User-editable/extensible skill taxonomy per field
- Matched/missing keyword highlighting directly within the resume text
- CSV export of batch ranking results
- Reverse mode: one resume vs. multiple job descriptions
- Confidence indicator surfaced in the UI when adaptive weighting is triggered

---

## License

This project is open source and available for learning and personal use.
