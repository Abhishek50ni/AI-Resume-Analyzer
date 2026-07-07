"""
evaluate.py -- Phase 12: Evaluation

Run this from inside your ats_project/ folder (same folder as ats_engine.py):

    python evaluate.py

This imports the REAL ats_engine.py that powers your deployed Streamlit app --
not a separate notebook copy -- so what we validate here is exactly what
your users experience. This matters: back in Phase 6, a notebook-only
section-splitting bug silently produced wrong results. Testing the actual
production module avoids that class of drift entirely.

What this script does:
    1. Runs several resume/JD pairs across DIFFERENT fields (not just tech)
       to test whether the pipeline generalizes, per the concern raised
       when we built the multi-field taxonomy.
    2. Validates expected orderings (a strong match should outscore a weak
       match against the same JD).
    3. Tests the adaptive-weighting fallback directly (Healthcare resume
       scored with the Technology / ML taxonomy -- a deliberate mismatch).
    4. Tests edge cases: an empty-ish resume, a resume with no skills at all.
    5. Prints a summary table and a written discussion of limitations.
"""

import ats_engine as engine

print("Loading models (one-time cost)...")
engine.get_model()
engine.get_spacy_nlp()
print("Models loaded.\n")


# ============================================================
# 1. Test data: multiple resumes and JDs across DIFFERENT fields
# ============================================================

test_cases = {
    "tech_strong": {
        "field": "Technology / ML",
        "jd": """
            We are hiring a Machine Learning Engineer.
            Requirements: 3+ years experience with Python and PyTorch,
            NLP and transformer-based models, AWS or GCP, Docker and Kubernetes.
            Bachelor's or Master's degree in Computer Science required.
        """,
        "resume": """
            Jane ML
            Experience:
            - 5 years building deep learning models using PyTorch and Hugging Face Transformers
            - Deployed NLP pipelines on AWS using Docker and Kubernetes
            Skills: Python, PyTorch, NLP, AWS, Docker, Kubernetes
            Education:
            Master's degree in Computer Science
        """,
    },
    "tech_weak": {
        "field": "Technology / ML",
        "jd": """
            We are hiring a Machine Learning Engineer.
            Requirements: 3+ years experience with Python and PyTorch,
            NLP and transformer-based models, AWS or GCP, Docker and Kubernetes.
            Bachelor's or Master's degree in Computer Science required.
        """,
        "resume": """
            Sam Sales
            Experience:
            - 2 years in retail sales and customer relationship management
            Skills: Communication, Negotiation, Salesforce
            Education:
            Bachelor's degree in Business
        """,
    },
    "healthcare_strong": {
        "field": "Healthcare",
        "jd": """
            We are hiring a Registered Nurse for our ICU department.
            Requirements: RN license, BLS and ACLS certification,
            3+ years of ICU patient care experience, familiarity with Epic EHR system.
            Bachelor's degree in Nursing required.
        """,
        "resume": """
            Maria Nurse
            Experience:
            - 4 years providing ICU patient care including vital signs monitoring and triage
            - Proficient with Epic EHR system for clinical documentation
            Skills: RN, BLS, ACLS, Patient Care, Epic, Triage
            Education:
            Bachelor's degree in Nursing
        """,
    },
    "healthcare_scored_as_tech": {
        # DELIBERATE MISMATCH TEST: same healthcare resume/JD, but scored
        # using the Technology / ML taxonomy -- this should trigger our
        # adaptive weighting fallback (Phase 8), since the tech taxonomy
        # won't recognize almost any of these terms.
        "field": "Technology / ML",
        "jd": """
            We are hiring a Registered Nurse for our ICU department.
            Requirements: RN license, BLS and ACLS certification,
            3+ years of ICU patient care experience, familiarity with Epic EHR system.
            Bachelor's degree in Nursing required.
        """,
        "resume": """
            Maria Nurse
            Experience:
            - 4 years providing ICU patient care including vital signs monitoring and triage
            - Proficient with Epic EHR system for clinical documentation
            Skills: RN, BLS, ACLS, Patient Care, Epic, Triage
            Education:
            Bachelor's degree in Nursing
        """,
    },
    "marketing_strong": {
        "field": "Marketing",
        "jd": """
            We are hiring a Digital Marketing Specialist.
            Requirements: 2+ years experience with SEO, Google Ads, and content marketing,
            familiarity with Google Analytics and HubSpot.
            Bachelor's degree preferred.
        """,
        "resume": """
            Alex Marketer
            Experience:
            - 3 years running SEO and Google Ads campaigns for e-commerce clients
            - Built content marketing strategy using HubSpot and Google Analytics
            Skills: SEO, Google Ads, Content Marketing, HubSpot, Google Analytics
            Education:
            Bachelor's degree in Marketing
        """,
    },
    "edge_case_sparse_resume": {
        # EDGE CASE: an extremely short, low-content resume
        "field": "Technology / ML",
        "jd": """
            We are hiring a Machine Learning Engineer.
            Requirements: 3+ years experience with Python and PyTorch.
        """,
        "resume": """
            John Minimal
            Skills: Python
        """,
    },
}


# ============================================================
# 2. Run every test case and collect results
# ============================================================

results = {}

for name, case in test_cases.items():
    report = engine.generate_recommendations(case["resume"], case["jd"], field=case["field"])
    results[name] = report

print("=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Test Case':<30}{'Field':<18}{'Score':<8}{'Weights Adjusted?'}")
print("-" * 70)

for name, report in results.items():
    weights = report["breakdown"]  # breakdown doesn't show weights directly; check via details
    # Re-derive whether adaptive weighting kicked in by recomputing skills
    case = test_cases[name]
    resume_skills = engine.extract_skills(case["resume"], field=case["field"])
    jd_skills = engine.extract_skills(case["jd"], field=case["field"])
    adjusted = (len(resume_skills) + len(jd_skills)) < engine.LOW_SKILL_SIGNAL_THRESHOLD

    print(f"{name:<30}{report['field_used']:<18}{report['final_ats_score']:<8}{'YES' if adjusted else 'no'}")

print()


# ============================================================
# 3. Validate expected orderings (sanity checks with assertions)
# ============================================================

print("=" * 70)
print("VALIDATION CHECKS")
print("=" * 70)

checks_passed = 0
checks_total = 0


def check(condition, description):
    global checks_passed, checks_total
    checks_total += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        checks_passed += 1
    print(f"[{status}] {description}")


check(
    results["tech_strong"]["final_ats_score"] > results["tech_weak"]["final_ats_score"],
    "Strong-match tech resume scores higher than weak-match tech resume",
)

check(
    results["healthcare_strong"]["final_ats_score"] > 60,
    "Healthcare resume (correct taxonomy) scores reasonably high (>60) against its own JD",
)

check(
    results["healthcare_scored_as_tech"]["final_ats_score"]
    < results["healthcare_strong"]["final_ats_score"],
    "Same healthcare resume scores LOWER when evaluated with the WRONG "
    "(Technology / ML) taxonomy than with the correct (Healthcare) taxonomy",
)

check(
    len(results["healthcare_scored_as_tech"]["details"]["matched_skills"]) == 0,
    "Wrong-taxonomy case correctly finds ZERO matched skills "
    "(tech taxonomy doesn't recognize nursing terms)",
)

check(
    results["marketing_strong"]["final_ats_score"] > 50,
    "Marketing resume scores reasonably against its own field's JD "
    "(confirms multi-field taxonomy works beyond tech)",
)

check(
    results["edge_case_sparse_resume"]["final_ats_score"] >= 0,
    "Sparse/minimal resume does not crash the pipeline and returns a valid score",
)

print(f"\n{checks_passed}/{checks_total} checks passed.\n")


# ============================================================
# 4. Print full detail for the most interesting case: the mismatch test
# ============================================================

print("=" * 70)
print("DETAILED VIEW: Healthcare resume scored with WRONG taxonomy")
print("(This demonstrates the adaptive weighting fallback from Phase 8)")
print("=" * 70)

mismatch_report = results["healthcare_scored_as_tech"]
correct_report = results["healthcare_strong"]

print(f"Score with CORRECT taxonomy (Healthcare): {correct_report['final_ats_score']}")
print(f"  Breakdown: {correct_report['breakdown']}")
print()
print(f"Score with WRONG taxonomy (Technology / ML): {mismatch_report['final_ats_score']}")
print(f"  Breakdown: {mismatch_report['breakdown']}")
print()
print(
    "Interpretation: even with the wrong taxonomy, the semantic_score component "
    "should still carry meaningful signal (it doesn't depend on any skill dictionary), "
    "while skills_score collapses to near-zero matches. This is the adaptive weighting "
    "safety net from Phase 8 doing its job -- but note the score is still lower than the "
    "correct-taxonomy version, showing the fallback REDUCES damage, it doesn't fully "
    "eliminate it. Choosing the right field in the UI still matters."
)


# ============================================================
# 5. Documented limitations (for Phase 13 README)
# ============================================================

print("\n" + "=" * 70)
print("KNOWN LIMITATIONS (to document in Phase 13 README)")
print("=" * 70)
print("""
1. Semantic score rarely exceeds ~0.6-0.7 even for strong matches, since it
   compares dense multi-topic sections against dense multi-topic JDs
   (see Phase 5/6 discussion) -- the score is NOT on an intuitive 0-1 "percent
   match" scale, it is a relative signal.

2. Skill taxonomies are hand-curated and finite. A resume using terminology
   outside our taxonomy for a given field will show artificially low
   skills_score even if genuinely qualified. Adaptive weighting mitigates
   but does not eliminate this (see mismatch test above).

3. Field selection is manual (per your choice) -- if a user picks the wrong
   field, results degrade as shown above. No automatic validation currently
   warns the user their chosen field doesn't match their resume/JD content.

4. Experience/education extraction uses regex/keyword heuristics -- unusual
   phrasing (e.g. "half a decade of experience" instead of "5 years") will
   not be detected and defaults to 0, which can unfairly lower a score.

5. Formatting score is a weak proxy (counts detected sections) -- it does
   not check for genuinely ATS-breaking issues like text-in-images, tables,
   or multi-column layouts beyond what our Phase 2 section splitter catches.
""")