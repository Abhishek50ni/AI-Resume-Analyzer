"""
app.py

AI Resume Analyzer & ATS Score Predictor -- Streamlit Application (Phase 11)

Run locally with:
    streamlit run app.py

This file contains ONLY UI/orchestration code. All actual logic
(PDF parsing, embeddings, scoring, recommendations) lives in ats_engine.py.
"""

import streamlit as st
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import ats_engine as engine


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer & ATS Score Predictor",
    page_icon="\U0001F4C4",
    layout="wide",
)

st.title("\U0001F4C4 AI Resume Analyzer & ATS Score Predictor")
st.caption("Semantic resume-to-job matching powered by Sentence Transformers -- not just keyword matching.")


# ============================================================
# Sidebar: inputs
# ============================================================

with st.sidebar:
    st.header("Inputs")

    uploaded_pdfs = st.file_uploader(
        "Upload one or more resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    job_description_text = st.text_area(
        "Paste the job description",
        height=250,
        placeholder="Paste the full job description text here...",
    )

    field = st.selectbox(
        "Select the job field",
        options=["Technology / ML"] + [f for f in engine.AVAILABLE_FIELDS if f != "Technology / ML"],
        help="Choosing the right field improves skill-matching accuracy. "
             "The semantic score works well regardless of field; "
             "the skills checklist is field-specific.",
    )

    top_n_choice = st.selectbox(
        "Show top N candidates",
        options=["Top 3", "Top 5", "Top 10", "All"],
        index=1,
    )

    analyze_clicked = st.button("Analyze Resume(s)", type="primary", use_container_width=True)


# ============================================================
# Helper: cache expensive model loads across reruns
# ============================================================

@st.cache_resource(show_spinner="Loading language model (first run only)...")
def load_models():
    engine.get_model()
    engine.get_spacy_nlp()
    return True


# ============================================================
# Main logic
# ============================================================

if analyze_clicked:
    if not uploaded_pdfs:
        st.warning("Please upload at least one resume PDF before analyzing.")
        st.stop()

    if not job_description_text or not job_description_text.strip():
        st.warning("Please paste a job description before analyzing.")
        st.stop()

    load_models()

    results = []  # list of dicts: {filename, resume_text, report}
    progress_bar = st.progress(0.0, text="Starting analysis...")

    for i, uploaded_pdf in enumerate(uploaded_pdfs):
        progress_bar.progress(
            i / len(uploaded_pdfs), text=f"Analyzing {uploaded_pdf.name} ({i + 1}/{len(uploaded_pdfs)})..."
        )

        try:
            pdf_bytes = uploaded_pdf.read()
            resume_text = engine.extract_text_from_pdf_bytes(pdf_bytes)
            resume_text = engine.preprocess_for_embedding(resume_text)
        except Exception as e:
            st.warning(f"Skipped '{uploaded_pdf.name}': could not extract text ({e}).")
            continue

        if not resume_text.strip():
            st.warning(
                f"Skipped '{uploaded_pdf.name}': no extractable text found "
                f"(likely a scanned/image-based PDF)."
            )
            continue

        report = engine.generate_recommendations(resume_text, job_description_text, field=field)
        results.append({
            "filename": uploaded_pdf.name,
            "resume_text": resume_text,
            "report": report,
        })

    progress_bar.progress(1.0, text="Done.")
    progress_bar.empty()

    if not results:
        st.error("No resumes could be analyzed. Please check your uploaded files.")
        st.stop()

    # Sort by final ATS score, descending -- this is our ranking
    results.sort(key=lambda r: r["report"]["final_ats_score"], reverse=True)

    st.session_state["results"] = results
    st.session_state["job_description_text"] = job_description_text
    st.session_state["top_n_choice"] = top_n_choice


# ============================================================
# Display results (persists across reruns via session_state)
# ============================================================

def build_text_report(report: dict) -> str:
    lines = []
    lines.append("AI RESUME ANALYZER -- ATS REPORT")
    lines.append("=" * 40)
    lines.append(f"Field: {report['field_used']}")
    lines.append(f"Final ATS Score: {report['final_ats_score']} / 100")
    lines.append("")
    lines.append("SCORE BREAKDOWN")
    for k, v in report["breakdown"].items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("MATCHED SKILLS")
    lines.append(f"  {', '.join(report['details']['matched_skills']) or 'None'}")
    lines.append("")
    lines.append("MISSING SKILLS")
    lines.append(f"  {', '.join(report['details']['missing_skills']) or 'None'}")
    lines.append("")
    lines.append("STRENGTHS")
    for item in report["strengths"]:
        lines.append(f"  - {item}")
    lines.append("")
    lines.append("WEAKNESSES")
    for item in report["weaknesses"]:
        lines.append(f"  - {item}")
    lines.append("")
    lines.append("RECOMMENDATIONS")
    for item in report["missing_skills_recommendations"] + report["section_recommendations"]:
        lines.append(f"  - {item}")
    return "\n".join(lines)


def render_candidate_detail(filename: str, resume_text: str, report: dict):
    """Renders the full detailed dashboard for ONE candidate (reusable)."""

    # --- Top row: gauge + radar ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Final ATS Score")
        score = report["final_ats_score"]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00C896"},
                "steps": [
                    {"range": [0, 40], "color": "#4a1f1f"},
                    {"range": [40, 70], "color": "#4a431f"},
                    {"range": [70, 100], "color": "#1f4a2e"},
                ],
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#FAFAFA"},
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("Score Breakdown")
        breakdown = report["breakdown"]
        categories = list(breakdown.keys())
        values = list(breakdown.values())
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed, theta=categories_closed, fill="toself",
            line_color="#00C896",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#FAFAFA"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#FAFAFA"},
            height=300,
            margin=dict(l=40, r=40, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- Skill coverage ---
    st.subheader("Skill Coverage")
    matched = report["details"]["matched_skills"]
    missing = report["details"]["missing_skills"]

    fig_bar = go.Figure(data=[
        go.Bar(name="Matched", x=["Skills"], y=[len(matched)], marker_color="#00C896"),
        go.Bar(name="Missing", x=["Skills"], y=[len(missing)], marker_color="#D64545"),
    ])
    fig_bar.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#FAFAFA"},
        height=250,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    skill_col1, skill_col2 = st.columns(2)
    with skill_col1:
        st.markdown("**\u2705 Matched Skills**")
        st.write(", ".join(matched) if matched else "None detected.")
    with skill_col2:
        st.markdown("**\u26A0\uFE0F Missing Skills**")
        st.write(", ".join(missing) if missing else "None -- full coverage!")

    # --- Strengths / Weaknesses ---
    st.divider()
    strengths_col, weaknesses_col = st.columns(2)

    with strengths_col:
        st.subheader("\U0001F4AA Strengths")
        for item in report["strengths"]:
            st.markdown(f"- {item}")

    with weaknesses_col:
        st.subheader("\U0001F6A9 Weaknesses")
        for item in report["weaknesses"]:
            st.markdown(f"- {item}")

    # --- Recommendations ---
    st.divider()
    st.subheader("\U0001F4A1 Recommendations")

    for item in report["missing_skills_recommendations"]:
        st.markdown(f"- {item}")
    for item in report["section_recommendations"]:
        st.markdown(f"- {item}")

    # --- Word cloud (illustrative only) ---
    with st.expander("View Resume Word Cloud (illustrative only)"):
        wc = WordCloud(width=800, height=400, background_color="#0E1117", colormap="viridis").generate(resume_text)
        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig_wc.patch.set_facecolor("#0E1117")
        st.pyplot(fig_wc)

    # --- Downloadable report ---
    st.divider()
    report_text = build_text_report(report)
    st.download_button(
        "\U0001F4E5 Download Full ATS Report (.txt)",
        data=report_text,
        file_name=f"ats_report_{filename.rsplit('.', 1)[0]}.txt",
        mime="text/plain",
        use_container_width=True,
        key=f"download_{filename}",
    )


# ============================================================
# Top-level display: ranking table + comparison chart + drill-down
# ============================================================

if "results" in st.session_state:
    results = st.session_state["results"]
    top_n_choice = st.session_state.get("top_n_choice", "Top 5")

    n_map = {"Top 3": 3, "Top 5": 5, "Top 10": 10, "All": len(results)}
    top_n = min(n_map[top_n_choice], len(results))

    st.divider()
    st.subheader(f"\U0001F3C6 Ranking -- {top_n_choice} of {len(results)} resume(s) analyzed")

    # --- Ranked table ---
    table_rows = []
    for rank, r in enumerate(results, start=1):
        table_rows.append({
            "Rank": rank,
            "Filename": r["filename"],
            "ATS Score": r["report"]["final_ats_score"],
            "Matched Skills": len(r["report"]["details"]["matched_skills"]),
            "Missing Skills": len(r["report"]["details"]["missing_skills"]),
        })

    st.dataframe(
        table_rows[:top_n],
        use_container_width=True,
        hide_index=True,
    )

    if len(results) > 1:
        st.caption(f"Showing top {top_n} of {len(results)} total resumes. Change this in the sidebar.")

    # --- Comparison bar chart across all candidates ---
    if len(results) > 1:
        st.subheader("Score Comparison")
        fig_compare = go.Figure(data=[
            go.Bar(
                x=[r["filename"] for r in results[:top_n]],
                y=[r["report"]["final_ats_score"] for r in results[:top_n]],
                marker_color="#00C896",
            )
        ])
        fig_compare.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#FAFAFA"},
            yaxis_title="ATS Score",
            height=350,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    # --- Drill-down into one candidate's full report ---
    st.divider()
    st.subheader("\U0001F4CB Detailed Candidate Report")

    selected_filename = st.selectbox(
        "Select a candidate to view full details",
        options=[r["filename"] for r in results[:top_n]],
    )

    selected_result = next(r for r in results if r["filename"] == selected_filename)
    render_candidate_detail(
        selected_result["filename"],
        selected_result["resume_text"],
        selected_result["report"],
    )

else:
    st.info(
        "Upload one or more resume PDFs and paste a job description in the sidebar, "
        "then click **Analyze Resume(s)**."
    )