import os
import json
import datetime
import pandas as pd
import streamlit as st
from groq import Groq
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    AnswerSimilarity,
    ContextEntityRecall,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Evaluator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}

/* Header */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #00ffcc;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1rem;
    color: #666680;
    letter-spacing: 0.05em;
    margin-bottom: 2rem;
    font-family: 'Space Mono', monospace;
}

/* Metric cards */
.metric-card {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--accent);
    border-radius: 3px 0 0 3px;
}
.metric-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #888899;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-desc {
    font-size: 0.78rem;
    color: #555566;
    margin-top: 0.2rem;
}

/* Score bar */
.score-bar-bg {
    background: #1e1e2e;
    border-radius: 4px;
    height: 6px;
    margin-top: 0.5rem;
}
.score-bar-fill {
    height: 6px;
    border-radius: 4px;
    background: var(--accent);
    transition: width 1s ease;
}

/* QA pair cards */
.qa-card {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.qa-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #555566;
    margin-bottom: 0.3rem;
}
.qa-content {
    font-size: 0.95rem;
    color: #c8c8d8;
    line-height: 1.5;
}

/* Buttons */
.stButton > button {
    background: #00ffcc !important;
    color: #0a0a0f !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #00ccaa !important;
    transform: translateY(-1px) !important;
}

/* Text inputs & text areas */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #13131f !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextArea > div > div > textarea:focus,
.stTextInput > div > div > input:focus {
    border-color: #00ffcc !important;
    box-shadow: 0 0 0 2px rgba(0,255,204,0.1) !important;
}

/* Section divider */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #00ffcc;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e2e;
}

/* Status badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.badge-green { background: rgba(0,255,204,0.1); color: #00ffcc; border: 1px solid rgba(0,255,204,0.2); }
.badge-yellow { background: rgba(255,200,0,0.1); color: #ffc800; border: 1px solid rgba(255,200,0,0.2); }
.badge-red { background: rgba(255,80,80,0.1); color: #ff5050; border: 1px solid rgba(255,80,80,0.2); }

/* Expander */
.streamlit-expanderHeader {
    background: #13131f !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_accent_color(score: float) -> str:
    if score >= 0.8:
        return "#00ffcc"
    elif score >= 0.5:
        return "#ffc800"
    else:
        return "#ff5050"

def get_badge(score: float) -> str:
    if score >= 0.8:
        return '<span class="badge badge-green">Excellent</span>'
    elif score >= 0.5:
        return '<span class="badge badge-yellow">Fair</span>'
    else:
        return '<span class="badge badge-red">Poor</span>'

def render_metric_card(name: str, value: float, description: str):
    accent = get_accent_color(value)
    badge = get_badge(value)
    pct = int(value * 100)
    st.markdown(f"""
    <div class="metric-card" style="--accent: {accent}">
        <div class="metric-name">{name.replace("_", " ")}</div>
        <div style="display:flex; align-items:baseline; gap:0.8rem;">
            <div class="metric-value">{value:.3f}</div>
            {badge}
        </div>
        <div class="metric-desc">{description}</div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct}%; --accent:{accent}"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )


def run_rag_pipeline(questions, ground_truths, docs, groq_api_key):
    # Clients
    groq_client = Groq(api_key=groq_api_key)

    # Retriever
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(docs)

    def retrieve(query, k=2):
        q_vec = vectorizer.transform([query])
        scores = cosine_similarity(q_vec, doc_vectors)[0]
        top_k = scores.argsort()[-k:][::-1]
        return [docs[i] for i in top_k]

    def generate_answer(query, contexts):
        context_text = "\n".join(contexts)
        prompt = f"""You are a helpful assistant. Answer ONLY using the provided context.

Context:
{context_text}

Question:
{query}

Answer:"""
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content

    answers, contexts = [], []
    progress = st.progress(0, text="Running RAG pipeline...")
    for i, q in enumerate(questions):
        ctx = retrieve(q)
        ans = generate_answer(q, ctx)
        answers.append(ans)
        contexts.append(ctx)
        progress.progress((i + 1) / len(questions), text=f"Generating answer {i+1}/{len(questions)}...")

    progress.empty()
    return answers, contexts


def run_evaluation(questions, answers, contexts, ground_truths, groq_api_key, embeddings):
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    groq_openai = OpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    ragas_llm = llm_factory(model="llama-3.3-70b-versatile", client=groq_openai)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
        AnswerCorrectness(llm=ragas_llm, embeddings=embeddings),
        AnswerSimilarity(embeddings=embeddings),
        ContextEntityRecall(llm=ragas_llm),
    ]

    with st.spinner("⚙️ Evaluating with RAGAS metrics — this takes a minute..."):
        result = evaluate(dataset, metrics=metrics, llm=ragas_llm, embeddings=embeddings)

    scores_df = pd.DataFrame(result.scores)
    scores_dict = scores_df.mean().to_dict()
    return scores_dict, scores_df, answers, contexts


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.4rem;">⚙️ Config</div>', unsafe_allow_html=True)
    st.markdown("---")

    groq_api_key = st.text_input(
        "GROQ API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_..."
    )

    st.markdown('<div class="section-title">Knowledge Base</div>', unsafe_allow_html=True)
    st.caption("One document per line")
    default_docs = """Docker is a platform for building and running containers.
Python is a programming language used for AI and web development.
RAG combines retrieval and generation in LLM systems.
Vector databases store embeddings for semantic search."""

    docs_input = st.text_area("Documents", value=default_docs, height=160)

    st.markdown('<div class="section-title">Test Questions</div>', unsafe_allow_html=True)
    st.caption("One question per line")
    questions_input = st.text_area(
        "Questions",
        value="What is Docker?\nWhat is RAG?",
        height=100
    )

    st.markdown('<div class="section-title">Ground Truth Answers</div>', unsafe_allow_html=True)
    st.caption("Must match question count")
    ground_truth_input = st.text_area(
        "Ground Truths",
        value="Docker is a platform for building and running containers.\nRAG combines retrieval and generation in LLM systems.",
        height=100
    )

    run_btn = st.button("▶ Run Evaluation", use_container_width=True)

    st.markdown("---")
    st.markdown('<p style="font-family:Space Mono,monospace; font-size:0.65rem; color:#333344; text-align:center;">RAG EVALUATOR · POWERED BY GROQ + RAGAS</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">RAG Evaluator</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">// retrieval-augmented generation · quality metrics</div>', unsafe_allow_html=True)

METRIC_DESCRIPTIONS = {
    "faithfulness":          "Answer grounded in context?",
    "answer_relevancy":      "Answer relevant to question?",
    "context_precision":     "Retrieved chunks useful?",
    "context_recall":        "Context covers ground truth?",
    "answer_correctness":    "Answer correct vs ground truth?",
    "answer_similarity":     "Semantic similarity to truth?",
    "context_entity_recall": "Key entities recalled?",
}

if not run_btn:
    # Placeholder state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card" style="--accent:#1e1e2e; opacity:0.5;">
            <div class="metric-name">Faithfulness</div>
            <div class="metric-value" style="color:#1e1e2e;">—</div>
            <div class="score-bar-bg"><div class="score-bar-fill" style="width:0%"></div></div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card" style="--accent:#1e1e2e; opacity:0.5;">
            <div class="metric-name">Answer Relevancy</div>
            <div class="metric-value" style="color:#1e1e2e;">—</div>
            <div class="score-bar-bg"><div class="score-bar-fill" style="width:0%"></div></div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card" style="--accent:#1e1e2e; opacity:0.5;">
            <div class="metric-name">Context Precision</div>
            <div class="metric-value" style="color:#1e1e2e;">—</div>
            <div class="score-bar-bg"><div class="score-bar-fill" style="width:0%"></div></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding: 4rem 0; color: #333344;">
        <div style="font-family:'Space Mono',monospace; font-size:3rem;">🧠</div>
        <div style="font-family:'Space Mono',monospace; font-size:0.8rem; letter-spacing:0.2em; margin-top:1rem;">
            CONFIGURE AND RUN TO SEE RESULTS
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Validate inputs ──
    if not groq_api_key:
        st.error("⚠️ Please enter your GROQ API key in the sidebar.")
        st.stop()

    docs = [d.strip() for d in docs_input.strip().split("\n") if d.strip()]
    questions = [q.strip() for q in questions_input.strip().split("\n") if q.strip()]
    ground_truths = [g.strip() for g in ground_truth_input.strip().split("\n") if g.strip()]

    if len(questions) != len(ground_truths):
        st.error(f"⚠️ Question count ({len(questions)}) must match ground truth count ({len(ground_truths)}).")
        st.stop()

    if not docs:
        st.error("⚠️ Please add at least one document to the knowledge base.")
        st.stop()

    # ── Load embeddings (cached) ──
    with st.spinner("Loading embedding model..."):
        embeddings = load_embeddings()

    # ── Run pipeline ──
    answers, contexts = run_rag_pipeline(questions, ground_truths, docs, groq_api_key)

    # ── Run evaluation ──
    scores_dict, scores_df, answers, contexts = run_evaluation(
        questions, answers, contexts, ground_truths, groq_api_key, embeddings
    )

    # ─── SCORES GRID ───
    st.markdown('<div class="section-title">Evaluation Metrics</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, _ , __ = st.columns(3)

    metric_items = [(k, scores_dict.get(k, 0.0), METRIC_DESCRIPTIONS.get(k, "")) for k in METRIC_DESCRIPTIONS]

    columns = [col1, col2, col3, col4, col5, col6, col7]
    for i, (name, value, desc) in enumerate(metric_items):
        with columns[i]:
            render_metric_card(name, value, desc)

    # ─── OVERALL SCORE ───
    overall = sum(scores_dict.values()) / len(scores_dict)
    accent = get_accent_color(overall)
    st.markdown(f"""
    <div class="metric-card" style="--accent:{accent}; margin-top:1rem;">
        <div class="metric-name">Overall Average Score</div>
        <div style="display:flex; align-items:baseline; gap:0.8rem;">
            <div class="metric-value">{overall:.3f}</div>
            {get_badge(overall)}
        </div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{int(overall*100)}%; --accent:{accent}"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── PER-QUESTION BREAKDOWN ───
    st.markdown('<div class="section-title">Per-Question Details</div>', unsafe_allow_html=True)

    for i, q in enumerate(questions):
        with st.expander(f"Q{i+1}: {q}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="qa-label">Question</div><div class="qa-content">{q}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="qa-label" style="margin-top:1rem;">Answer</div><div class="qa-content">{answers[i]}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="qa-label">Ground Truth</div><div class="qa-content">{ground_truths[i]}</div>', unsafe_allow_html=True)
                ctx_html = "<br>".join([f"• {c}" for c in contexts[i]])
                st.markdown(f'<div class="qa-label" style="margin-top:1rem;">Retrieved Context</div><div class="qa-content">{ctx_html}</div>', unsafe_allow_html=True)

            if i < len(scores_df):
                st.markdown('<div class="qa-label" style="margin-top:1rem;">Metric Scores</div>', unsafe_allow_html=True)
                row = scores_df.iloc[i].to_dict()
                mcols = st.columns(len(row))
                for j, (metric, val) in enumerate(row.items()):
                    with mcols[j]:
                        color = get_accent_color(val)
                        st.markdown(f"""
                        <div style="text-align:center; padding:0.5rem; background:#0f0f1a; border-radius:8px; border:1px solid #1e1e2e;">
                            <div style="font-family:'Space Mono',monospace; font-size:0.55rem; color:#555566; text-transform:uppercase; letter-spacing:0.1em;">{metric.replace('_',' ')}</div>
                            <div style="font-family:'Space Mono',monospace; font-size:1.1rem; color:{color}; font-weight:700;">{val:.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # ─── SAVE & DOWNLOAD ───
    st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "model": "llama-3.3-70b-versatile",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_questions": len(questions),
        "overall_score": round(overall, 4),
        "qa_pairs": [
            {
                "question":     questions[i],
                "context":      contexts[i],
                "answer":       answers[i],
                "ground_truth": ground_truths[i],
            }
            for i in range(len(questions))
        ],
        "aggregate_scores": {
            k: round(float(v), 4) for k, v in scores_dict.items()
        },
        "per_question_scores": scores_df.to_dict(orient="records"),
    }

    json_str = json.dumps(output, indent=2, ensure_ascii=False)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            label="⬇ Download JSON Report",
            data=json_str,
            file_name=f"rag_evaluation_{timestamp}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_b:
        csv_data = scores_df.to_csv(index=False)
        st.download_button(
            label="⬇ Download CSV Scores",
            data=csv_data,
            file_name=f"rag_scores_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )