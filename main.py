import os
import json
import datetime
import pandas as pd
from groq import Groq
from openai import OpenAI

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from datasets import Dataset

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    AnswerSimilarity,
    ContextEntityRecall,
)

# =========================
# 1. GROQ CLIENT (GENERATION)
# =========================

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================
# 2. KNOWLEDGE BASE
# =========================

docs = [
    "Docker is a platform for building and running containers.",
    "Python is a programming language used for AI and web development.",
    "RAG combines retrieval and generation in LLM systems.",
    "Vector databases store embeddings for semantic search."
]

# =========================
# 3. RETRIEVER (TF-IDF)
# =========================

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(docs)

def retrieve(query, k=2):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, doc_vectors)[0]
    top_k = scores.argsort()[-k:][::-1]
    return [docs[i] for i in top_k]

# =========================
# 4. GROQ GENERATION
# =========================

def generate_answer(query, contexts):
    context_text = "\n".join(contexts)

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the provided context.

Context:
{context_text}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content

# =========================
# 5. TEST DATA
# =========================

questions = [
    "What is Docker?",
    "What is RAG?"
]

ground_truth = [
    "Docker is a platform for building and running containers.",
    "RAG combines retrieval and generation in LLM systems."
]

# =========================
# 6. RUN RAG PIPELINE
# =========================

answers = []
contexts = []

print("\n🚀 Running RAG pipeline...\n")

for q in questions:
    ctx = retrieve(q)
    ans = generate_answer(q, ctx)

    answers.append(ans)
    contexts.append(ctx)

    print("=" * 60)
    print("QUESTION :", q)
    print("CONTEXT  :", ctx)
    print("ANSWER   :", ans)
    print("=" * 60)

# =========================
# 7. DATASET FOR RAGAS
# =========================

dataset = Dataset.from_dict({
    "question":     questions,
    "answer":       answers,
    "contexts":     contexts,
    "ground_truth": ground_truth
})

# =========================
# 8. LLM + EMBEDDINGS SETUP
# =========================

print("\n📊 Running Ragas evaluation...\n")

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

ragas_llm = llm_factory(
    model="llama-3.3-70b-versatile",
    client=groq_client
)

embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

# =========================
# 9. ALL METRICS
# =========================

metrics = [
    Faithfulness(llm=ragas_llm),                               # Is the answer grounded in the context?
    AnswerRelevancy(llm=ragas_llm, embeddings=embeddings),      # Is the answer relevant to the question?
    ContextPrecision(llm=ragas_llm),                           # Are retrieved chunks actually useful?
    ContextRecall(llm=ragas_llm),                              # Does context cover the ground truth?
    AnswerCorrectness(llm=ragas_llm, embeddings=embeddings),    # Is the answer factually correct vs ground truth?
    AnswerSimilarity(embeddings=embeddings),                    # Semantic similarity: answer vs ground truth
    ContextEntityRecall(llm=ragas_llm),                        # Are key entities from ground truth in context?
]

# =========================
# 10. EVALUATE
# =========================

result = evaluate(
    dataset,
    metrics=metrics,
    llm=ragas_llm,
    embeddings=embeddings
)

# =========================
# 11. DISPLAY RESULTS
# =========================

# ✅ result.scores is a list of dicts — convert to DataFrame for mean
scores_df = pd.DataFrame(result.scores)
scores_dict = scores_df.mean().to_dict()

print("\n" + "=" * 60)
print("🔥  FINAL EVALUATION SCORES")
print("=" * 60)

metric_descriptions = {
    "faithfulness":          "Answer grounded in context?       (0–1, higher = better)",
    "answer_relevancy":      "Answer relevant to question?      (0–1, higher = better)",
    "context_precision":     "Retrieved chunks useful?          (0–1, higher = better)",
    "context_recall":        "Context covers ground truth?      (0–1, higher = better)",
    "answer_correctness":    "Answer correct vs ground truth?   (0–1, higher = better)",
    "answer_similarity":     "Semantic similarity to truth?     (0–1, higher = better)",
    "context_entity_recall": "Key entities recalled?            (0–1, higher = better)",
}

for metric, description in metric_descriptions.items():
    value = scores_dict.get(metric, "N/A")
    value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
    print(f"  {metric:<25} {value_str}   |  {description}")

print("=" * 60)

# =========================
# 12. SAVE TO FILE
# =========================

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"rag_evaluation_{timestamp}.json"

output = {
    "timestamp": timestamp,
    "model": "llama-3.3-70b-versatile",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "num_questions": len(questions),

    # Per-question detail
    "qa_pairs": [
        {
            "question":     questions[i],
            "context":      contexts[i],
            "answer":       answers[i],
            "ground_truth": ground_truth[i],
        }
        for i in range(len(questions))
    ],

    # Aggregate scores (mean across all questions)
    "aggregate_scores": {
        metric: round(float(value), 4) if isinstance(value, float) else value
        for metric, value in scores_dict.items()
    },

    # ✅ Per-question scores — use DataFrame to_dict, NOT result.scores.to_dict
    "per_question_scores": scores_df.to_dict(orient="records")
}

with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n💾 Results saved to: {output_filename}\n")