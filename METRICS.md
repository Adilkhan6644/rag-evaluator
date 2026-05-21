# RAG Evaluation Metrics

This document explains the evaluation metrics used by this RAG Evaluator to measure the quality of your retrieval-augmented generation pipeline.

## Overview

RAG evaluation consists of three main components:

1. **Retrieval Quality**: Did we retrieve the right information?
2. **Generation Quality**: Did we generate a good answer based on the context?
3. **Answer Correctness**: Is the answer factually correct compared to ground truth?

We measure these using **7 distinct metrics**, each focusing on a different aspect of the RAG pipeline. All metrics use a **0–1 scale** where **higher is better**.

---

## Metrics Explained

### 1. **Faithfulness** 📋

**What it measures:**  
How well the generated answer is grounded in the provided context. Does the answer stick to the facts in the context, or does it make up information?

**How it's measured:**  
- Uses the LLM to detect hallucinations
- Checks if answer statements can be verified against the context
- Flags information that appears to come from the model's training data rather than the context

**Range:** 0–1  
**Interpretation:**
- `0.9–1.0` → Excellent: Answer is fully grounded in context
- `0.7–0.9` → Good: Mostly grounded with minor hallucinations
- `0.5–0.7` → Fair: Some hallucinated content present
- `< 0.5` → Poor: Significant hallucinations detected

**Why it matters:**  
RAG is supposed to reduce hallucinations by grounding answers in retrieved context. A low faithfulness score means the model is ignoring the context.

---

### 2. **Answer Relevancy** 🎯

**What it measures:**  
Does the generated answer directly address the question asked? Is it on-topic and addressing what the user asked for?

**How it's measured:**  
- Uses embeddings to compute semantic similarity between the question and answer
- Extracts essential information from the question and checks if the answer covers it
- LLM verifies that the answer is topical and relevant

**Range:** 0–1  
**Interpretation:**
- `0.9–1.0` → Excellent: Answer directly and fully addresses the question
- `0.7–0.9` → Good: Answer is mostly relevant with minor tangents
- `0.5–0.7` → Fair: Answer is partially relevant but missing key aspects
- `< 0.5` → Poor: Answer is off-topic or irrelevant

**Why it matters:**  
A perfectly grounded but irrelevant answer is useless. This metric ensures the answer is actually answering the user's question.

---

### 3. **Context Precision** 🎲

**What it measures:**  
Of the chunks we retrieved, how many are actually useful for answering the question? Are we retrieving noise alongside good context?

**How it's measured:**  
- LLM checks each retrieved context chunk
- Evaluates which chunks are necessary to answer the question
- Calculates the ratio of relevant chunks to total chunks retrieved

**Range:** 0–1  
**Interpretation:**
- `0.9–1.0` → Excellent: All retrieved chunks are useful
- `0.7–0.9` → Good: Most chunks are useful, some noise
- `0.5–0.7` → Fair: About half the chunks are useful
- `< 0.5` → Poor: Most retrieved chunks are irrelevant

**Why it matters:**  
Noisy retrieval wastes model context and can lead to confusion. High precision means your retriever is working well.

---

### 4. **Context Recall** 🔍

**What it measures:**  
Does the retrieved context contain all the information needed to answer the question? Are we missing important details?

**How it measures:**  
- LLM extracts key facts from the ground truth answer
- Checks if those facts appear in the retrieved context
- Calculates the fraction of facts found in context

**Range:** 0–1  
**Interpretation:**
- `0.9–1.0` → Excellent: All necessary information was retrieved
- `0.7–0.9` → Good: Most information retrieved, some gaps
- `0.5–0.7` → Fair: About half the needed information retrieved
- `< 0.5` → Poor: Critical information was not retrieved

**Why it matters:**  
If the retriever doesn't fetch the right documents, the model can't generate a correct answer. Low recall means you have a retrieval problem.

---

### 5. **Answer Correctness** ✅

**What it measures:**  
Is the generated answer factually correct when compared against the ground truth answer? Does it contain the right information?

**How it's measured:**  
- Embedding similarity between generated answer and ground truth
- LLM assesses semantic equivalence
- Checks if key facts in ground truth appear in the generated answer

**Range:** 0–1  
**Interpretation:**
- `0.9–1.0` → Excellent: Answer is factually correct and complete
- `0.7–0.9` → Good: Mostly correct with minor omissions
- `0.5–0.7` → Fair: Partially correct, some errors
- `< 0.5` → Poor: Significantly incorrect or missing key facts

**Why it matters:**  
This is the ultimate measure of RAG quality: did we give the user the right answer? It combines all other components.

---

### 6. **Answer Similarity** 🔗

**What it measures:**  
How semantically similar is the generated answer to the ground truth answer? Using only embeddings, without LLM judgment.

**How it's measured:**  
- Embeds both the generated and ground truth answers
- Calculates cosine similarity between the two embeddings
- Pure embedding-based metric (doesn't use LLM)

**Range:** 0–1  
**Interpretation:**
- `0.9–1.0` → Excellent: Answers are nearly identical or paraphrases
- `0.7–0.9` → Good: Answers convey similar meaning
- `0.5–0.7` → Fair: Answers overlap but differ significantly
- `< 0.5` → Poor: Answers are semantically dissimilar

**Why it matters:**  
A fast, lightweight metric that catches when the answer is phrased differently but means the same thing. Complements Answer Correctness.

---

### 7. **Context Entity Recall** 🏷️

**What it measures:**  
Did we retrieve the key named entities (people, organizations, locations, dates, etc.) mentioned in the ground truth answer?

**How it's measured:**  
- Extracts named entities from the ground truth answer
- Checks if those entities appear in the retrieved context
- Calculates the fraction of entities found

**Range:** 0–1  
**Interpretation:**
- `0.9–1.0` → Excellent: All key entities were retrieved
- `0.7–0.9` → Good: Most entities retrieved
- `0.5–0.7` → Fair: About half the entities retrieved
- `< 0.5` → Poor: Most entities were not retrieved

**Why it matters:**  
Entities are often critical to the question (e.g., "Who founded X?"). This metric ensures we're retrieving context about the right people, places, and things.

---

## Interpreting Overall Results

### All Scores High (0.8+)
✅ Your RAG pipeline is working well:
- Retrieval is precise and complete
- Generation is faithful and relevant
- Answers are factually correct

### Retrieval Metrics Low (Context Precision/Recall)
⚠️ Problem: Your retriever is not finding the right documents
- Try: Improve vectorization, update search strategy, expand knowledge base
- RAG can't work if the retriever fails

### Faithfulness Low + Others High
⚠️ Problem: Model is hallucinating despite good context
- Try: Lower model temperature, add explicit instructions to stick to context, switch to a more reliable model

### Answer Relevancy/Correctness Low
⚠️ Problem: Generation is not addressing the question or is generating wrong answers
- Try: Refine prompts, provide clearer context formatting, use a more capable model

### Answer Similarity Low but Correctness High
✅ This is OK: The answer is correct even if phrased differently than ground truth

---

## Example Report

```
Faithfulness:          0.9200   |  Answer grounded in context?
AnswerRelevancy:       0.8500   |  Answer relevant to question?
ContextPrecision:      0.9000   |  Retrieved chunks useful?
ContextRecall:         0.8700   |  Context covers ground truth?
AnswerCorrectness:     0.8900   |  Answer correct vs ground truth?
AnswerSimilarity:      0.8100   |  Semantic similarity to truth?
ContextEntityRecall:   0.9100   |  Key entities recalled?
```

**Overall Assessment:** This is a strong result. The pipeline is retrieving good context, generating faithful answers, and achieving high correctness.

---

## Technical Details

- **LLM-based metrics** use `llama-3.3-70b-versatile` via Groq for assessment
- **Embedding-based metrics** use `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity
- **Scores are averaged** across all Q&A pairs in a run
- Each run also saves **per-question scores** for debugging specific failures

---

## References

- [Ragas Documentation](https://docs.ragas.io/)
- [RAG Evaluation Papers](https://docs.ragas.io/en/stable/references/)

