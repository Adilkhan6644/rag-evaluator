# RAG Evaluator

This project evaluates a simple retrieval-augmented generation (RAG) pipeline with [Ragas](https://docs.ragas.io/) and shows the results in a Streamlit app.

It includes two entry points:

- `main.py` runs a CLI version of the pipeline, generates answers, evaluates them, and saves the metrics to a timestamped JSON file.
- `app.py` runs a Streamlit dashboard with the same evaluation flow and a styled UI for reviewing the scores.

## Features

- TF-IDF retrieval over a small sample knowledge base
- Answer generation through Groq using `llama-3.3-70b-versatile`
- Ragas evaluation with metrics such as faithfulness, answer relevancy, context precision, context recall, answer correctness, answer similarity, and context entity recall
- JSON export of each run for later comparison

## Requirements

- Python 3.10 or newer
- A Groq API key

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Add your API key to a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

## Run

Run the command-line evaluator:

```bash
python main.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Output

Each run writes a file named like `rag_evaluation_YYYYMMDD_HHMMSS.json` in the project root. The file includes:

- the questions used
- retrieved contexts
- generated answers
- ground truth answers
- aggregate scores
- per-question scores

## Notes

- The sample knowledge base is intentionally small and only meant to demonstrate the evaluation workflow.
- The `.env` file and virtual environment are ignored by git so secrets and local packages are not committed.