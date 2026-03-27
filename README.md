# Auto-Merging Retrieval with LlamaIndex

A RAG (Retrieval-Augmented Generation) pipeline that uses **hierarchical node parsing** and **auto-merging retrieval** to answer questions over a PDF document. Optionally evaluates performance with **TruLens** across answer relevance, context relevance, and groundedness.

---

## How It Works

```
PDF → Pages → Merged Document
                    ↓
        HierarchicalNodeParser
                    ↓
    ┌───────────────────────────┐
    │  Layer 1: 2048-token nodes│  ← parent nodes (stored in docstore)
    │  Layer 2:  512-token nodes│  ← mid nodes
    │  Layer 3:  128-token nodes│  ← leaf nodes (indexed in VectorStore)
    └───────────────────────────┘
                    ↓
    Query → retrieve top-K leaf nodes
                    ↓
    AutoMergingRetriever: if enough siblings hit, swap for parent node
                    ↓
    SentenceTransformerRerank → top-N nodes
                    ↓
    LLM synthesizes final answer
```

---

## Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)

---

## Setup

### 1. Clone or download the script

Place `auto_merging_retrieval.py` in a working directory.

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install llama-index \
            llama-index-embeddings-huggingface \
            llama-index-llms-openai \
            torch \
            sentence-transformers \
            trulens-eval           # optional — only needed for evaluation
```

> **Note:** `torch` and `sentence-transformers` can be large downloads (~1–2 GB). The embedding and reranker models are downloaded automatically from HuggingFace on first run.

### 4. Set your OpenAI API key

```bash
# macOS / Linux
export OPENAI_API_KEY="sk-..."

# Windows (Command Prompt)
set OPENAI_API_KEY=sk-...

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
```

### 5. Add your PDF

Place your PDF in the same directory as the script. By default the script looks for:

```
eBook-How-to-Build-a-Career-in-AI.pdf
```

You can change this at the top of the script:

```python
PDF_PATH = "your-document.pdf"
```

### 6. (Optional) Add evaluation questions

Create a plain-text file with one question per line, e.g. `generated_questions.text`. If the file is present the script will automatically run TruLens evaluation after the demo queries. Set the path at the top of the script:

```python
EVAL_QUESTIONS_PATH = "generated_questions.text"
```

---

## Running

```bash
python auto_merge.py
```

### What happens

1. **Loads** the PDF and merges all pages into one document.
2. **Builds** (or reloads from disk) two indexes:
   - `merging_index_0/` — two-layer hierarchy (2048 / 512 tokens)
   - `merging_index_1/` — three-layer hierarchy (2048 / 512 / 128 tokens)
3. **Runs** a demo query against each index.
4. **(Optional)** If `generated_questions.text` exists and `trulens-eval` is installed, evaluates both engines and prints a leaderboard table, then launches the TruLens dashboard at `http://localhost:8501`.

### Re-runs

Indexes are persisted to disk. On subsequent runs the script loads them instead of rebuilding, making re-runs much faster. To force a rebuild, delete the index directories:

```bash
rm -rf merging_index_0 merging_index_1
```

---

## Configuration

All tuneable parameters are at the top of `auto_merging_retrieval.py`:

| Variable | Default | Description |
|---|---|---|
| `PDF_PATH` | `eBook-How-to-Build-a-Career-in-AI.pdf` | Path to your input PDF |
| `EVAL_QUESTIONS_PATH` | `generated_questions.text` | Path to evaluation questions file |
| `EMBED_MODEL` | `local:BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | HuggingFace cross-encoder reranker |
| `LLM_MODEL` | `gpt-3.5-turbo` | OpenAI chat model |
| `LLM_TEMPERATURE` | `0.1` | LLM sampling temperature |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `EnvironmentError: OPENAI_API_KEY is not set` | Export the key as shown in Step 4 |
| `ModuleNotFoundError` | Re-run the `pip install` command from Step 3 |
| CUDA / torch errors on CPU-only machine | Install the CPU-only torch build: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| TruLens dashboard doesn't open | Install `trulens-eval` and ensure port 8501 is free |
| Index loads stale data after changing the PDF | Delete the `merging_index_*/` directories and re-run |

---

## Project Structure

```
.
├── auto_merging_retrieval.py   ← main script
├── eBook-How-to-Build-a-Career-in-AI.pdf
├── generated_questions.text    ← optional
├── merging_index_0/            ← auto-created on first run
└── merging_index_1/            ← auto-created on first run
```