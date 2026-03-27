# RAG Retrieval Strategies with LlamaIndex

Two complementary RAG (Retrieval-Augmented Generation) pipelines built with LlamaIndex, each using a different retrieval strategy and evaluated with TruLens.

| Script | Strategy | Best for |
|---|---|---|
| `auto_merging_retrieval.py` | Hierarchical nodes — retrieve small chunks, merge up to parent if enough hits | Broad topic coverage across long documents |
| `sentence_window_retrieval.py` | Sentence-level nodes with surrounding context window | Precise, localized answers that need a little surrounding context |

---

## How Each Strategy Works

### Auto-Merging Retrieval

```
PDF → Merged Document
           ↓
  HierarchicalNodeParser
           ↓
  ┌──────────────────────────┐
  │ Layer 1: 2048-token nodes│  ← parent nodes (stored in docstore)
  │ Layer 2:  512-token nodes│  ← mid nodes
  │ Layer 3:  128-token nodes│  ← leaf nodes (indexed in VectorStore)
  └──────────────────────────┘
           ↓
  Query → retrieve top-K leaf nodes
           ↓
  AutoMergingRetriever: if enough siblings hit → swap for parent node
           ↓
  SentenceTransformerRerank → top-N nodes → LLM answer
```

### Sentence Window Retrieval

```
PDF → Merged Document
           ↓
  SentenceWindowNodeParser
           ↓
  Each node = one sentence
  Metadata carries ±N surrounding sentences as "window"
           ↓
  Query → retrieve top-K matching sentences
           ↓
  MetadataReplacementPostProcessor: swap sentence text → full window
           ↓
  SentenceTransformerRerank → top-N windows → LLM answer
```

---

## Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)

---

## Setup

### 1. Place the scripts in a working directory

```
project/
├── auto_merging_retrieval.py
├── sentence_window_retrieval.py
├── eBook-How-to-Build-a-Career-in-AI.pdf
└── generated_questions.text   ← optional
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
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

Place your PDF in the same directory as the scripts. Both scripts look for:

```
eBook-How-to-Build-a-Career-in-AI.pdf
```

You can change this at the top of either script:

```python
PDF_PATH = "your-document.pdf"
```

### 6. (Optional) Add evaluation questions

Create a plain-text file with one question per line, e.g. `generated_questions.text`. If the file is present, both scripts will automatically run TruLens evaluation after the demo queries. Update the path at the top of each script if needed:

```python
EVAL_QUESTIONS_PATH = "generated_questions.text"
```

---

## Running

### Auto-Merging Retrieval

```bash
python auto_merging_retrieval.py
```

**What happens:**
1. Loads the PDF and merges all pages into one document.
2. Builds (or reloads from disk) two indexes:
   - `merging_index_0/` — two-layer hierarchy (2048 / 512 tokens)
   - `merging_index_1/` — three-layer hierarchy (2048 / 512 / 128 tokens)
3. Runs a demo query against each index.
4. *(Optional)* Evaluates both engines with TruLens and launches a dashboard at `http://localhost:8501`.

### Sentence Window Retrieval

```bash
python sentence_window_retrieval.py
```

**What happens:**
1. Loads the PDF and merges all pages into one document.
2. Builds (or reloads from disk) two indexes:
   - `sentence_index_1/` — window of 1 surrounding sentence on each side
   - `sentence_index_3/` — window of 3 surrounding sentences on each side
3. Runs a demo query against each index.
4. *(Optional)* Evaluates both engines with TruLens and launches a dashboard at `http://localhost:8501`.

---

## Configuration

All tuneable parameters live at the top of each script:

| Variable | Default | Description |
|---|---|---|
| `PDF_PATH` | `eBook-How-to-Build-a-Career-in-AI.pdf` | Path to your input PDF |
| `EVAL_QUESTIONS_PATH` | `generated_questions.text` | Path to evaluation questions file |
| `EMBED_MODEL` | `local:BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | HuggingFace cross-encoder reranker |
| `LLM_MODEL` | `gpt-3.5-turbo` | OpenAI chat model |
| `LLM_TEMPERATURE` | `0.1` | LLM sampling temperature |

**Auto-merging specific:**

| Variable | Where set | Description |
|---|---|---|
| `chunk_sizes` | `build_automerging_index()` call | Token sizes per hierarchy layer, e.g. `[2048, 512, 128]` |
| `similarity_top_k` | `get_automerging_query_engine()` call | Leaf nodes retrieved before merging |
| `rerank_top_n` | `get_automerging_query_engine()` call | Nodes passed to the LLM after reranking |

**Sentence window specific:**

| Variable | Where set | Description |
|---|---|---|
| `sentence_window_size` | `build_sentence_window_index()` call | Sentences on each side included as context (1 or 3) |
| `similarity_top_k` | `get_sentence_window_query_engine()` call | Sentence nodes retrieved before window expansion |
| `rerank_top_n` | `get_sentence_window_query_engine()` call | Windows passed to the LLM after reranking |


---


## Re-runs

Indexes are persisted to disk. On subsequent runs each script loads existing indexes instead of rebuilding, making re-runs much faster. To force a full rebuild, delete the relevant index directories:

```bash
# Auto-merging
rm -rf merging_index_0 merging_index_1

# Sentence window
rm -rf sentence_index_1 sentence_index_3
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `EnvironmentError: OPENAI_API_KEY is not set` | Export the key as shown in Step 4 |
| `ModuleNotFoundError` | Re-run the `pip install` command from Step 3 |
| CUDA / torch errors on a CPU-only machine | Install CPU-only torch: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| TruLens dashboard doesn't open | Ensure `trulens-eval` is installed and port 8501 is free |
| Index loads stale data after changing the PDF | Delete the relevant `*_index_*/` directories and re-run |
| Null responses in TruLens evaluation | Re-run the evaluation — occasional API timeouts are normal; smaller batches help |