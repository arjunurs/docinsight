# DocInsight

A simple Retrieval Augmented Generation (RAG) application that allows users to upload PDF and text files, index their contents, and perform semantic searches or ask questions based on the uploaded documents.

## Features

- Upload PDF and text files
- Process and index document content
- Ask questions about your documents
- Get answers with source references
- Clean and intuitive user interface

## Technology Stack

- **LlamaIndex**: Main framework for document indexing and retrieval
- **ChromaDB**: Vector database for storing and searching document embeddings
- **Streamlit**: Frontend framework for the user interface
- **OpenAI API**: For generating embeddings and LLM responses

## Implementation Details

### Document Processing

Documents are processed through the following workflow:
1. Files are loaded using LlamaIndex's SimpleDirectoryReader
2. Each document gets a stable id derived from a hash of the file's contents
3. Text is chunked once, using SentenceSplitter with configurable chunk size and overlap
4. Chunks from documents not already in the index are embedded with OpenAI's embedding model
5. The chunks and their embeddings are stored in ChromaDB, which persists across restarts

### Retrieval and Response Generation

When a user asks a question:
1. The query is processed through the VectorStoreIndex to retrieve relevant document chunks
2. Retrieved chunks are used as context for the OpenAI LLM (gpt-4o-mini)
3. The model generates a response based on the provided context
4. Both the answer and source references are displayed to the user

## Evaluation

DocInsight ships with an offline harness in [`evaluation/`](evaluation/) that measures retrieval quality and answer faithfulness on a public dataset, so changes to chunking, embeddings or retrieval can be judged by numbers rather than by eyeballing answers.

### Dataset

[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) dev set (Rajpurkar et al., 2016, CC BY-SA 4.0): 48 Wikipedia articles and 10,570 questions. It is downloaded on first run and cached in `data/eval/` (not committed).

Each article is ingested as one document, the way a user uploads a whole file, and chunked with the app's settings (`SentenceSplitter`, 512 tokens, 50 overlap), which gives 772 chunks. The harness chunks each article through the app's own ingestion path (`DocumentProcessor`, exactly as an upload is processed), so the numbers measure what the app indexes. SQuAD records the character offset of every answer, so a chunk counts as relevant when it fully contains a gold answer span. Relevance labels therefore follow whatever chunking is being tested, and no manual labeling is needed. Questions whose answer is split across every chunk boundary are reported and skipped (none at the default settings).

### Metrics

| Metric | Definition |
|---|---|
| recall@k | Share of questions with a relevant chunk in the top k. The app retrieves 3 chunks, so recall@3 is the operating point. |
| MRR@10 | Mean of 1 / rank of the first relevant chunk, 0 if none in the top 10. |
| Faithfulness | The judge LLM splits each generated answer into atomic claims and checks each against the retrieved context; score = supported / total claims (the RAGAS definition, [Es et al., 2023](https://arxiv.org/abs/2309.15217)). Answers with no factual claims are counted separately. |
| Answer recall, token F1 | Whether the normalized gold answer appears in the generated answer, and SQuAD token F1. These separate "faithful but wrong" from "faithful and right". |

Faithfulness runs the app's own `QueryEngine` (top 3, `compact` mode) over a seeded sample of questions. Chunking, model and temperature default to the app's `AppConfig`, so a run measures what the app ships; each can be overridden by a CLI flag.

### Retrievers

- `bm25`: lexical baseline (`rank-bm25`), runs offline with no API key.
- `dense`: the app's stack, OpenAI `text-embedding-3-small` into ChromaDB.
- `hybrid`: reciprocal rank fusion of the two (k = 60, 50 candidates each; [Cormack et al., 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)).

### Baseline results

Full corpus (48 documents, 772 chunks), all 10,570 questions, seed 13, run before the ingestion and config fixes in #2 (single-pass chunking, content-hash dedup, settings read from `.env`, answer temperature 0.7 to 0.0) on 2026-10-09, and again after #2 and the chunk-metadata fix in #4 on 2026-10-10.

Retrieval moved by at most 0.002 across the fixes, so one table, from the latest run, covers all of them:

| Retriever | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|---|
| bm25 | 0.773 | **0.910** | **0.943** | 0.969 | **0.847** |
| dense (app default) | 0.544 | 0.774 | 0.844 | 0.916 | 0.673 |
| hybrid (RRF) | 0.700 | 0.890 | 0.939 | **0.975** | 0.802 |

Answer quality, 100 sampled questions through the app's `QueryEngine` (dense, top 3, `gpt-4o-mini`, judged by `gpt-4o-mini`):

| Run | Faithfulness | Fully faithful | No-claim answers | Answer recall | Context has answer |
|---|---|---|---|---|---|
| Before #2 (temperature 0.7) | 0.960 | 95.5% | 12 / 100 | 0.67 | 0.72 |
| After #2 and #4 (temperature 0.0) | 0.971 | 96.6% | 13 / 100 | 0.65 | 0.73 |

The differences between runs are within noise for n = 100.

What the numbers say:

- **Retrieval is the bottleneck, not generation.** Only about 73% of answers had the gold answer in their top 3 chunks, and answer recall (0.65) sits just under that ceiling. Answers are wrong mostly because the right chunk was not retrieved.
- **The app's dense retriever is the weakest of the three on this data.** Switching to hybrid retrieval raises recall@3 from 0.774 to 0.890 for well under 1 ms more per query. Part of that gap is SQuAD's lexical bias (see caveats), so confirm on less lexical questions before treating BM25 as the winner on its own.
- **Every unfaithful answer after the fixes came from a retrieval miss.** Both answers scored 0.0 in the latest run had no gold chunk in context, and the model fell back on its own knowledge: once correctly (Sufism) and once not ("24 points" instead of 308). The earlier post-#2 run showed the same pattern in all four of its 0.0 answers. An answer prompt that says "reply that you don't know when the context lacks the answer" is the obvious next change to measure.
- **The judge is noisy at the edges.** Before the fixes it marked one correct, in-context answer unsupported; after, it split one fact into two claims and graded them differently. Treat single-example scores with care and compare aggregate runs.

Raw output lives in [`evaluation/results/`](evaluation/results/): `bm25_baseline.json` (offline run), `baseline.json` (before #2) and `baseline_postfix.json` (after #2 and #4). A full run costs well under $1 (about 0.5M embedding tokens plus 200 `gpt-4o-mini` calls).

### Running it

```
pip install -r requirements-dev.txt
python -m pytest tests                      # offline unit tests
python -m evaluation.run --retrievers bm25  # offline baseline

# Needs OPENAI_API_KEY (read from .env)
python -m evaluation.run --retrievers bm25,dense,hybrid --faithfulness 100 \
    --output evaluation/results/baseline.json
```

`--max-articles` and `--max-questions` shrink the run for quick iteration; `--chunk-size`, `--chunk-overlap`, `--embedding-model`, `--temperature` and `--judge-model` make it easy to compare variants against the baseline.

### Caveats

- SQuAD questions were written by annotators looking at the paragraph, so they share many words with it. That favors BM25, and absolute recall here will be higher than on real user questions. Use the numbers to compare variants, not as a production estimate.
- The judge defaults to the same model that writes the answers, which can inflate faithfulness. Pass a different `--judge-model` for a stricter check.
- The harness chunks once and builds a fresh index per run, so it never measured the double chunking or re-upload duplicates fixed in #2; that is why retrieval numbers did not move.

## Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key

### Installation

#### Using Docker (Recommended)

1. Clone this repository:
   ```
   git clone https://github.com/arjunurs/docinsight.git
   cd docinsight
   ```

2. Create a `.env` file with your OpenAI API key:
   ```
   cp .env.example .env
   # Edit .env with your actual API key
   ```

3. Build and run the Docker container:
   ```
   docker-compose up -d
   ```

4. Access the application at http://localhost:8501

#### Manual Installation

1. Clone this repository:
   ```
   git clone https://github.com/arjunurs/docinsight.git
   cd docinsight
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your OpenAI API key:
   ```
   cp .env.example .env
   # Edit .env with your actual API key
   ```

5. Run the application:
   ```
   streamlit run app.py
   ```

6. Access the application at http://localhost:8501

## Usage

1. Upload one or more PDF or text files
2. Click "Process Documents" to index them
3. Enter your question in the text field
4. Click "Ask" to get answers based on your documents

## Configuration

All settings are optional environment variables, read at startup (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHUNK_SIZE` | 512 | Tokens per chunk |
| `CHUNK_OVERLAP` | 50 | Overlapping tokens between chunks |
| `EMBEDDING_MODEL` | text-embedding-3-small | OpenAI embedding model |
| `LLM_MODEL` | gpt-4o-mini | OpenAI model used to answer |
| `LLM_TEMPERATURE` | 0.0 | Kept at 0 so answers stay grounded and repeatable |

Changing `CHUNK_SIZE`, `CHUNK_OVERLAP` or `EMBEDDING_MODEL` only affects newly indexed documents; clear `data/chroma` to re-index everything.

Indexed documents persist in `data/chroma` and are available again after a restart. Each document is identified by a hash of its contents, so uploading the same file again does not create duplicate entries.

## Running Tests

```
pip install -r requirements-dev.txt
pytest
```

The tests use a mock embedding model and make no OpenAI calls.

## Troubleshooting

If you encounter any issues:

1. Make sure your OpenAI API key is correctly set in the .env file
2. For Docker deployment, ensure ports 8501 is not already in use
3. If you see errors related to document processing, check that your PDF files are text-based and not scanned images
4. Restart the application with `docker-compose down && docker-compose up -d` if needed

## Future Enhancements

### Document Handling
- Support for additional file formats (DOCX, HTML)
- Metadata extraction from documents
- Document chunking optimization
- Multi-modal support (images, audio)

### Search & Retrieval
- Hybrid search capabilities (keyword + semantic)
- Advanced filtering and structured queries
- API integration with external knowledge sources

### User Experience
- Conversation memory with follow-up question handling
- Chat-like interface with history
- Responsive design

