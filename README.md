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
1. Hybrid search finds the 3 most relevant chunks: BM25 keyword search and dense vector search each score the stored chunks, and the two scores are fused, 70% dense and 30% BM25 after scaling each to 0-1 per question (see [Fusing BM25 and dense](#fusing-bm25-and-dense); set `RETRIEVAL_MODE=dense` for vector search only)
2. Retrieved chunks are used as context for the OpenAI LLM (gpt-4o-mini)
3. The model answers only from that context; when the chunks hold nothing relevant it replies "I don't know based on the provided documents." instead of guessing
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
- `hybrid`: the app's default retriever, the same code as the app in `utils/retrieval.py`. By default it fuses scores (`--fusion score`, `--dense-weight 0.7`); `--fusion rrf` gives the former reciprocal rank fusion (k = 60, 50 candidates each; [Cormack et al., 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)).
- `rerank`: hybrid's top 10 reordered by the optional cross-encoder (`--reranker minilm`, needs `requirements-rerank.txt`); the same code as the app's `RERANKER`, in `utils/reranking.py`.

### Baseline results

Full corpus (48 documents, 772 chunks), all 10,570 questions, seed 13, run before the ingestion and config fixes in #2 (single-pass chunking, content-hash dedup, settings read from `.env`, answer temperature 0.7 to 0.0) on 2026-10-09, and again after #2 and the chunk-metadata fix in #4 on 2026-10-10.

Retrieval moved by at most 0.002 across the fixes, so one table, from the latest run, covers all of them:

| Retriever | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|---|
| bm25 | **0.773** | 0.910 | 0.943 | 0.969 | **0.847** |
| dense (app default before hybrid) | 0.544 | 0.774 | 0.844 | 0.916 | 0.673 |
| hybrid, RRF (former default) | 0.700 | 0.890 | 0.939 | 0.975 | 0.802 |
| hybrid, score fusion (app default) | 0.752 | **0.917** | **0.953** | **0.978** | 0.839 |

Answer quality, 100 sampled questions through the app's `QueryEngine` (top 3, `gpt-4o-mini`, judged by `gpt-4o-mini`):

| Run | Retriever | Faithfulness | Fully faithful | No-claim answers | Answer recall | Context has answer |
|---|---|---|---|---|---|---|
| Before #2 (temperature 0.7) | dense | 0.960 | 95.5% | 12 / 100 | 0.67 | 0.72 |
| After #2 and #4 (temperature 0.0) | dense | 0.971 | 96.6% | 13 / 100 | 0.65 | 0.73 |
| Hybrid app, same-day dense rerun | dense | 0.977 | 97.7% | 13 / 100 | 0.65 | 0.73 |
| Hybrid app | hybrid, RRF | 1.000 | 100% | 5 / 100 | 0.77 | 0.87 |
| Score-fusion app | **hybrid, score fusion** | **1.000** | **100%** | 5 / 100 | **0.84** | **0.94** |

The dense rerun and the RRF row are the same 100 seeded questions on the same code, with only `--retrieval-mode` changed; the score-fusion row is the same questions with the shipped prompt and fusion. Differences between the dense runs are within noise for n = 100; the hybrid gains are not.

What the numbers say:

- **Retrieval was the bottleneck, not generation.** With dense retrieval only about 73% of answers had the gold answer in their top 3 chunks, and answer recall (0.65) sat just under that ceiling. Answers were wrong mostly because the right chunk was not retrieved.
- **Switching the app to hybrid retrieval fixed most of that.** Hybrid raises recall@3 from 0.774 to 0.890 for well under 1 ms more per query. In the app, the share of answers whose context held the gold answer rose from 0.73 to 0.87 and answer recall from 0.65 to 0.77. Every hybrid answer was fully faithful, and the 5 no-claim answers were the model saying the context did not contain the answer. Part of the gap is SQuAD's lexical bias (see caveats), so confirm on less lexical questions before treating BM25 as the winner on its own.
- **Every unfaithful dense answer came from a retrieval miss.** Both answers scored 0.0 in the post-#4 dense run had no gold chunk in context, and the model fell back on its own knowledge: once correctly (Sufism) and once not ("24 points" instead of 308). The earlier post-#2 run showed the same pattern in all four of its 0.0 answers. That motivated the answer prompt below.
- **A grounded answer prompt mostly makes "not in the documents" explicit.** The app's prompt (`utils/prompts.py`) answers only from the retrieved chunks and, when they hold nothing relevant, replies with one fixed sentence ("I don't know based on the provided documents.") that code can detect. On the hybrid app, same 100 questions, only the prompt changed:

  | Answer prompt | Faithfulness | Answer recall | Fixed "don't know" replies | ...with the gold chunk retrieved | Answered without the gold chunk |
  |---|---|---|---|---|---|
  | LlamaIndex default | 1.000 | 0.77 | 0 (5 declines, each worded differently) | 0 | 13 |
  | First draft ("do not guess") | 1.000 | 0.77 | 15 | 5 | 3 |
  | **Shipped** (answer when the context answers even partly) | **1.000** | **0.79** | 9 | 2 | 6 |

  With hybrid retrieval the default prompt already stayed faithful, so the gain is consistency rather than accuracy; the recall difference is two questions, within noise. The first draft was too cautious: it declined 5 questions whose answer it had been given, 4 of which the default answered correctly, so it was not shipped. "Answered without the gold chunk" counts strict span matches only, and the judge found every such answer supported by the retrieved text.
- **The judge is noisy at the edges.** Before the fixes it marked one correct, in-context answer unsupported; after, it split one fact into two claims and graded them differently. Treat single-example scores with care and compare aggregate runs.

Raw output lives in [`evaluation/results/`](evaluation/results/): `bm25_baseline.json` (offline run), `baseline.json` (before #2), `baseline_postfix.json` (after #2 and #4) `hybrid_app.json` (dense vs hybrid app, same questions), `score_fusion.json` (score fusion on both datasets), `abstain_prompt.json` (answer prompt comparison) and `abstain_nq.json` (wrong declines on NQ). A full run costs well under $1 (about 0.5M embedding tokens plus 200 `gpt-4o-mini` calls).

### Second dataset: Natural Questions

SQuAD's questions were written by annotators looking at the paragraph, so they share its words (see caveats). [Natural Questions](https://ai.google.com/research/NaturalQuestions) (Kwiatkowski et al., 2019, CC BY-SA 3.0) uses real Google search queries written before anyone saw the answer, so it tests whether retrieval holds up on questions phrased the way users phrase them. The harness loads the dev split as released for the [MRQA 2019 shared task](https://github.com/mrqa/MRQA-Shared-Task-2019) (Fisch et al., 2019), which puts NQ in SQuAD's span format, so relevance labels work exactly as above with no manual labeling. Pass `--dataset nq`; the file is downloaded on first run and cached in `data/eval/`.

How the loader (`evaluation/nq.py`) shapes it:

- Each distinct Wikipedia passage (NQ's "long answer": a paragraph, list or table) becomes one document, giving 5,332 documents and 5,670 chunks at the app's settings. That is a corpus of many short documents, unlike SQuAD's 48 long articles.
- NQ's HTML tags (`<P>`, `<Td>`, ...) are stripped so documents read like plain-text uploads, with rows, list items and paragraphs on their own lines, and answer offsets are remapped to the cleaned text.
- NQ dev has up to five annotators per question, and MRQA keeps each annotation as its own example (12,836 examples, 4,177 questions). The loader merges them into one query whose gold answers are the union of the annotators' spans, across whichever passages they chose, so a question does not count once per annotator.

BM25 on all 4,177 questions (seed 13, 2026-10-10):

| Dataset | Chunks | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|---|---|
| SQuAD v1.1 dev | 772 | 0.773 | 0.910 | 0.943 | 0.969 | 0.847 |
| NQ (MRQA dev) | 5,670 | 0.610 | 0.733 | 0.774 | 0.820 | 0.680 |

BM25 loses 18 points of recall@3 on real queries. Part of that is the larger corpus (more distractors), part is less word overlap. With embeddings the ranking flips (all 4,177 questions, from the reranker run below):

| NQ retriever | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|---|
| bm25 | 0.610 | 0.733 | 0.774 | 0.820 | 0.680 |
| dense | **0.811** | **0.918** | **0.942** | **0.965** | **0.868** |
| hybrid, RRF (former default) | 0.722 | 0.842 | 0.876 | 0.921 | 0.790 |
| hybrid, score fusion (app default) | 0.791 | 0.900 | 0.930 | 0.961 | 0.852 |

**On questions phrased the way users phrase them, dense retrieval beat the former RRF hybrid by 7.7 points of recall@3**, because equal-weight rank fusion lets the weak BM25 ranking pull good dense hits down. On SQuAD it is the reverse (hybrid 0.891, dense 0.774). Score fusion, the app's default now, closes most of that gap: 1.8 points behind dense on NQ, and the best recall@3 of any retriever on SQuAD. See [Fusing BM25 and dense](#fusing-bm25-and-dense).

To reproduce (needs `OPENAI_API_KEY`; about 0.9M embedding tokens, a few cents, plus 200 `gpt-4o-mini` calls for the answer run):

```
python -m evaluation.run --dataset nq --retrievers bm25,dense,hybrid --faithfulness 100 \
    --output evaluation/results/nq_baseline.json
```

Raw output of the BM25 run: [`evaluation/results/nq_bm25_baseline.json`](evaluation/results/nq_bm25_baseline.json).

### Fusing BM25 and dense

The two datasets disagree about BM25: it is the strongest single retriever on SQuAD and the weakest on NQ, so the way hybrid combines it with dense decides which kind of question the app serves well.

**Rejected: down-weighting BM25 in rank fusion.** Weighted RRF (BM25's 1/(k + rank) terms multiplied by a weight below 1; measured with a `--bm25-weights` sweep that is not part of the harness) only trades one dataset for the other. Every lower weight helps NQ and hurts SQuAD, and none comes near dense on NQ or equal-weight RRF on SQuAD. Rank fusion cannot tell a decisive keyword match from a weak one, because both arrive as a rank. recall@3, all questions:

| BM25 weight in RRF | SQuAD | NQ |
|---|---|---|
| 1 (former default) | 0.891 | 0.842 |
| 0.5 | 0.864 | 0.857 |
| 0.33 | 0.849 | 0.869 |
| 0.25 | 0.840 | 0.877 |
| dense alone, for reference | 0.774 | 0.919 |

**Shipped: score fusion.** For each question, the dense and BM25 scores of every candidate (the union of each retriever's top 50) are scaled to 0-1 and summed with weights. A strong keyword match can then lift a chunk, while weak keyword matches barely move the dense order. Every candidate gets an exact score from both sides: chunks only BM25 found are scored against the query embedding from their vectors stored in Chroma, so there is no second embedding call. The dense weight, recall@3 on all questions:

| Dense weight | SQuAD | NQ |
|---|---|---|
| 1.0 (dense only) | 0.774 | 0.919 |
| 0.9 | 0.845 | 0.921 |
| 0.8 | 0.892 | 0.916 |
| **0.7 (default)** | **0.917** | **0.900** |
| 0.6 | 0.930 | 0.881 |
| 0.5 | 0.933 | 0.857 |

0.7 has the smallest worst-case loss: within 1.6 points of the best weight on SQuAD and 2.1 on NQ. Against the former RRF default it gains 2.6 points on SQuAD and 5.8 on NQ, and through the app (same 100 questions per dataset) context has the answer rises from 0.87 to 0.94 on SQuAD and from 0.85 to 0.90 on NQ, with faithfulness 1.000 on both. The extra Chroma lookup and scaling cost about 1.5-3 ms per query over RRF (7.2 vs 5.8 ms on SQuAD, 15.0 vs 12.1 ms on NQ). The weight was chosen on these two datasets, so treat the margins as optimistic; `HYBRID_DENSE_WEIGHT` and `HYBRID_FUSION=rrf` let you change it. The 0.7 row is the shipped code's own run ([`evaluation/results/score_fusion.json`](evaluation/results/score_fusion.json)); the other weights come from a one-off sweep with the same exact scoring.

### When the app says it doesn't know

On NQ the app replied with the fixed "I don't know based on the provided documents." to 7 of 100 questions whose gold chunk it had retrieved (SQuAD: 3). Capturing the exact prompts showed a clean pipeline: one LLM call per question, no refine step, nothing truncated. The cause is the chunks. Each gold chunk answers its question only indirectly, next to two related chunks that don't:

- "where was part of the classic surfing movie endless summer filmed": the gold chunk lists the coasts the surfers travel to but never names the film.
- "who played the princess in kingsman secret service": "Hanna Alström ... appear as Crown Princess Tilde of Sweden", with no mention of Kingsman.
- "who wrote the theme song to law and order": Mike Post is "best known for his TV theme songs for such series as Law & Order", next to a chunk about another Law & Order fact and one about another theme song.

Six answer setups were run on those 7 questions, the 16 questions whose gold chunk was not retrieved (10 NQ, 6 SQuAD), and the remaining SQuAD decline, all with the same retrieved chunks (`gpt-4o-mini`, temperature 0):

| Answer setup | Answered of the 7 (correct) | Fixed declines on the 16 without the gold chunk | Unsupported claims on those 16 |
|---|---|---|---|
| **Shipped** (grounded prompt naming indirect answers) | 0 (0) | 9 | 0 |
| Other wordings with the fixed decline: a shorter prompt, the same split into system and user messages, and LlamaIndex's default plus one decline sentence | 0 (0) | 8 to 10 | 0 |
| Evidence first: quote the supporting sentence, decline when there is none | 0 (0) | 10 | 1 |
| LlamaIndex default, no fixed decline | 7 (5) | 0 (7 declines in free text the app cannot detect) | 0 |
| LlamaIndex default plus a second call that classifies the reply as an answer or a decline | 7 (5) | 7 | 0 |

**Kept: the grounded prompt.** Whenever the prompt allows a decline, `gpt-4o-mini` takes it on these near-miss chunks, whatever the wording or format. The only setup that answers them is one without a decline in the prompt, plus a second LLM call to detect declines. That would roughly gain 5 correct NQ answers per 100, but it turned 2 correct declines into wrong answers the context supports (Ronaldo's La Liga goals "286" for 309) and misread a hedged correct answer as a decline, and it doubles the LLM calls per query. For a document Q&A app, a decline on a chunk that never names the film is the safer error. The shipped wording ([`utils/prompts.py`](utils/prompts.py)) names indirect answers (tables, lists, picking the latest, "as of" dates) and leaves NQ unchanged; on SQuAD it cut wrong declines from 3 to 1, within noise for n = 100. Through the app, same 100 questions per dataset:

| Dataset | Faithfulness | Answer recall | Wrong declines | Answered without the gold chunk |
|---|---|---|---|---|
| SQuAD | 1.000 | 0.85 (was 0.84) | 1 (was 3) | 3 (was 4) |
| NQ | 1.000 | 0.61 (was 0.60) | 7 (was 7) | 4 (was 4) |

Summaries: [`evaluation/results/abstain_nq.json`](evaluation/results/abstain_nq.json). The variant sweep ran from a one-off script that is not part of the harness.

### Running it

```
pip install -r requirements-dev.txt
python -m pytest tests                      # offline unit tests
python -m evaluation.run --retrievers bm25  # offline baseline
python -m evaluation.run --dataset nq --retrievers bm25  # same on Natural Questions

# Needs OPENAI_API_KEY (read from .env)
python -m evaluation.run --retrievers bm25,dense,hybrid --faithfulness 100 \
    --output evaluation/results/baseline.json
```

`--max-articles` (alias `--max-documents`) and `--max-questions` shrink the run for quick iteration; `--chunk-size`, `--chunk-overlap`, `--embedding-model`, `--temperature`, `--judge-model`, `--retrieval-mode` (the retriever the faithfulness run's `QueryEngine` uses, `hybrid` by default), `--prompt` (`grounded`, the app's, or LlamaIndex's `default`), and `--reranker`, `--rerank-candidates` and `--rerank-base` for the optional reranker make it easy to compare variants against the baseline.

### Caveats

- SQuAD questions were written by annotators looking at the paragraph, so they share many words with it. That favors BM25, and absolute recall here will be higher than on real user questions. Use the numbers to compare variants, not as a production estimate.
- The judge defaults to the same model that writes the answers, which can inflate faithfulness. Pass a different `--judge-model` for a stricter check.
- The harness chunks once and builds a fresh index per run, so it never measured the double chunking or re-upload duplicates fixed in #2; that is why retrieval numbers did not move.

## Getting Started

### Prerequisites

- Python 3.10+ (3.12 recommended; the Docker image and CI use 3.12)
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
| `RETRIEVAL_MODE` | hybrid | `hybrid` (BM25 + dense, fused) or `dense` (vector search only) |
| `HYBRID_FUSION` | score | How hybrid fuses BM25 and dense: `score` (weighted sum of per-query scaled scores) or `rrf` (reciprocal rank fusion, the former default) |
| `HYBRID_DENSE_WEIGHT` | 0.7 | Weight on dense in score fusion, 0 to 1; BM25 gets the rest |
| `RERANKER` | off | `minilm` reranks the retrieved candidates with a cross-encoder before the top 3 go to the LLM. Optional extra, see [Optional reranker](#optional-reranker) |
| `RERANK_CANDIDATES` | 10 | Chunks retrieved for the reranker to choose from |

Changing `CHUNK_SIZE`, `CHUNK_OVERLAP` or `EMBEDDING_MODEL` only affects newly indexed documents; clear `data/chroma` to re-index everything.

Indexed documents persist in `data/chroma` and are available again after a restart. Each document is identified by a hash of its contents, so uploading the same file again does not create duplicate entries.

## Optional reranker

A cross-encoder reranker can reorder the retrieved chunks before the top 3 go to the LLM. It is off by default because it needs PyTorch (a few hundred MB with the CPU-only build, over 2 GB with pip's default CUDA build on Linux) and adds about 40-50 ms per question on a laptop CPU.

```
pip install -r requirements-rerank.txt   # sentence-transformers + PyTorch
RERANKER=minilm streamlit run app.py     # cross-encoder/ms-marco-MiniLM-L-6-v2
EXTRAS=rerank docker-compose up -d --build   # Docker image with CPU-only PyTorch and the model baked in
```

For Docker, also add `RERANKER=minilm` to `.env`: `EXTRAS` only installs the reranker into the image, and the container reads `RERANKER` at runtime.

The retriever returns `RERANK_CANDIDATES` chunks (10 by default), the cross-encoder scores each against the question, and the best 3 go to the LLM. It works with either `RETRIEVAL_MODE`.

Measured with `--retrievers rerank --reranker minilm` (recall@3, every evaluated question):

| Retrieval | SQuAD | NQ |
|---|---|---|
| dense | 0.773 | **0.919** |
| hybrid, RRF | 0.891 | 0.842 |
| hybrid, RRF + reranker | **0.911** | 0.874 |
| dense + reranker (`--rerank-base dense`) | 0.866 | 0.897 |

These were measured on the RRF hybrid, before score fusion became the default. The reranker improves it on both datasets (+2.0 and +3.2 points) and lifts SQuAD answers through the app (context has the answer 0.87 to 0.93, answer recall 0.79 to 0.83, same 100 questions). No configuration wins both datasets: on NQ, dense alone stays best and reranking its candidates costs 2.1 points, likely because this MS MARCO-trained model handles NQ's tables and lists less well than the embeddings do. Larger rerankers did not help in a 2,000-question trial (MiniLM-L-12: same recall@3; bge-reranker-base: 0.814 on SQuAD). Raw output: [`evaluation/results/reranker.json`](evaluation/results/reranker.json).

## Running Tests

```
pip install -r requirements-dev.txt
pytest
```

The tests use a mock embedding model and make no OpenAI calls. GitHub Actions runs them on every pull request and on pushes to `main`, on Python 3.12, the same version as the Docker image.

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
- Advanced filtering and structured queries
- API integration with external knowledge sources

### User Experience
- Conversation memory with follow-up question handling
- Chat-like interface with history
- Responsive design

