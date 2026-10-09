# Traceable PDF RAG Chatbot

A portable interview project that answers questions from ten 200+ page IPCC reports, shows the
retrieved evidence, and cites the source PDF and page. Embeddings, OCR and vector search run
locally; only the five selected passages are sent to the hosted generation model.

## What the demo shows

- PDF ingestion with native text extraction and Tesseract OCR through Docling.
- Deterministic, overlapping, provenance-rich chunks.
- Local BGE embeddings stored in Qdrant's HNSW index.
- Two-stage retrieval with a cross-encoder reranker.
- Grounded answers whose citation labels are validated by application code.
- A retrieval inspector, document inventory, and measured evaluation dashboard.

Read [APPROACH_AND_TECHNIQUES.md](APPROACH_AND_TECHNIQUES.md) before the interview. It explains
why every technique was selected and the alternatives considered. The exploratory work is in
[notebooks/01_rag_prototype.ipynb](notebooks/01_rag_prototype.ipynb).

## Windows demo setup

Prerequisites:

- Windows 10/11 with WSL2 enabled.
- Docker Desktop with Linux containers and Compose v2.
- At least 16 GB RAM and 30 GB free disk.
- Internet access for the hosted generator.
- A free Groq API key.

From PowerShell in the copied project directory:

```powershell
Copy-Item .env.example .env
notepad .env                 # set GROQ_API_KEY
.\scripts\verify-transfer.ps1 # present in a packaged transfer bundle
.\scripts\setup.ps1
```

Open <http://localhost:8501>. The setup builds the container from the source code in this
directory, starts Qdrant, restores the supplied snapshot when present, and then starts Streamlit.
The source is not hidden in a prebuilt application image.

The generated starter transfer ZIP contains source code, not the multi-gigabyte corpus or a
fabricated index. If no snapshot is present, build the real index on the destination laptop with:

```powershell
.\scripts\setup.ps1 -IngestIfMissing
```

That option downloads the official PDFs and models, then performs the full Docling/OCR ingestion.
It is a one-time offline preparation step and can take substantially longer than a query.

Subsequent starts:

```powershell
.\scripts\run.ps1
```

Stop everything without deleting the index:

```powershell
docker compose down
```

Do not use `docker compose down -v` unless you intend to delete the local Qdrant volume.

## Development setup

Docker is the supported and most reproducible path on both Linux and Windows:

```bash
cp .env.example .env
docker compose build app
docker compose up -d qdrant app
```

For a native environment, use Python 3.12 and install Tesseract first:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,notebook]'
```

## Corpus and ingestion

The corpus manifest contains ten official IPCC URLs. Downloads and model artifacts are ignored by
Git because they are large.

```bash
docker compose up -d qdrant
docker compose run --rm app rag-ingest --manifest data/corpus.yaml
```

Ingest one report while demonstrating the pipeline:

```bash
docker compose run --rm app rag-ingest --manifest data/corpus.yaml --document-id sr15
```

The ingester validates the PDF signature and minimum page count, computes SHA-256, converts with
Docling, stores cached JSONL chunks, embeds locally, and replaces only that document's Qdrant
points. Failed documents remain explicitly failed in `artifacts/ingestion-report.json`.

The first run downloads Docling and Hugging Face models. To create an offline-ready transfer after
ingestion, retain the `models/`, `data/corpus/`, and `artifacts/` directories; `rag-transfer`
automatically includes them when they exist.

Prepare all model caches explicitly before disconnecting the build machine:

```bash
docker compose run --rm app rag-models
```

## Evaluation and snapshots

```bash
docker compose run --rm app rag-evaluate
docker compose run --rm app rag-evaluate --with-generation
docker compose run --rm app rag-snapshot export
docker compose run --rm app rag-transfer --output transfer/traceable-pdf-rag.zip
```

Retrieval evaluation computes R@5, MRR and retrieval latency. `--with-generation` additionally
measures citation-label validity and refusal accuracy. Hallucination rate is deliberately left for
manual claim-level review; a citation being syntactically valid is not proof that it supports a
claim.

## Notebook

```bash
docker compose --profile notebook up notebook
```

Open <http://localhost:8888> and run `notebooks/01_rag_prototype.ipynb`. It uses a separate,
small experiment path and does not mutate the production Qdrant collection.

## Suggested demo sequence

1. Show the notebook and its extraction/chunk-size/retrieval experiments.
2. Open Documents and show ten validated PDFs, page count, chunks and checksums.
3. Ask: “What risks does sea-level rise create for low-lying coastal communities?”
4. Expand citations and show the page numbers, excerpts and bounding boxes.
5. Open Retrieval Inspector and compare dense versus reranker scores.
6. Ask an out-of-corpus question and show the evidence-based refusal.
7. Open Evaluation and show R@5, MRR, citation validity and p95 latency.

## Troubleshooting

- **`GROQ_API_KEY is not configured`:** add it to `.env`, then restart the app container.
- **Qdrant connection failure:** run `docker compose ps` and `docker compose up -d qdrant`.
- **Empty Documents tab:** restore a snapshot or run ingestion.
- **Slow first query:** model loading is a cold start; warm the app with two questions before demo.
- **Port already used:** change the host side of `8501:8501` or `6333:6333` in Compose.
- **Docker out of memory:** allocate at least 8 GB to Docker Desktop and ingest one PDF at a time.
- **OCR failure:** verify `tesseract --version` inside the app container.
