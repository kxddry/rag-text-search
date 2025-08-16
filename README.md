### RAG Text Search (CLI + TUI)

Interactive Retrieval-Augmented Search over local .txt and .md documents. Ingest one or more text files, build vector indexes (TF‑IDF by default, optional OpenAI‑compatible remote embeddings), and explore results in a terminal UI.

### Features
- **Interactive TUI**: Type your query and press Enter; use **Up/Down** arrows to switch between results; **Ctrl+C/Ctrl+D** to quit
- **Unicode-aware tokenization**: Works with many languages (uses `\p{L}` word matching)
- **Chunking**: Sentence-based chunker with configurable overlap
- **Embedders**:
  - TF‑IDF (default, local, fast)
  - OpenAI‑compatible remote embeddings (supports OpenAI API and Ollama-compatible servers)
- **Vector stores**:
  - In-memory (default)
  - Qdrant (HTTP API; collection auto-created if missing)
- **Summarization**: Frequency-based summarizer produces a short overview after ingestion
- **Configurable** via YAML; **.env** auto-loaded for secrets

### Requirements
- Go 1.24+
- Optional: Qdrant server if using the Qdrant vector store
- Optional: OpenAI-compatible embedding server and API key if using the remote embedder

### Install
```bash
git clone https://github.com/kxddry/rag-tts
cd rag
go build ./cmd/rag

```

### Quick start
```bash
# Ingest and search one or more .txt/.md files
./rag <files...>

# Shell expansion works too
./rag *.txt
```

On first run, a default config is created at `~/.config/rag/config.yaml` if none is found. You can also pass a custom config:
```bash
./rag --config=/path/to/config.yaml *.txt
```

### Usage
```text
rag [--config=config.yaml] file1.txt [file2.txt ...]

- Only .txt files are ingested; other extensions are ignored
- Enter runs the query; Up/Down arrows switch results; Left/Right edit the query as usual
- Quit with Ctrl+C or Ctrl+D
```

### Configuration
The app loads config in this order:
- `./config.yaml` (if present)
- `~/.config/rag/config.yaml` (created with defaults if missing)

An example config with all options:
```yaml
embedder:
  # "tfidf" (default) or "openai"
  type: tfidf
  openai:
    # Used when type == "openai"
    base_url: https://api.openai.com/v1 # can also use http://localhost:11451/api for Ollama
    api_key_env: OPENAI_API_KEY
    model: text-embedding-3-small
    timeout_secs: 30
    batch_size: 32

chunker:
  # currently only "sentence" is supported
  type: sentence
  sentences_per_chunk: 5
  overlap_sentences: 1

vector_store:
  # "memory" (default) or "qdrant"
  type: memory
  qdrant:
    url: ... # qdrant url
    api_key: "" # optional
    collection: rag_chunks
    distance: Cosine  # informational; implementation assumes cosine
    timeout_secs: 15

summarizer:
  # currently only "frequency" is supported
  type: frequency
  max_sentences: 5
```

The app auto-loads environment variables from a `.env` file in the working directory if present. For remote embeddings, set your key:
```dotenv
OPENAI_API_KEY=sk-...
```

### TUI Controls
- **Type**: Enter your query at the prompt
- **Enter**: Run the search
- **Up/Down**: Navigate between results (the text cursor will not move)
- **Left/Right**: Edit the query normally (do not switch results)
- **Ctrl+C/Ctrl+D**: Quit

The result view shows a relevance score and highlights the sentence that best matches your query terms.

### How it works (high-level)
1. **Ingest**
   - Loads the provided `.txt` files
   - Chunks by sentences with configurable overlap
   - Prepares the embedder (TF‑IDF builds a vocabulary and IDF table)
   - Initializes the vector store and upserts chunk vectors
   - Generates a brief summary of the full corpus for context
2. **Query**
   - Embeds the query and searches the vector store
   - If the query produces no signal (e.g., empty tokens), falls back to lexical ranking
   - Displays top results, with best-matching sentence highlighted

### OpenAI-compatible embeddings
- Select by setting `embedder.type: openai`
- Supports OpenAI responses and Ollama-compatible `{ "embedding": [...] }` responses
- Respects `Retry-After` and applies exponential backoff for 429/5xx
- Configure server via `base_url`, model via `model`, and API key via `api_key_env`

### Qdrant vector store
- Select by setting `vector_store.type: qdrant`
- Configure `url`, optional `api_key`, and `collection`
- The collection is created if missing; cosine similarity is used

### Development
```bash
# Build
go build ./...

# Run with config
go run ./cmd/rag --config=./config.yaml *.txt

# Lint (if you use golangci-lint or similar)
# golangci-lint run
```

Project layout highlights:
- `cmd/rag/`: CLI entrypoint (loads config, wires components, starts TUI)
- `internal/chunker/`: Sentence chunker
- `internal/embedding/`: TF‑IDF and OpenAI-compatible embedders
- `internal/vectorstore/`: In-memory and Qdrant stores
- `internal/summarizer/`: Frequency-based summarizer
- `internal/service/`: Orchestrates ingest and query
- `internal/tui/`: Bubbletea-based terminal UI

### License
See `LICENSE`.