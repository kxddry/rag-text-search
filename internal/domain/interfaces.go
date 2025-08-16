package domain

// Document represents a single text file loaded into the system.
type Document struct {
	ID      string
	Path    string
	Content string
}

// Chunk is a semantically meaningful part of a document used for indexing.
type Chunk struct {
	DocumentID string
	ChunkID    string
	Text       string
	Index      int
}

// SearchResult represents a matching chunk with a relevance score.
type SearchResult struct {
	Chunk Chunk
	Score float64
}

// Embedder converts free text into a numeric vector representation.
// Implementations may require a preparation phase over the corpus.
type Embedder interface {
	Name() string
	Prepare(corpus []string) error
	Dimension() int
	Embed(text string) ([]float64, error)
}

// Chunker splits documents into chunks suitable for retrieval indexing.
type Chunker interface {
	Chunk(document Document) ([]Chunk, error)
}

// VectorStore persists vectors and supports similarity search.
type VectorStore interface {
	Init(dimension int) error
	Upsert(chunks []Chunk, vectors [][]float64) error
	Search(vector []float64, topK int) ([]SearchResult, error)
	Clear() error
}

// Summarizer produces a brief summary of the provided text.
type Summarizer interface {
	Summarize(text string, maxSentences int) (string, error)
}

// RAGService defines the operations exposed by the application core.
type RAGService interface {
	IngestDocuments(paths []string) (summary string, err error)
	Query(query string, topK int) ([]SearchResult, error)
}
