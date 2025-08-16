package vectorstore

import "rag/internal/domain"

// Storage persists vectors and supports similarity search.
type Storage interface {
	Init(dimension int) error
	Upsert(chunks []domain.Chunk, vectors [][]float64) error
	Search(vector []float64, topK int) ([]domain.SearchResult, error)
	Clear() error
}
