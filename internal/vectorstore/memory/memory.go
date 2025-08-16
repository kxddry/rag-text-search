package memory

import (
	"errors"
	"sync"

	"rag/internal/domain"
)

// Storage is a simple in-memory vector store using brute-force cosine similarity.
type Storage struct {
	mu        sync.RWMutex
	dimension int
	vectors   [][]float64
	chunks    []domain.Chunk
}

func NewStorage() *Storage { return &Storage{} }

func (s *Storage) Init(dimension int) error {
	if dimension <= 0 {
		return errors.New("invalid dimension")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.dimension = dimension
	s.vectors = nil
	s.chunks = nil
	return nil
}

func (s *Storage) Upsert(chunks []domain.Chunk, vectors [][]float64) error {
	if len(chunks) != len(vectors) {
		return errors.New("chunks and vectors length mismatch")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, v := range vectors {
		if len(v) != s.dimension {
			return errors.New("vector dimension mismatch")
		}
	}
	s.chunks = append(s.chunks, chunks...)
	s.vectors = append(s.vectors, vectors...)
	return nil
}

func (s *Storage) Search(vector []float64, topK int) ([]domain.SearchResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if topK <= 0 {
		topK = 5
	}
	// compute cosine similarity (vectors are assumed L2-normalized)
	scores := make([]float64, len(s.vectors))
	for i := range s.vectors {
		scores[i] = dot(s.vectors[i], vector)
	}
	// Get topK indexes
	idxs := argsortDesc(scores)
	if topK > len(idxs) {
		topK = len(idxs)
	}
	results := make([]domain.SearchResult, 0, topK)
	for i := 0; i < topK; i++ {
		j := idxs[i]
		results = append(results, domain.SearchResult{Chunk: s.chunks[j], Score: scores[j]})
	}
	return results, nil
}

func (s *Storage) Clear() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.vectors = nil
	s.chunks = nil
	return nil
}

func dot(a, b []float64) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func argsortDesc(vals []float64) []int {
	idxs := make([]int, len(vals))
	for i := range vals {
		idxs[i] = i
	}
	// Partial selection sort for small topK could be faster, but we keep it simple here
	// Stable sort not required
	quicksort(idxs, vals, 0, len(idxs)-1)
	return idxs
}

func quicksort(idxs []int, vals []float64, lo, hi int) {
	if lo >= hi {
		return
	}
	i, j := lo, hi
	pivot := vals[idxs[(lo+hi)/2]]
	for i <= j {
		for vals[idxs[i]] > pivot { // desc order
			i++
		}
		for vals[idxs[j]] < pivot {
			j--
		}
		if i <= j {
			idxs[i], idxs[j] = idxs[j], idxs[i]
			i++
			j--
		}
	}
	if lo < j {
		quicksort(idxs, vals, lo, j)
	}
	if i < hi {
		quicksort(idxs, vals, i, hi)
	}
}
