package qdrant

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"rag/internal/domain"
)

// Storage is a minimal REST client to Qdrant.
// It assumes cosine distance and creates the collection if missing.
type Storage struct {
	url        string
	apiKey     string
	collection string
	dimension  int
	client     *http.Client
}

type Config struct {
	URL        string
	APIKey     string
	Collection string
	Timeout    time.Duration
}

func NewStorage(cfg Config) *Storage {
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 15 * time.Second
	}
	return &Storage{
		url:        cfg.URL,
		apiKey:     cfg.APIKey,
		collection: cfg.Collection,
		client:     &http.Client{Timeout: timeout},
	}
}

func (s *Storage) Init(dimension int) error {
	if dimension <= 0 {
		return errors.New("invalid dimension")
	}
	s.dimension = dimension
	// Create collection if not exists
	body := map[string]any{
		"vectors": map[string]any{
			"size":     dimension,
			"distance": "Cosine",
		},
	}
	if err := s.putJSON(fmt.Sprintf("%s/collections/%s", s.url, s.collection), body); err != nil {
		// Qdrant returns 200 OK if collection exists with same schema; if error, propagate
		return err
	}
	return nil
}

func (s *Storage) Upsert(chunks []domain.Chunk, vectors [][]float64) error {
	if len(chunks) != len(vectors) {
		return errors.New("chunks and vectors length mismatch")
	}
	points := make([]map[string]any, len(chunks))
	for i := range chunks {
		points[i] = map[string]any{
			"id":     fmt.Sprintf("%s:%d", chunks[i].DocumentID, chunks[i].Index),
			"vector": vectors[i],
			"payload": map[string]any{
				"document_id": chunks[i].DocumentID,
				"chunk_id":    chunks[i].ChunkID,
				"index":       chunks[i].Index,
				"text":        chunks[i].Text,
			},
		}
	}
	body := map[string]any{"points": points}
	return s.putJSON(fmt.Sprintf("%s/collections/%s/points?wait=true", s.url, s.collection), body)
}

func (s *Storage) Search(vector []float64, topK int) ([]domain.SearchResult, error) {
	if topK <= 0 {
		topK = 5
	}
	req := map[string]any{
		"vector":       vector,
		"limit":        topK,
		"with_payload": true,
	}
	var resp struct {
		Result []struct {
			Score   float64        `json:"score"`
			Payload map[string]any `json:"payload"`
		} `json:"result"`
	}
	if err := s.postJSON(fmt.Sprintf("%s/collections/%s/points/search", s.url, s.collection), req, &resp); err != nil {
		return nil, err
	}
	results := make([]domain.SearchResult, 0, len(resp.Result))
	for _, r := range resp.Result {
		chunk := domain.Chunk{}
		if v, ok := r.Payload["document_id"].(string); ok {
			chunk.DocumentID = v
		}
		if v, ok := r.Payload["chunk_id"].(string); ok {
			chunk.ChunkID = v
		}
		if v, ok := r.Payload["index"].(float64); ok {
			chunk.Index = int(v)
		}
		if v, ok := r.Payload["text"].(string); ok {
			chunk.Text = v
		}
		results = append(results, domain.SearchResult{Chunk: chunk, Score: r.Score})
	}
	return results, nil
}

func (s *Storage) Clear() error {
	// Best-effort: drop collection
	req, _ := http.NewRequest(http.MethodDelete, fmt.Sprintf("%s/collections/%s", s.url, s.collection), nil)
	if s.apiKey != "" {
		req.Header.Set("api-key", s.apiKey)
	}
	_, _ = s.client.Do(req)
	return nil
}

func (s *Storage) putJSON(url string, body any) error {
	data, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPut, url, bytes.NewReader(data))
	req.Header.Set("Content-Type", "application/json")
	if s.apiKey != "" {
		req.Header.Set("api-key", s.apiKey)
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("qdrant PUT %s failed: %s", url, resp.Status)
	}
	return nil
}

func (s *Storage) postJSON(url string, body any, out any) error {
	data, _ := json.Marshal(body)
	req, _ := http.NewRequest(http.MethodPost, url, bytes.NewReader(data))
	req.Header.Set("Content-Type", "application/json")
	if s.apiKey != "" {
		req.Header.Set("api-key", s.apiKey)
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("qdrant POST %s failed: %s", url, resp.Status)
	}
	if out != nil {
		dec := json.NewDecoder(resp.Body)
		return dec.Decode(out)
	}
	return nil
}
