package service

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"strings"

	"rag-text-search/internal/domain"
)

type RAGServiceImpl struct {
	chunker             domain.Chunker
	embedder            domain.Embedder
	store               domain.VectorStore
	summarizer          domain.Summarizer
	summaryMaxSentences int
}

func NewRAGService(chunker domain.Chunker, embedder domain.Embedder, store domain.VectorStore, summarizer domain.Summarizer, summaryMaxSentences int) *RAGServiceImpl {
	return &RAGServiceImpl{chunker: chunker, embedder: embedder, store: store, summarizer: summarizer, summaryMaxSentences: summaryMaxSentences}
}

func (s *RAGServiceImpl) IngestDocuments(paths []string) (string, error) {
	var documents []domain.Document
	for _, p := range paths {
		matches, _ := filepath.Glob(p)
		if matches == nil {
			matches = []string{p}
		}
		for _, m := range matches {
			if !strings.HasSuffix(strings.ToLower(m), ".txt") {
				continue
			}
			data, err := ioutil.ReadFile(m)
			if err != nil {
				return "", err
			}
			id := hashString(m)
			documents = append(documents, domain.Document{ID: id, Path: m, Content: string(data)})
		}
	}
	if len(documents) == 0 {
		return "", fmt.Errorf("no .txt documents found")
	}
	// Chunk
	var allChunks []domain.Chunk
	var allTexts []string
	var allTextConcat strings.Builder
	for _, d := range documents {
		chunks, err := s.chunker.Chunk(d)
		if err != nil {
			return "", err
		}
		for _, ch := range chunks {
			allChunks = append(allChunks, ch)
			allTexts = append(allTexts, ch.Text)
		}
		allTextConcat.WriteString("\n")
		allTextConcat.WriteString(d.Content)
	}
	// Prepare embedder with corpus
	if err := s.embedder.Prepare(allTexts); err != nil {
		return "", err
	}
	if err := s.store.Init(s.embedder.Dimension()); err != nil {
		return "", err
	}
	// Embed and upsert
	vectors := make([][]float64, len(allChunks))
	for i := range allChunks {
		vec, err := s.embedder.Embed(allChunks[i].Text)
		if err != nil {
			return "", err
		}
		vectors[i] = vec
	}
	if err := s.store.Clear(); err != nil {
		return "", err
	}
	if err := s.store.Upsert(allChunks, vectors); err != nil {
		return "", err
	}
	// Summarize
	summary, err := s.summarizer.Summarize(allTextConcat.String(), s.summaryMaxSentences)
	if err != nil {
		return "", err
	}
	return summary, nil
}

func (s *RAGServiceImpl) Query(query string, topK int) ([]domain.SearchResult, error) {
	vec, err := s.embedder.Embed(query)
	if err != nil {
		return nil, err
	}
	return s.store.Search(vec, topK)
}

func hashString(s string) string {
	h := sha1.Sum([]byte(s))
	return hex.EncodeToString(h[:8])
}
