package service

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"rag/internal/domain"
)

type RAGServiceImpl struct {
	chunker             domain.Chunker
	embedder            domain.Embedder
	store               domain.VectorStore
	summarizer          domain.Summarizer
	summaryMaxSentences int
	chunks              []domain.Chunk
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
			data, err := os.ReadFile(m)
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
	// Keep chunks for fallback ranking
	s.chunks = allChunks
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
	// Detect zero vector (no tokens)
	zero := true
	for _, v := range vec {
		if v != 0 {
			zero = false
			break
		}
	}
	if zero {
		return s.lexicalSearch(query, topK), nil
	}
	res, err := s.store.Search(vec, topK)
	if err != nil {
		return nil, err
	}
	allZero := true
	for _, r := range res {
		if r.Score > 1e-9 {
			allZero = false
			break
		}
	}
	if allZero {
		return s.lexicalSearch(query, topK), nil
	}
	return res, nil
}

var (
	unicodeWordRe = regexp.MustCompile(`\p{L}+(?:['’]\p{L}+)*`)
)

func (s *RAGServiceImpl) lexicalSearch(query string, topK int) []domain.SearchResult {
	qset := toTokenSet(query)
	type pair struct {
		idx   int
		score float64
	}
	scores := make([]pair, len(s.chunks))
	for i, ch := range s.chunks {
		scores[i] = pair{i, overlapOchiai(qset, ch.Text)}
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
	if topK <= 0 {
		topK = 5
	}
	if topK > len(scores) {
		topK = len(scores)
	}
	out := make([]domain.SearchResult, 0, topK)
	for i := 0; i < topK; i++ {
		p := scores[i]
		out = append(out, domain.SearchResult{Chunk: s.chunks[p.idx], Score: p.score})
	}
	return out
}

func toTokenSet(s string) map[string]struct{} {
	tokens := unicodeWordRe.FindAllString(strings.ToLower(s), -1)
	m := make(map[string]struct{}, len(tokens))
	for _, t := range tokens {
		m[t] = struct{}{}
	}
	return m
}

func overlapOchiai(qset map[string]struct{}, text string) float64 {
	stoks := unicodeWordRe.FindAllString(strings.ToLower(text), -1)
	seen := make(map[string]struct{}, len(stoks))
	inter := 0
	for _, t := range stoks {
		if _, ok := seen[t]; ok {
			continue
		}
		seen[t] = struct{}{}
		if _, ok := qset[t]; ok {
			inter++
		}
	}
	if len(qset) == 0 || len(seen) == 0 {
		return 0
	}
	// Ochiai coefficient: |A∩B| / sqrt(|A||B|)
	// sqrt sizes; use float64
	qa := float64(len(qset))
	ba := float64(len(seen))
	return float64(inter) / (sqrt(qa) * sqrt(ba))
}

func sqrt(x float64) float64 {
	// small inline sqrt to avoid extra imports
	// use Newton's method for a couple of iterations
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 6; i++ {
		z = 0.5 * (z + x/z)
	}
	return z
}

func hashString(s string) string {
	h := sha1.Sum([]byte(s))
	return hex.EncodeToString(h[:8])
}
