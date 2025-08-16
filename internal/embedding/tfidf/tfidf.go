package tfidf

import (
	"errors"
	"math"
	"regexp"
	"sort"
	"strings"
)

// Embedder implements a simple TF-IDF vectorizer.
// It builds a vocabulary from the corpus and computes IDF values.
type Embedder struct {
	vocabulary   map[string]int
	idf          []float64
	dimension    int
	prepared     bool
	tokenPattern *regexp.Regexp
	stopwords    map[string]struct{}
}

// NewEmbedder creates an unprepared TF-IDF embedder.
func NewEmbedder() *Embedder {
	return &Embedder{
		vocabulary:   make(map[string]int),
		tokenPattern: regexp.MustCompile(`\p{L}+(?:['’]\p{L}+)*`),
		stopwords:    defaultStopwords(),
	}
}

// Name returns the identifier of this embedder implementation.
func (e *Embedder) Name() string { return "tfidf" }

// Prepare builds the vocabulary and IDF values from the provided corpus.
func (e *Embedder) Prepare(corpus []string) error {
	if len(corpus) == 0 {
		return errors.New("empty corpus for TF-IDF prepare")
	}
	// Build vocabulary and document frequencies
	df := make(map[string]int)
	for _, text := range corpus {
		tokens := e.tokenize(text)
		seen := make(map[string]struct{})
		for _, tok := range tokens {
			if _, isStop := e.stopwords[tok]; isStop {
				continue
			}
			if _, ok := seen[tok]; ok {
				continue
			}
			seen[tok] = struct{}{}
			df[tok]++
		}
	}
	// Create stable ordering for vocabulary
	terms := make([]string, 0, len(df))
	for term := range df {
		terms = append(terms, term)
	}
	sort.Strings(terms)
	if len(terms) == 0 {
		return errors.New("no tokens found in corpus; ensure tokenizer supports your language")
	}
	e.vocabulary = make(map[string]int, len(terms))
	e.idf = make([]float64, len(terms))
	N := float64(len(corpus))
	for i, term := range terms {
		e.vocabulary[term] = i
		// Smoothed IDF
		e.idf[i] = math.Log((1+N)/(1+float64(df[term]))) + 1.0
	}
	e.dimension = len(terms)
	e.prepared = true
	return nil
}

// Dimension returns the dimensionality of the produced embedding vectors.
func (e *Embedder) Dimension() int { return e.dimension }

// Embed computes the TF-IDF embedding for the given text.
func (e *Embedder) Embed(text string) ([]float64, error) {
	if !e.prepared {
		return nil, errors.New("tfidf embedder not prepared")
	}
	vec := make([]float64, e.dimension)
	tokens := e.tokenize(text)
	tf := make(map[int]int)
	total := 0
	for _, tok := range tokens {
		if _, isStop := e.stopwords[tok]; isStop {
			continue
		}
		if idx, ok := e.vocabulary[tok]; ok {
			tf[idx]++
			total++
		}
	}
	if total == 0 {
		return vec, nil
	}
	for idx, count := range tf {
		tfv := float64(count) / float64(total)
		vec[idx] = tfv * e.idf[idx]
	}
	// L2 normalize
	norm := 0.0
	for _, v := range vec {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}
	return vec, nil
}

func (e *Embedder) tokenize(text string) []string {
	lower := strings.ToLower(text)
	raw := e.tokenPattern.FindAllString(lower, -1)
	if len(raw) == 0 {
		return nil
	}
	out := raw[:0]
	for _, t := range raw {
		if _, isStop := e.stopwords[t]; isStop {
			continue
		}
		out = append(out, t)
	}
	return out
}

func defaultStopwords() map[string]struct{} {
	words := []string{
		"a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "to", "of", "in", "on", "at", "by", "with", "as", "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these", "those", "from", "up", "down", "over", "under", "again", "further", "than", "so", "such", "into", "about", "between", "through", "during", "before", "after", "above", "below", "out", "off", "own", "same", "too", "very", "can", "will", "just", "don", "should", "now",
	}
	m := make(map[string]struct{}, len(words))
	for _, w := range words {
		m[w] = struct{}{}
	}
	return m
}
