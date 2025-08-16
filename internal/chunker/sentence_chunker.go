package chunker

import (
	"regexp"
	"strconv"
	"strings"

	"rag-text-search/internal/domain"
)

// SentenceChunker splits text into sentence-based chunks with overlap.
type SentenceChunker struct {
	sentencesPerChunk int
	overlapSentences  int
	splitter          *regexp.Regexp
}

func NewSentenceChunker(sentencesPerChunk, overlapSentences int) *SentenceChunker {
	if sentencesPerChunk <= 0 {
		sentencesPerChunk = 5
	}
	if overlapSentences < 0 {
		overlapSentences = 0
	}
	return &SentenceChunker{
		sentencesPerChunk: sentencesPerChunk,
		overlapSentences:  overlapSentences,
		splitter:          regexp.MustCompile(`(?m)(?U)([^.!?]+[.!?])`),
	}
}

func (c *SentenceChunker) Chunk(document domain.Document) ([]domain.Chunk, error) {
	sentences := c.splitter.FindAllString(document.Content, -1)
	if len(sentences) == 0 {
		trimmed := strings.TrimSpace(document.Content)
		if trimmed == "" {
			return nil, nil
		}
		sentences = []string{trimmed}
	}
	// Trim spaces
	for i := range sentences {
		sentences[i] = strings.TrimSpace(sentences[i])
	}
	var chunks []domain.Chunk
	i := 0
	idx := 0
	for i < len(sentences) {
		end := i + c.sentencesPerChunk
		if end > len(sentences) {
			end = len(sentences)
		}
		text := strings.Join(sentences[i:end], " ")
		chunk := domain.Chunk{
			DocumentID: document.ID,
			ChunkID:    document.ID + ":" + strconv.Itoa(idx),
			Text:       text,
			Index:      idx,
		}
		chunks = append(chunks, chunk)
		if end == len(sentences) {
			break
		}
		i = end - c.overlapSentences
		if i < 0 {
			i = 0
		}
		idx++
	}
	return chunks, nil
}
