package summarizer

import (
	"math"
	"regexp"
	"sort"
	"strings"
)

// FrequencySummarizer ranks sentences by word frequency (stopwords filtered).
type FrequencySummarizer struct {
	tokenPattern *regexp.Regexp
	stopwords    map[string]struct{}
}

// NewFrequencySummarizer creates a frequency-based sentence ranker summarizer.
func NewFrequencySummarizer() *FrequencySummarizer {
	return &FrequencySummarizer{
		tokenPattern: regexp.MustCompile(`\p{L}+(?:['’]\p{L}+)*`),
		stopwords:    defaultStopwords(),
	}
}

// Summarize returns a short summary by ranking sentences using token frequency.
func (s *FrequencySummarizer) Summarize(text string, maxSentences int) (string, error) {
	if maxSentences <= 0 {
		maxSentences = 5
	}
	// Split into sentences
	sentences := regexp.MustCompile(`(?m)(?U)([^.!?]+[.!?])`).FindAllString(text, -1)
	if len(sentences) == 0 {
		return strings.TrimSpace(text), nil
	}
	// Compute word frequencies
	freq := map[string]float64{}
	for _, sent := range sentences {
		for _, tok := range s.tokens(sent) {
			if _, ok := s.stopwords[tok]; ok {
				continue
			}
			freq[tok]++
		}
	}
	// Normalize frequencies
	maxF := 0.0
	for _, v := range freq {
		if v > maxF {
			maxF = v
		}
	}
	if maxF > 0 {
		for k, v := range freq {
			freq[k] = v / maxF
		}
	}
	// Score sentences
	type pair struct {
		idx   int
		score float64
	}
	scores := make([]pair, len(sentences))
	for i, sent := range sentences {
		sscore := 0.0
		for _, tok := range s.tokens(sent) {
			if v, ok := freq[tok]; ok {
				sscore += v
			}
		}
		// Normalize by sentence length to avoid bias
		l := float64(len(s.tokens(sent)))
		if l > 0 {
			sscore /= math.Sqrt(l)
		}
		scores[i] = pair{i, sscore}
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
	if maxSentences > len(scores) {
		maxSentences = len(scores)
	}
	// Keep original order among selected
	selected := make([]int, maxSentences)
	for i := 0; i < maxSentences; i++ {
		selected[i] = scores[i].idx
	}
	sort.Ints(selected)
	var out []string
	for _, idx := range selected {
		out = append(out, strings.TrimSpace(sentences[idx]))
	}
	return strings.Join(out, " "), nil
}

func (s *FrequencySummarizer) tokens(text string) []string {
	lower := strings.ToLower(text)
	return s.tokenPattern.FindAllString(lower, -1)
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
