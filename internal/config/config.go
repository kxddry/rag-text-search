package config

import (
	"errors"
	"io/ioutil"
	"os"

	"gopkg.in/yaml.v3"
)

type EmbedderConfig struct {
	Type string `yaml:"type"`
}

type ChunkerConfig struct {
	Type              string `yaml:"type"`
	SentencesPerChunk int    `yaml:"sentences_per_chunk"`
	OverlapSentences  int    `yaml:"overlap_sentences"`
}

type VectorStoreConfig struct {
	Type   string        `yaml:"type"`
	Qdrant *QdrantConfig `yaml:"qdrant,omitempty"`
}

type QdrantConfig struct {
	URL         string `yaml:"url"`
	APIKey      string `yaml:"api_key"`
	Collection  string `yaml:"collection"`
	Distance    string `yaml:"distance"`
	TimeoutSecs int    `yaml:"timeout_secs"`
}

type SummarizerConfig struct {
	Type         string `yaml:"type"`
	MaxSentences int    `yaml:"max_sentences"`
}

type AppConfig struct {
	Embedder    EmbedderConfig    `yaml:"embedder"`
	Chunker     ChunkerConfig     `yaml:"chunker"`
	VectorStore VectorStoreConfig `yaml:"vector_store"`
	Summarizer  SummarizerConfig  `yaml:"summarizer"`
}

func Load(path string) (*AppConfig, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			// Default configuration if no file is present
			return &AppConfig{
				Embedder:    EmbedderConfig{Type: "tfidf"},
				Chunker:     ChunkerConfig{Type: "sentence", SentencesPerChunk: 5, OverlapSentences: 1},
				VectorStore: VectorStoreConfig{Type: "memory"},
				Summarizer:  SummarizerConfig{Type: "frequency", MaxSentences: 5},
			}, nil
		}
		return nil, err
	}
	var cfg AppConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	if cfg.Chunker.SentencesPerChunk == 0 {
		cfg.Chunker.SentencesPerChunk = 5
	}
	return &cfg, nil
}
