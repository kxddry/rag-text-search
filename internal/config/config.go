package config

import (
	"errors"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// OpenAIEmbedderConfig holds configuration for the OpenAI-compatible embedder.
type OpenAIEmbedderConfig struct {
	BaseURL     string `yaml:"base_url"`
	APIKeyEnv   string `yaml:"api_key_env"`
	Model       string `yaml:"model"`
	TimeoutSecs int    `yaml:"timeout_secs"`
	BatchSize   int    `yaml:"batch_size"`
}

// EmbedderConfig selects and configures the text embedder implementation.
type EmbedderConfig struct {
	Type   string                `yaml:"type"`
	OpenAI *OpenAIEmbedderConfig `yaml:"openai,omitempty"`
}

// ChunkerConfig configures how documents are split into chunks.
type ChunkerConfig struct {
	Type              string `yaml:"type"`
	SentencesPerChunk int    `yaml:"sentences_per_chunk"`
	OverlapSentences  int    `yaml:"overlap_sentences"`
}

// VectorStoreConfig selects and configures the vector store implementation.
type VectorStoreConfig struct {
	Type   string        `yaml:"type"`
	Qdrant *QdrantConfig `yaml:"qdrant,omitempty"`
}

// QdrantConfig contains connection details for a Qdrant vector store.
type QdrantConfig struct {
	URL         string `yaml:"url"`
	APIKey      string `yaml:"api_key"`
	Collection  string `yaml:"collection"`
	Distance    string `yaml:"distance"`
	TimeoutSecs int    `yaml:"timeout_secs"`
}

// SummarizerConfig selects and configures the summarizer.
type SummarizerConfig struct {
	Type         string `yaml:"type"`
	MaxSentences int    `yaml:"max_sentences"`
}

// AppConfig is the root application configuration structure.
type AppConfig struct {
	Embedder    EmbedderConfig    `yaml:"embedder"`
	Chunker     ChunkerConfig     `yaml:"chunker"`
	VectorStore VectorStoreConfig `yaml:"vector_store"`
	Summarizer  SummarizerConfig  `yaml:"summarizer"`
}

// Load reads a config from a specified path. If the file does not exist, returns defaults.
func Load(path string) (*AppConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			cfg := defaultConfig()
			return cfg, nil
		}
		return nil, err
	}
	var cfg AppConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	applyConfigDefaults(&cfg)
	return &cfg, nil
}

// LoadDefault tries ./config.yaml first, then ~/.config/rag/config.yaml.
// If neither exists, it writes defaults to ~/.config/rag/config.yaml and returns them.
func LoadDefault() (*AppConfig, string, error) {
	cwdPath := "config.yaml"
	if _, err := os.Stat(cwdPath); err == nil {
		cfg, err := Load(cwdPath)
		return cfg, cwdPath, err
	}
	userPath, err := defaultUserConfigPath()
	if err != nil {
		return nil, "", err
	}
	if _, err := os.Stat(userPath); err == nil {
		cfg, err := Load(userPath)
		return cfg, userPath, err
	}
	cfg := defaultConfig()
	if err := Save(userPath, cfg); err != nil {
		return nil, "", err
	}
	return cfg, userPath, nil
}

// Save writes the config to the given path, creating directories as needed.
func Save(path string, cfg *AppConfig) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func defaultUserConfigPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".config", "rag", "config.yaml"), nil
}

func defaultConfig() *AppConfig {
	cfg := &AppConfig{
		Embedder:    EmbedderConfig{Type: "tfidf"},
		Chunker:     ChunkerConfig{Type: "sentence", SentencesPerChunk: 5, OverlapSentences: 1},
		VectorStore: VectorStoreConfig{Type: "memory"},
		Summarizer:  SummarizerConfig{Type: "frequency", MaxSentences: 5},
	}
	return cfg
}

func applyConfigDefaults(cfg *AppConfig) {
	if cfg.Chunker.SentencesPerChunk == 0 {
		cfg.Chunker.SentencesPerChunk = 5
	}
	if cfg.Embedder.Type == "openai" && cfg.Embedder.OpenAI != nil {
		if cfg.Embedder.OpenAI.BaseURL == "" {
			cfg.Embedder.OpenAI.BaseURL = "https://api.openai.com/v1"
		}
		if cfg.Embedder.OpenAI.APIKeyEnv == "" {
			cfg.Embedder.OpenAI.APIKeyEnv = "OPENAI_API_KEY"
		}
		if cfg.Embedder.OpenAI.Model == "" {
			cfg.Embedder.OpenAI.Model = "text-embedding-3-small"
		}
		if cfg.Embedder.OpenAI.TimeoutSecs == 0 {
			cfg.Embedder.OpenAI.TimeoutSecs = 30
		}
		if cfg.Embedder.OpenAI.BatchSize == 0 {
			cfg.Embedder.OpenAI.BatchSize = 32
		}
	}
}
