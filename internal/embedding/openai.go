package embedding

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

type OpenAIClient struct {
	baseURL   string
	apiKey    string
	model     string
	timeout   time.Duration
	dimension int
	client    *http.Client
}

type OpenAIConfig struct {
	BaseURL   string
	APIKeyEnv string
	Model     string
	Timeout   time.Duration
}

func NewOpenAIClient(cfg OpenAIConfig) (*OpenAIClient, error) {
	key := os.Getenv(cfg.APIKeyEnv)
	if key == "" {
		return nil, fmt.Errorf("missing API key in env %s", cfg.APIKeyEnv)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	if cfg.Model == "" {
		cfg.Model = "text-embedding-3-small"
	}
	t := cfg.Timeout
	if t == 0 {
		t = 30 * time.Second
	}
	return &OpenAIClient{
		baseURL: cfg.BaseURL,
		apiKey:  key,
		model:   cfg.Model,
		timeout: t,
		client:  &http.Client{Timeout: t},
	}, nil
}

func (c *OpenAIClient) Name() string { return "openai" }

// Prepare is not required for remote embedding. We will lazily set dimension on first embed.
func (c *OpenAIClient) Prepare(corpus []string) error { return nil }

func (c *OpenAIClient) Dimension() int { return c.dimension }

func (c *OpenAIClient) Embed(text string) ([]float64, error) {
	type reqBody struct {
		Input  string `json:"input,omitempty"`
		Prompt string `json:"prompt,omitempty"`
		Model  string `json:"model"`
	}
	body := reqBody{Input: text, Prompt: text, Model: c.model}
	data, _ := json.Marshal(body)
	url := fmt.Sprintf("%s/embeddings", c.baseURL)
	req, _ := http.NewRequest(http.MethodPost, url, bytes.NewReader(data))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("openai embeddings failed: %s", resp.Status)
	}
	// Read payload and support both OpenAI-compatible and Ollama-native shapes
	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	// Try OpenAI-compatible response first
	var openaiOut struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(payload, &openaiOut); err == nil {
		if len(openaiOut.Data) > 0 && len(openaiOut.Data[0].Embedding) > 0 {
			v := openaiOut.Data[0].Embedding
			if c.dimension == 0 {
				c.dimension = len(v)
			}
			return v, nil
		}
	}
	// Fallback to Ollama-native shape: { "embedding": [...] }
	var ollamaOut struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.Unmarshal(payload, &ollamaOut); err == nil {
		if len(ollamaOut.Embedding) > 0 {
			v := ollamaOut.Embedding
			if c.dimension == 0 {
				c.dimension = len(v)
			}
			return v, nil
		}
	}
	return nil, errors.New("no embedding returned")
}
