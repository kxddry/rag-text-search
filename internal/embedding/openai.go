package embedding

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
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
		Input string `json:"input"`
		Model string `json:"model"`
	}
	body := reqBody{Input: text, Model: c.model}
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
	var out struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	dec := json.NewDecoder(resp.Body)
	if err := dec.Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Data) == 0 {
		return nil, errors.New("no embedding returned")
	}
	v := out.Data[0].Embedding
	if len(v) == 0 {
		return nil, errors.New("empty embedding")
	}
	if c.dimension == 0 {
		c.dimension = len(v)
	}
	return v, nil
}
