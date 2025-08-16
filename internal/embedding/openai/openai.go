package openai

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"time"
)

// Client is an OpenAI-compatible embeddings client implementing the Embedder interface.
type Client struct {
	baseURL    string
	apiKey     string
	model      string
	timeout    time.Duration
	dimension  int
	client     *http.Client
	maxRetries int
}

// Config configures the OpenAI-compatible embeddings client.
type Config struct {
	BaseURL   string
	APIKeyEnv string
	Model     string
	Timeout   time.Duration
}

// NewClient creates a new embeddings client using the provided configuration.
func NewClient(cfg Config) (*Client, error) {
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
	return &Client{
		baseURL:    cfg.BaseURL,
		apiKey:     key,
		model:      cfg.Model,
		timeout:    t,
		client:     &http.Client{Timeout: t},
		maxRetries: 5,
	}, nil
}

// Name returns the identifier of this embedder implementation.
func (c *Client) Name() string { return "openai" }

// Prepare is not required for remote embedding. We will lazily set dimension on first embed.
func (c *Client) Prepare(corpus []string) error { return nil }

// Dimension returns the dimensionality of the produced embedding vectors.
func (c *Client) Dimension() int { return c.dimension }

// Embed returns an embedding vector for the given text.
func (c *Client) Embed(text string) ([]float64, error) {
	type reqBody struct {
		Input  string `json:"input,omitempty"`
		Prompt string `json:"prompt,omitempty"`
		Model  string `json:"model"`
	}
	url := fmt.Sprintf("%s/embeddings", c.baseURL)
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		body := reqBody{Input: text, Prompt: text, Model: c.model}
		data, _ := json.Marshal(body)
		req, _ := http.NewRequest(http.MethodPost, url, bytes.NewReader(data))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+c.apiKey)

		resp, err := c.client.Do(req)
		if err != nil {
			if attempt < c.maxRetries {
				time.Sleep(retryDelay(attempt))
				continue
			}
			return nil, err
		}

		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			// Respect Retry-After if provided
			if ra := resp.Header.Get("Retry-After"); ra != "" {
				if secs, err := strconv.Atoi(ra); err == nil {
					_ = resp.Body.Close()
					time.Sleep(time.Duration(secs) * time.Second)
				} else {
					_ = resp.Body.Close()
					time.Sleep(retryDelay(attempt))
				}
			} else {
				_ = resp.Body.Close()
				time.Sleep(retryDelay(attempt))
			}
			if attempt < c.maxRetries {
				continue
			}
			return nil, fmt.Errorf("openai embeddings failed: %s", resp.Status)
		}

		if resp.StatusCode >= 300 {
			defer resp.Body.Close()
			return nil, fmt.Errorf("openai embeddings failed: %s", resp.Status)
		}

		payload, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			if attempt < c.maxRetries {
				time.Sleep(retryDelay(attempt))
				continue
			}
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
		// If decoding failed, and retries remain, backoff and retry
		if attempt < c.maxRetries {
			time.Sleep(retryDelay(attempt))
			continue
		}
		return nil, errors.New("no embedding returned")
	}
	return nil, errors.New("no embedding returned")
}

func retryDelay(attempt int) time.Duration {
	if attempt < 0 {
		attempt = 0
	}
	base := 200 * time.Millisecond
	// exponential backoff capped at 5s
	d := base << attempt
	if d > 5*time.Second {
		d = 5 * time.Second
	}
	return d
}
