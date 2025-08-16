package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/joho/godotenv"

	"rag/internal/chunker"
	"rag/internal/config"
	"rag/internal/domain"
	"rag/internal/embedding"
	"rag/internal/service"
	"rag/internal/summarizer"
	"rag/internal/tui"
	"rag/internal/vectorstore"
)

func main() {
	// Load environment variables from .env if present; ignore errors
	_ = godotenv.Load()

	cfgPath := flag.String("config", "", "Path to config YAML (optional; otherwise uses persisted default)")
	flag.Parse()
	inputs := flag.Args()
	if len(inputs) == 0 {
		fmt.Println("Usage: rag [--config=config.yaml] file1.txt [file2.txt ...]")
		os.Exit(1)
	}

	var cfg *config.AppConfig
	var err error
	if *cfgPath == "" {
		cfg, _, err = config.LoadDefault()
	} else {
		cfg, err = config.Load(*cfgPath)
	}
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}

	// Assemble components via interfaces
	var emb domain.Embedder
	switch cfg.Embedder.Type {
	case "tfidf", "":
		emb = embedding.NewTFIDFEmbedder()
	case "openai":
		if cfg.Embedder.OpenAI == nil {
			log.Fatalf("openai embedder config missing")
		}
		client, err := embedding.NewOpenAIClient(embedding.OpenAIConfig{
			BaseURL:   cfg.Embedder.OpenAI.BaseURL,
			APIKeyEnv: cfg.Embedder.OpenAI.APIKeyEnv,
			Model:     cfg.Embedder.OpenAI.Model,
			Timeout:   time.Duration(cfg.Embedder.OpenAI.TimeoutSecs) * time.Second,
		})
		if err != nil {
			log.Fatalf("openai embedder init failed: %v", err)
		}
		emb = client
	default:
		log.Fatalf("unknown embedder: %s", cfg.Embedder.Type)
	}

	var ch domain.Chunker
	switch cfg.Chunker.Type {
	case "sentence", "":
		ch = chunker.NewSentenceChunker(cfg.Chunker.SentencesPerChunk, cfg.Chunker.OverlapSentences)
	default:
		log.Fatalf("unknown chunker: %s", cfg.Chunker.Type)
	}

	var st domain.VectorStore
	switch cfg.VectorStore.Type {
	case "memory", "":
		st = vectorstore.NewMemoryStore()
	case "qdrant":
		if cfg.VectorStore.Qdrant == nil {
			log.Fatalf("qdrant config missing")
		}
		qcfg := vectorstore.QdrantConfig{
			URL:        cfg.VectorStore.Qdrant.URL,
			APIKey:     cfg.VectorStore.Qdrant.APIKey,
			Collection: cfg.VectorStore.Qdrant.Collection,
		}
		st = vectorstore.NewQdrantStore(qcfg)
	default:
		log.Fatalf("unknown vector store: %s", cfg.VectorStore.Type)
	}

	var sum domain.Summarizer
	switch cfg.Summarizer.Type {
	case "frequency", "":
		sum = summarizer.NewFrequencySummarizer()
	default:
		log.Fatalf("unknown summarizer: %s", cfg.Summarizer.Type)
	}

	svc := service.NewRAGService(ch, emb, st, sum, cfg.Summarizer.MaxSentences)
	summary, err := svc.IngestDocuments(inputs)
	if err != nil {
		log.Fatalf("ingest failed: %v", err)
	}

	m := tui.New(svc, summary)
	if err := tea.NewProgram(m).Start(); err != nil {
		log.Fatal(err)
	}
}
