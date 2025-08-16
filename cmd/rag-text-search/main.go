package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	tea "github.com/charmbracelet/bubbletea"

	"rag-text-search/internal/chunker"
	"rag-text-search/internal/config"
	"rag-text-search/internal/domain"
	"rag-text-search/internal/embedding"
	"rag-text-search/internal/service"
	"rag-text-search/internal/summarizer"
	"rag-text-search/internal/tui"
	"rag-text-search/internal/vectorstore"
)

func main() {
	cfgPath := flag.String("config", "config.yaml", "Path to config YAML")
	flag.Parse()
	inputs := flag.Args()
	if len(inputs) == 0 {
		fmt.Println("Usage: rag-text-search [--config=config.yaml] file1.txt [file2.txt ...]")
		os.Exit(1)
	}

	cfg, err := config.Load(*cfgPath)
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}

	// Assemble components via interfaces
	var emb domain.Embedder
	switch cfg.Embedder.Type {
	case "tfidf", "":
		emb = embedding.NewTFIDFEmbedder()
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
