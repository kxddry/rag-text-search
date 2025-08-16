package embedding

// Embedder converts free text into a numeric vector representation.
// Implementations may require a preparation phase over the corpus.
type Embedder interface {
	Name() string
	Prepare(corpus []string) error
	Dimension() int
	Embed(text string) ([]float64, error)
}
