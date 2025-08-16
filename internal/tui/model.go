package tui

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"rag/internal/domain"
)

// RAGPort is the TUI-facing subset of the RAG service.
type RAGPort interface {
	IngestDocuments(paths []string) (string, error)
	Query(query string, topK int) ([]domain.SearchResult, error)
}

// Model is the Bubble Tea model for the TUI application.
type Model struct {
	service   RAGPort
	input     textinput.Model
	viewport  viewport.Model
	results   []domain.SearchResult
	summary   string
	status    string
	cursor    int
	ready     bool
	lastQuery string
}

// New creates a new TUI model instance.
func New(service RAGPort, summary string) Model {
	ti := textinput.New()
	ti.Prompt = "> "
	ti.Placeholder = "Type query and press Enter"
	ti.Focus()
	ti.CharLimit = 0
	vp := viewport.New(0, 0)
	return Model{service: service, input: ti, viewport: vp, summary: summary, status: "Loaded. Type to search."}
}

// Init initializes the model (text input cursor blink).
func (m Model) Init() tea.Cmd { return textinput.Blink }

// Update handles key and window events and updates the view state.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.ready = true
		// account for frames around result and query boxes
		_, rh := resultBoxStyle.GetFrameSize()
		_, qh := queryBoxStyle.GetFrameSize()
		totalHeaderLines := 2                                    // header + summary
		totalFooterLines := 1                                    // status
		reserved := totalHeaderLines + totalFooterLines + qh + 1 // 1 spacer
		vh := msg.Height - reserved
		if vh < 3 {
			vh = 3
		}
		m.viewport.Width = max(20, msg.Width)
		m.viewport.Height = max(3, vh-rh)
		m.viewport.SetContent(m.renderCurrentResult())
		return m, nil
	case tea.KeyMsg:
		// Global quits
		if msg.Type == tea.KeyCtrlC || msg.Type == tea.KeyCtrlD {
			return m, tea.Quit
		}
		switch msg.String() {
		case "enter":
			q := strings.TrimSpace(m.input.Value())
			if q != "" {
				res, err := m.service.Query(q, 10)
				if err != nil {
					m.status = "Error: " + err.Error()
					m.results = nil
				} else {
					m.status = fmt.Sprintf("Results for %q", q)
					m.results = res
					m.cursor = 0
					m.lastQuery = q
				}
				m.viewport.SetContent(m.renderCurrentResult())
				return m, nil
			}
		case "down":
			if len(m.results) > 0 {
				m.cursor = (m.cursor + 1) % len(m.results)
				m.viewport.SetContent(m.renderCurrentResult())
				return m, nil
			}
		case "up":
			if len(m.results) > 0 {
				m.cursor = (m.cursor - 1 + len(m.results)) % len(m.results)
				m.viewport.SetContent(m.renderCurrentResult())
				return m, nil
			}
		}
	}
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	return m, cmd
}

// View renders the TUI layout and current result.
func (m Model) View() string {
	if !m.ready {
		return "Loading..."
	}
	header := lipgloss.NewStyle().Bold(true).Render("RAG Text Search")
	summary := lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Render(m.summary)
	input := queryBoxStyle.Render(m.input.View())
	status := lipgloss.NewStyle().Foreground(lipgloss.Color("10")).Render(m.status)
	results := resultBoxStyle.Render(m.viewport.View())
	return header + "\n" + summary + "\n" + results + "\n" + input + "\n" + status
}

func (m Model) renderCurrentResult() string {
	if len(m.results) == 0 {
		return "No results yet."
	}
	r := m.results[m.cursor]
	title := fmt.Sprintf("Result %d/%d  score=%.3f", m.cursor+1, len(m.results), r.Score)
	body := highlightBestSentence(r.Chunk.Text, m.lastQuery)
	return title + "\n\n" + body
}

var (
	resultBoxStyle = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).Padding(0, 1)
	queryBoxStyle  = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).Padding(0, 1)
	highlightStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("11")).Bold(true)
	unicodeWordRe  = regexp.MustCompile(`\p{L}+(?:['’]\p{L}+)*`)
	sentenceRe     = regexp.MustCompile(`(?m)(?U)([^.!?]+[.!?])`)
)

func highlightBestSentence(text, query string) string {
	if strings.TrimSpace(text) == "" {
		return text
	}
	sentences := sentenceRe.FindAllString(text, -1)
	if len(sentences) == 0 {
		sentences = []string{strings.TrimSpace(text)}
	}
	qTokens := toTokenSet(query)
	if len(qTokens) == 0 {
		return strings.Join(sentences, " ")
	}
	bestIdx := 0
	bestScore := -1
	for i, s := range sentences {
		score := tokenOverlapScore(qTokens, s)
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}
	for i := range sentences {
		sent := strings.TrimSpace(sentences[i])
		if i == bestIdx {
			sentences[i] = highlightStyle.Render(sent)
		} else {
			sentences[i] = sent
		}
	}
	return strings.Join(sentences, " ")
}

func toTokenSet(s string) map[string]struct{} {
	tokens := unicodeWordRe.FindAllString(strings.ToLower(s), -1)
	m := make(map[string]struct{}, len(tokens))
	for _, t := range tokens {
		m[t] = struct{}{}
	}
	return m
}

func tokenOverlapScore(queryTokens map[string]struct{}, sentence string) int {
	score := 0
	tokens := unicodeWordRe.FindAllString(strings.ToLower(sentence), -1)
	seen := make(map[string]struct{}, len(tokens))
	for _, t := range tokens {
		if _, ok := seen[t]; ok {
			continue
		}
		seen[t] = struct{}{}
		if _, ok := queryTokens[t]; ok {
			score++
		}
	}
	return score
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
