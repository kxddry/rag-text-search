package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"rag-text-search/internal/domain"
)

type RAGPort interface {
	IngestDocuments(paths []string) (string, error)
	Query(query string, topK int) ([]domain.SearchResult, error)
}

type Model struct {
	service  RAGPort
	input    textinput.Model
	viewport viewport.Model
	results  []domain.SearchResult
	summary  string
	status   string
	cursor   int
	ready    bool
}

func New(service RAGPort, summary string) Model {
	ti := textinput.New()
	ti.Prompt = "> "
	ti.Placeholder = "Type query and press Enter"
	ti.Focus()
	ti.CharLimit = 0
	vp := viewport.New(0, 0)
	return Model{service: service, input: ti, viewport: vp, summary: summary, status: "Loaded. Type to search."}
}

func (m Model) Init() tea.Cmd { return textinput.Blink }

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.ready = true
		m.viewport.Width = msg.Width
		m.viewport.Height = msg.Height - 3
		m.viewport.SetContent(m.renderResults())
		return m, nil
	case tea.KeyMsg:
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
				}
				m.viewport.SetContent(m.renderResults())
				return m, nil
			}
		case "n":
			if len(m.results) > 0 {
				m.cursor = (m.cursor + 1) % len(m.results)
				m.viewport.SetContent(m.renderResults())
			}
		case "N":
			if len(m.results) > 0 {
				m.cursor = (m.cursor - 1 + len(m.results)) % len(m.results)
				m.viewport.SetContent(m.renderResults())
			}
		}
	}
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	return m, cmd
}

func (m Model) View() string {
	if !m.ready {
		return "Loading..."
	}
	header := lipgloss.NewStyle().Bold(true).Render("RAG Text Search")
	summary := lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Render(m.summary)
	input := m.input.View()
	status := lipgloss.NewStyle().Foreground(lipgloss.Color("10")).Render(m.status)
	return header + "\n" + summary + "\n" + m.viewport.View() + "\n" + input + "\n" + status
}

func (m Model) renderResults() string {
	if len(m.results) == 0 {
		return "No results yet."
	}
	var b strings.Builder
	for i, r := range m.results {
		prefix := "  "
		if i == m.cursor {
			prefix = "> "
		}
		b.WriteString(fmt.Sprintf("%s[%0.3f] %s\n", prefix, r.Score, r.Chunk.Text))
	}
	return b.String()
}
