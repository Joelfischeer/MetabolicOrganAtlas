# Metabolic Reference Atlas

An interactive, literature-based atlas of metabolic crosstalk between **14 organs** (91 organ–organ pairs), with FDG-PET partial-correlation networks as an imaging-based reference.

**Live page:** https://joelfischeer.github.io/MetabolicOrganAtlas/

The organs are adrenal glands, bone marrow, brain, colon, heart, kidney, liver, lung, muscle, pancreas, small intestine, spleen, thyroid and white adipose tissue.

## What the atlas shows

- **Literature network.** PubMed papers from the last 20 years (search run on 17 Sep 2026) are kept for a pair of organs only if both organs and a crosstalk keyword appear in the same sentence of the title or abstract. Each connection links two organs and is backed by at least one paper.
- **Disease conditions.** An LLM labels every paper with one of seven conditions: No condition, Healthy, Obesity/Metabolic/Diabetes, Cardiovascular, Cancers, Infectious or Other diseases. The network can be viewed for all papers or for a single condition.
- **Connection types and descriptions.** An LLM assigns each connection one to three types (for example hormonal glucose control or nerve connection) and writes a short, cited summary.
- **Key players.** Hormones, metabolites and proteins mentioned in the papers of a connection are extracted and counted.
- **FDG-PET reference networks.** Partial-correlation networks for a healthy cohort (n = 112), an overweight cohort (n = 85) and a lung cancer cohort (n = 241), following [Geist et al. (2025)](https://doi.org/10.1016/j.medj.2025.100881). Blue edges are positive correlations and red edges are negative ones.

## How to use it

1. **Open the page** in any modern browser. No installation or server is needed.
2. **Click an organ or a connection** to open its details in the side panel: a metabolic overview for an organ, and a cited summary, connection types, papers and key players for a connection. Click empty space to reset.
3. **Choose a condition** with the buttons under *Conditions*. *All papers* combines every condition.
4. **Search** for an organ, hormone or molecule in the search box to highlight the matching connections.
5. **Filter key players** with the *Key Players* menu. It hides key players mentioned in fewer papers than the minimum you set (default 2).
6. **Compare with imaging data** with the *Show reference network* buttons (Healthy, Overweight or Lung cancer cohort). Use *Clear reference* to remove it.
7. **Show only strong connections** with *≥ OO threshold*. It keeps edges at or above the Otsuka–Ochiai elbow threshold of the current view.
8. **Change the layout** with *Circle*, *Grid* or *Fit*.
9. **Read how it was built** under *How it works*, and see the search, network and enrichment results under *Result Statistics*.

## Repository contents

| Path | Contents |
|---|---|
| `reference_network_only_metabolic/` | Pipeline configuration, cached search results and the generated `metabolic_literature_network.html` |
| `Literature_Search/` | PubMed search, filters and LLM steps |
| `Visualisation/` | Dashboard building blocks and info-panel texts |
| `run_metabolic_lit_search.py` | Main entry point of the pipeline |
| `docs/index.html` | Copy of the dashboard served as the GitHub Page |

## Rebuilding the dashboard

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
uv run python run_metabolic_lit_search.py --viz-only
```

`--viz-only` regenerates the HTML from the cached results without any search or LLM calls. The full pipeline needs PubMed access and an LLM backend (the Claude API or Ollama). After rebuilding, copy the HTML to `docs/index.html` to update the GitHub Page.

## Reference

Geist, B. K. et al. The metabolic organ connectome: A novel approach to measure allostatic load during health-to-disease transition. *Med* 6, 100881 (2025). https://doi.org/10.1016/j.medj.2025.100881
