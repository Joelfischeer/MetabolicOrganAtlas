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

## Used organ aliases for the Pubmed search

Adrenal Glands: adrenal gland, adrenal cortex, adrenal medulla, adrenocortical \
Bone Marrow: bone marrow, hematopoietic, haematopoietic, hematopoiesis, bone marrow niche \
Brain: brain, cerebral, hypothalamus, hypothalamic, pituitary, pituitary gland, arcuate nucleus, paraventricular nucleus, nucleus tractus solitarius, area postrema, hippocampus, hippocampal, amygdala, brainstem, brain stem, cerebellum, cerebellar, thalamus, thalamic, basal ganglia, prefrontal cortex, cerebral cortex, medulla oblongata, pons, forebrain, midbrain, hindbrain, central nervous system, CNS, neuronal, neuroendocrine \
Colon: colon, large intestine, colorectal, colonic \
Heart: heart, cardiac, myocardial, cardiomyocyte, myocardium, cardiac muscle \
Kidney: kidney, renal, nephron, glomerular, kidneys, tubular \
Liver: liver, hepatic, hepatocyte, hepatocellular\
Lung: lung, pulmonary, alveolar, bronchial\
Muscle: skeletal muscle, myocyte\
Pancreas: Pancreas; pancreas, pancreatic\
Small Intestine: small intestine, duodenum, jejunum, ileum, intestinal, gut, enterocyte, Peyer's patches\
Spleen: spleen, splenic, splenomegaly, lymphoid organ, immune organ\
Thyroid: thyroid, thyroid gland\
WAT: adipose tissue, white adipose tissue, white fat tissue, adipocyte, adipogenesis, visceral fat, subcutaneous fat, WAT\

Special organ alias cases: "gut" (including gut microbiota) counts as small intestine only and not towards colon to not make the assignments ambiguous.  The word “muscle” is not counted toward the organ “muscle” as it could also appear in “smooth muscle” or “cardiac muscle”

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

## Literature Retrieval and Analysis Pipeline

### Overview

The literature-derived reference network comprises **14 organs and 91 organ pairs**. The workflow consists of seven steps, with Steps 2–6 using **Claude Haiku 5.5** (`claude-haiku-5-5`, Anthropic API) for language-model-assisted tasks.

The exact prompts used for all LLM tasks are documented in the **LLM Prompts** tab of the interactive dashboard.

### Workflow

#### Step 1 — Literature Retrieval and Local Filtering

**PubMed retrieval**

Each organ is represented by its MeSH term and a curated list of text aliases searched in titles and abstracts. A separate query is constructed for each organ pair:

```text
(MeSH A OR aliases A)
AND (MeSH B OR aliases B)
AND metabolic term
AND crosstalk term
```

The query includes the following term lists:

| Category | Number of terms |
|---|---:|
| Metabolic terms | 22 |
| Crosstalk terms | 20 |

**Metabolic terms:**
`metabolism`, `metabolic`, `catabolism`, `anabolism`, `biosynthesis`, `metabolite`, `flux`, `substrate`, `nutrient`, `fuel`, `glucose`, `glucose uptake`, `glycolysis`, `gluconeogenesis`, `energy`, `bioenergetics`, `mitochondria`, `mitochondrial`, `thermogenesis`, `Krebs cycle`, `electron transport`, `oxidation`.

**Crosstalk terms:**
`network`, `interplay`, `cross-talk`, `crosstalk`, `connection`, `axis`, `inter-organ`, `interorgan`, `cross-organ`, `communication`, `bidirectional`, `feedback loop`, `signaling`, `regulation`, `mediates`, `modulates`, `interaction`, `link`, `coupling`, `cross-regulation`.

**Local filtering**

Retrieved papers undergo two filtering steps:

1. **Same-sentence co-location:** A paper is retained only if organ A, organ B, and a crosstalk term occur together in at least one sentence. Sentences are split at `.`, `!`, `?`, and `;`. Matching is whole-word and negation-aware (e.g., `non-renal` and `not renal` are not considered kidney mentions).

2. **Multi-organ disambiguation:** Sentences mentioning additional organs are evaluated to distinguish direct A–B relationships from separate relationships involving other organs. Organ mentions connected exclusively by list connectives (`,`, `and`, `or`, `/`, `&`, `as well as`) are treated as one enumeration group. A pair is rejected if:
   - Both organs belong to the same enumeration group.
   - Both organs are parallel locations connected by a repeated location preposition (e.g., "in the brain and in the lung").
   - Another organ outside their respective groups is mentioned before the later of the two target organs.

   Organs mentioned only after both target organs are ignored.

#### Step 2 — Disease-Condition Classification

Each paper is assigned one primary condition describing the studied subjects. Seven mutually exclusive labels are used:

| Label | Definition |
|---|---|
| No condition | No health state defined (e.g., general physiology, reviews, methods, in-vitro work, or not stated) |
| Healthy | Healthy humans or normal animals as study subjects, not merely controls |
| Obesity / metabolic syndrome / diabetes | Studies involving these conditions |
| Cardiovascular disease | Studies involving cardiovascular conditions |
| Cancer | Studies involving malignancies |
| Infectious diseases | Studies involving infectious conditions |
| Other diseases | All other pathological conditions |

The model receives the title, abstract, label definitions, and one worked example per label. It must return both the assigned label and the evidence sentence supporting its decision.

**Evidence validation:** To prevent unsupported or hallucinated classifications, every word in the evidence sentence (minimum four words) must occur in the title or abstract in the same order, ignoring case, punctuation, and HTML entities. Empty, invalid, refused, or non-matching responses are discarded.

#### Step 3 — Key-Player Extraction

For each organ pair and disease-condition view, titles, abstracts, and MeSH descriptors are scanned against three curated vocabularies. The number of papers containing each term is counted, with each term or synonym counted at most once per paper.

| Category | Vocabulary size | Source |
|---|---:|---|
| Hormones | 171 | Manually curated from published lists [23, 24] |
| Metabolites | 254 | Stanford Pathways of Human Metabolism map [25] |
| Proteins | 45,138 | Human Protein Atlas (HPA), UniProt, and neXtProt [26, 27] |

The protein vocabulary comprises four HPA gene tables covering intracellular, membrane, secreted, and secreted-and-membrane protein evidence.

**Vocabulary curation and disambiguation**

- Hormones appearing in the metabolite or protein vocabulary are retained exclusively as hormones.
- Terms and synonyms matching one of the 10,000 most frequent English words (Google Trillion Word Corpus, first20hours list [28]) are removed unless biologically important (e.g., ACC, ACE, INS, glucose, oxygen, cholesterol, AMP, GDP, dUMP).
- Synonym groups representing the same biological entity are established with ChatGPT assistance.
- HPA entries include gene names and hyphenated variants (e.g., `interleukin 2` / `interleukin-2`, `IL2` / `IL-2`).
- `Glucagon`, `insulin`, and `amp` are excluded when followed by `-like` or `-activated`.
- Short and official gene names are unified (e.g., `TGR5` / `GPBAR1`, `GLUT1–5` / `SLC2A1–5`).
- Ambiguous aliases such as `pigs`, `clock`, and `pgc` are removed.

#### Step 4 — Connection-Type Classification

For each organ pair and disease-condition view, the model receives the first 100 papers (titles and abstracts truncated to 400 characters) and a predefined list of 11 connection types.

Classification is repeated in **five independent runs**. Up to three connection types can be selected per run, provided each is clearly and independently supported by the literature.

A connection type is retained if selected in at least **3 of 5 runs**. If no type reaches this threshold, the most frequently selected type is used.

**Connection types**

| Connection type | Definition |
|---|---|
| Direct glucose transfer | One organ releases glucose that the other directly takes up as its primary fuel. |
| Hormonal glucose control | A hormone (insulin, glucagon, GLP-1, GIP) from one organ directly regulates glucose handling in the other. |
| Other hormonal influence | A hormone from one organ influences the metabolism or function of the other without directly affecting glucose handling. |
| Nerve connection | Interaction through a physical nerve connection (e.g., the vagus nerve in the gut–brain axis). |
| Other physical connection | Interaction through a non-neural physical connection (e.g., mechanical influence). |
| Ketone body exchange | Ketone bodies produced through hepatic ketogenesis are exported and oxidised as fuel by the other organ. |
| Mineral/electrolyte-coupled metabolic regulation | One organ regulates a mineral or electrolyte (e.g., calcium, phosphate, iron) whose levels feed back on metabolism in the other. |
| Adipokine/myokine signaling | An adipokine or myokine (e.g., leptin, adiponectin, irisin, FGF21) acts on metabolic pathways in the other organ. |
| Inflammatory/cytokine crosstalk | Cytokines released by one organ drive inflammation that alters metabolism or function in the other. |
| Metabolite crosstalk | Metabolites produced by one organ directly influence the metabolism of the other. |
| Circadian/clock synchronization | Molecular clocks in the two organs entrain each other, with misalignment disrupting metabolic rhythms. |

#### Step 5 — Organ-Pair Descriptions

For each organ pair and disease-condition view:

1. A random pool of up to 40 papers is selected.
2. The model selects the 10 most relevant papers from this pool.
3. Based exclusively on these 10 papers, the model generates a scientific summary of 5–7 sentences.
4. Each sentence must be supported by citations, with every cited passage verified against the corresponding paper.
5. If citation validation fails, up to three additional generation attempts are performed.

Only summaries passing citation validation are retained.

#### Step 6 — Organ Descriptions

For each of the 14 organs and their aliases, PubMed is searched for the **50 most recent papers** containing the 22 metabolic terms in their titles or abstracts.

Descriptions are generated using the same procedure as in Step 5, with a specific focus on the metabolic substrates consumed and produced by each organ.

#### Step 7 — Interactive Network Visualization

The organ network is presented in an interactive, single-file HTML dashboard built with **Cytoscape.js 3.29.2**.

**Dashboard features:**

- Separate network views for each disease-condition and an additional **All papers** view.
- Interactive organ-pair edges displaying:
  - Connection types
  - Key players (hormones, metabolites, and proteins)
  - Citation-supported organ-pair descriptions
  - Underlying literature references
- Overlay of partial-correlation networks derived from FDG-PET cohorts:
  - Healthy (BMI ≤ 25)
  - Overweight (BMI > 25)
  - Lung cancer

This allows literature-derived organ-crosstalk relationships to be explored alongside metabolic connectivity patterns observed in the FDG-PET cohorts.

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
