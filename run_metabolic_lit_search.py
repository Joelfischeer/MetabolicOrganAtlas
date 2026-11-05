"""
run_metabolic_lit_search.py
===========================
Metabolic-focused PubMed search for all organ-organ pairs.

Query per pair (single strategy, no cascade):
    (MeSH_A OR aliases_A) AND (MeSH_B OR aliases_B)
    AND <METABOLIC_FILTER>

where METABOLIC_FILTER is the curated metabolic keyword clause below.

Results saved to:  metabolic_data/metabolic_literature_results.json
Visualization at:  metabolic_data/metabolic_literature_network.html

Run:
    python run_metabolic_lit_search.py              # first run or resume
    python run_metabolic_lit_search.py --reset      # wipe cache and restart
    python run_metabolic_lit_search.py --force-empty  # re-search 0-paper edges
    python run_metabolic_lit_search.py --viz-only   # skip search, rebuild viz

LLM condition filter (this folder replaces the keyword condition filter of the
original pipeline -- everything else is unchanged):
    python run_metabolic_lit_search.py --classify-conditions   # label every paper (resumable)
    python run_metabolic_lit_search.py --condition healthy     # build ONE condition view (healthy | obesity |
                                                               #   cardiovascular | cancers | infectious | other | nocondition)
    python run_metabolic_lit_search.py --condition every       # build all seven condition views in turn
    python generate_condition_filter_statistics.py             # statistics of the filtering
"""

import sys
import json
import time
import re
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import requests
from collections import Counter

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# Windows console/redirected-output encoding defaults to cp1252, which can't
# encode the arrows/ellipses used in progress prints below — force UTF-8 so
# the run doesn't crash mid-way through (e.g. when stdout is piped to a file).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from Literature_Search.pubmed_search import (
    ORGAN_ALIASES,
    ORGAN_MESH,
    fetch_abstracts,
    extract_key_players,
    _merge_synonyms,
    HORMONE_SYNONYMS,
    METABOLITE_SYNONYMS,
    PROTEIN_SYNONYMS,
)
from Literature_Search.llm_condition_filter import (
    LLMConditionFilter, load_condition_labels, classify_condition_labels,
    CONDITION_TO_LABEL, LABELS as CONDITION_LABEL_NAMES, STATUSES as CONDITION_STATUSES,
    CONDITIONS, CONDITION_SLUGS, CONDITION_NAME, CONDITION_COLOR,
)
# Texts/layout of the dashboard's "how it works" info panel: edit Visualisation/info_panel_texts.html
from Visualisation.info_texts import render_section as _info_text

# ── Load configuration from the output folder ─────────────────────────────────
# Edit:  reference_network_only_metabolic/config.py
OUTPUT_DIR = HERE / "reference_network_only_metabolic"
sys.path.insert(0, str(OUTPUT_DIR))
import config as _cfg

MAX_PAPERS         = _cfg.MAX_PAPERS
YEARS_BACK         = _cfg.YEARS_BACK
DELAY              = _cfg.DELAY
METABOLIC_KEYWORDS = _cfg.METABOLIC_KEYWORDS
CROSSTALK_KEYWORDS = _cfg.CROSSTALK_KEYWORDS
CONNECTION_TYPES   = _cfg.CONNECTION_TYPES
LLM_MODEL          = _cfg.LLM_MODEL
DESCRIPTION_LLM_MODEL = getattr(_cfg, "DESCRIPTION_LLM_MODEL", _cfg.LLM_MODEL)
CONNECTION_TYPE_LLM_MODEL = getattr(_cfg, "CONNECTION_TYPE_LLM_MODEL", _cfg.LLM_MODEL)
LLM_MAX_PAPERS     = _cfg.LLM_MAX_PAPERS
VIZ_TITLE          = _cfg.VIZ_TITLE

# Condition filter: a local LLM labels every paper (title + abstract) with one of
# seven conditions and must quote the sentence that led to its decision; a regex
# test then checks that the sentence really occurs in the paper. Every condition
# view (one per label, all treated alike) consists of the papers carrying that label -- see
# Literature_Search/llm_condition_filter.py and config.py. The old keyword lists
# survive only as LEGACY_CONDITION_KEYWORDS, for the filtering statistics.
CONDITION_LLM_MODEL = _cfg.CONDITION_LLM_MODEL
LEGACY_CONDITION_KEYWORDS = {
    "healthy": _cfg.LEGACY_CONDITION_KEYWORDS_HEALTHY,
    "obese":   _cfg.LEGACY_CONDITION_KEYWORDS_OBESE,
}

OUTPUT_JSON         = OUTPUT_DIR / "metabolic_literature_results.json"
OUTPUT_HTML         = OUTPUT_DIR / "metabolic_literature_network.html"
OUTPUT_LLM_TYPES    = OUTPUT_DIR / "metabolic_connection_types.json"
# Search scope: every pair across the full 14-organ set (91 pairs). The
# healthy cohort CSV is no longer a search-time filter -- it's read
# separately in build_viz() to flag which pairs/organs belong to the
# "Healthy" display toggle in the dashboard.
EDGE_FILTER_CSV     = OUTPUT_DIR / "all_organ_connections.csv"
HEALTHY_FILTER_CSV  = OUTPUT_DIR / "healthy_cohort_connections.csv"
OBESE_FILTER_CSV    = OUTPUT_DIR / "obese_cohort_connections.csv"
LUNG_FILTER_CSV     = OUTPUT_DIR / "lung_cancer_cohort_connections.csv"   # compute_lung_cancer_PC_network.py
LLM_DESCRIPTIONS   = OUTPUT_DIR / "metabolic_llm_descriptions.json"
CONDITION_LABELS    = OUTPUT_DIR / "metabolic_condition_labels.jsonl"   # LLM condition label per PMID
ORGAN_DESCRIPTIONS  = HERE / "metabolic_data" / "organ_descriptions.json"

# Per-condition cohort CSV + output files -- kept fully separate from the
# "all" outputs above (OUTPUT_JSON/OUTPUT_LLM_TYPES/LLM_DESCRIPTIONS), which
# stay untouched by --condition runs.
CONDITION_COHORT_CSV = {
    "healthy": HEALTHY_FILTER_CSV,
    "obese":   OBESE_FILTER_CSV,
}


def condition_output_paths(condition: str) -> dict[str, Path]:
    return {
        "results":     OUTPUT_DIR / f"metabolic_literature_results_{condition}.json",
        "llm_types":   OUTPUT_DIR / f"metabolic_connection_types_{condition}.json",
        "llm_descs":   OUTPUT_DIR / f"metabolic_llm_descriptions_{condition}.json",
    }

def load_edge_filter(csv_path: Path) -> list[tuple[str, str]]:
    """
    Parse a 0/1 adjacency matrix CSV.  Returns sorted canonical pairs where
    the cell value is 1.  Both upper and lower triangle are accepted; diagonal
    and empty cells are ignored.
    """
    import csv as _csv
    pairs: set[tuple[str, str]] = set()
    with open(csv_path, encoding="utf-8") as f:
        reader = _csv.reader(f)
        header = next(reader)
        col_organs = header[1:]          # first column is the row-label
        for row in reader:
            if not row:
                continue
            row_organ = row[0].strip()
            for col_idx, val in enumerate(row[1:], start=0):
                val = val.strip()
                if val == "1" and col_idx < len(col_organs):
                    col_organ = col_organs[col_idx].strip()
                    if row_organ and col_organ and row_organ != col_organ:
                        pairs.add((min(row_organ, col_organ),
                                   max(row_organ, col_organ)))
    return sorted(pairs)


def _build_metabolic_filter(keywords: list[str]) -> str:
    """Convert the keyword list into a PubMed boolean clause."""
    parts = []
    for kw in keywords:
        # Multi-word phrases need quoting; single words do not
        if " " in kw or "-" in kw:
            parts.append(f'"{kw}"[Title/Abstract]')
        else:
            parts.append(f'{kw}[Title/Abstract]')
    return "(" + " OR ".join(parts) + ")"

METABOLIC_FILTER = _build_metabolic_filter(METABOLIC_KEYWORDS)
CROSSTALK_FILTER = _build_metabolic_filter(CROSSTALK_KEYWORDS)

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _ncbi_get(endpoint: str, params: dict, retries: int = 3, method: str = "GET"):
    params.setdefault("tool", "MetabolicReferenceNetwork")
    params.setdefault("email", "research@metabolic-network.org")
    for attempt in range(retries):
        try:
            if method == "POST":
                resp = requests.post(NCBI_BASE + endpoint, data=params, timeout=30)
            else:
                resp = requests.get(NCBI_BASE + endpoint, params=params, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"    [!] HTTP error (attempt {attempt+1}/{retries}): {e} — retrying in {wait}s")
            time.sleep(wait)
    return None


# ── Query builder ─────────────────────────────────────────────────────────────

def _alias_clause(organ: str) -> str:
    """Build (MeSH OR alias1 OR alias2 ...) clause for one organ."""
    mesh    = ORGAN_MESH.get(organ, f'"{organ}"[MeSH Terms]')
    aliases = ORGAN_ALIASES.get(organ, [organ])
    alias_terms = " OR ".join(f'"{a}"[Title/Abstract]' for a in aliases)
    return f"({mesh} OR {alias_terms})"


def build_query(organ1: str, organ2: str) -> str:
    """Single metabolic-focused query for a pair of organs."""
    c1 = _alias_clause(organ1)
    c2 = _alias_clause(organ2)
    return f"{c1} AND {c2} AND {METABOLIC_FILTER} AND {CROSSTALK_FILTER}"


# ── Word-boundary / negation-safe term matching ────────────────────────────
# Plain substring checks (`"renal" in text`) false-positive on words that
# merely contain the term, e.g. "renal" inside "adrenal" — a paper about the
# adrenal gland would otherwise count as kidney evidence. \b...\b anchors the
# match to whole-word boundaries so this can't happen. A leading negative
# lookbehind additionally excludes negated mentions ("non-renal", "nonrenal",
# "non renal", "not renal") from counting as a positive mention of that
# organ/keyword — these explicitly say the opposite.

def _word_boundary_pattern(term: str):
    escaped = re.escape(term.lower())
    return re.compile(r"(?<!non-)(?<!non )(?<!non)(?<!not )\b" + escaped + r"\b")


def _compile_patterns(terms) -> list:
    return [_word_boundary_pattern(t) for t in terms]


def _any_match(patterns: list, text: str) -> bool:
    return any(p.search(text) for p in patterns)


# ── Multi-organ same-sentence disambiguation ────────────────────────────────
# A qualifying sentence (organ1 + organ2 + crosstalk keyword co-present) can
# still be a false positive for the pair (organ1, organ2) specifically when a
# THIRD organ is also named in that sentence, e.g. "the pancreas modulates
# ... the heart and colon" -- Pancreas is the true agent, Heart/Colon are
# just co-listed targets, not a Heart<->Colon relationship. These helpers add
# a purely positional/lexical check (no LLM, no parsing) that generalizes to
# any number of organs mentioned in one sentence:
#
#  1. _list_groups() groups organ mentions that are joined ONLY by
#     enumeration glue (",", "and", "or", "/", "&", "as well as", articles)
#     with nothing else between them -- "heart, colon, and kidney" becomes
#     one group regardless of how many members it has. Two organs in the
#     same group are siblings in a list, not claimed to interact with each
#     other.
#  2. For organs in different groups, a pair is rejected if some OTHER organ
#     (not a fellow list-member of either side) is mentioned before the
#     later of the two -- the classic "X regulates ... A and B" pattern --
#     while organs mentioned only AFTER both (e.g. a trailing contrastive
#     "...unlike the kidney") are not held against the pair.
#  3. Parallel locations: when the glue between two organ mentions repeats a
#     location preposition after the conjunction ("...in the brain and in the
#     lung...") the organs are two places something was measured, not partners
#     of a relationship. Such organs are joined into one list group that is
#     flagged "parallel", and a pair inside a parallel group is rejected --
#     even when only those two organs are named (rules 1-2 never look at
#     two-organ sentences, which would otherwise be accepted). The repeated
#     preposition is required, so "between the brain and lung" is unaffected.
ALL_ORGAN_PATTERNS = {o: _compile_patterns(aliases) for o, aliases in ORGAN_ALIASES.items()}

_LIST_GLUE_RE = re.compile(r"^\s*(,|;|&|/)?\s*(and|or|as well as|&|/)?\s*(the|a|an)?\s*$")
_PARALLEL_LOC_GLUE_RE = re.compile(
    r"^\s*[,;]?\s*(?:and|or|as well as|and also|but also)\s+(?:in|within|at)\s+(?:the|a|an)?\s*$",
    re.IGNORECASE)


def _find_organ_mentions(sent: str, all_organ_patterns: dict) -> list[tuple[int, int, str]]:
    """All (start, end, organ_name) mentions in sent, sorted by position."""
    hits = [(m.start(), m.end(), organ)
            for organ, pats in all_organ_patterns.items()
            for p in pats for m in p.finditer(sent)]
    hits.sort(key=lambda t: t[0])
    return hits


def _list_groups(mentions: list[tuple[int, int, str]],
                 sent: str) -> tuple[dict[str, int], set[int]]:
    """Assign each organ an enumeration-list group id. Consecutive mentions
    (in sentence order) separated only by list glue ("," "and" "/" ...) get
    the same id; anything else (a verb, a clause) starts a new group.

    Also returns the ids of "parallel" groups: those containing at least one
    link whose glue repeats a location preposition ("and in the", "or within")."""
    groups: dict[str, int] = {}
    parallel_groups: set[int] = set()
    group_id = 0
    prev_end = None
    for start, end, organ in mentions:
        if prev_end is not None:
            glue = sent[prev_end:start]
            if _PARALLEL_LOC_GLUE_RE.match(glue):
                parallel_groups.add(group_id)
            elif not _LIST_GLUE_RE.match(glue):
                group_id += 1
        groups.setdefault(organ, group_id)
        prev_end = end
    return groups, parallel_groups


def _passes_multi_organ_check(sent: str, organ1: str, organ2: str,
                               all_organ_patterns: dict) -> bool:
    """False if the sentence's phrasing indicates organ1/organ2 are not
    directly linked to each other (see module comment above)."""
    mentions = _find_organ_mentions(sent, all_organ_patterns)
    groups, parallel_groups = _list_groups(mentions, sent)

    # Parallel locations ("...in the brain and in the lung..."). Checked before
    # the two-organ shortcut below, which would otherwise accept it.
    if groups.get(organ1) == groups.get(organ2) and groups.get(organ1) in parallel_groups:
        return False

    if len(mentions) <= 2:
        return True  # nothing else mentioned -- no ambiguity to resolve

    if groups.get(organ1) == groups.get(organ2):
        return False  # co-listed siblings ("...in the heart and colon")

    first_pos: dict[str, int] = {}
    for start, _end, organ in mentions:
        first_pos.setdefault(organ, start)
    later = max(first_pos[organ1], first_pos[organ2])
    own_groups = {groups[organ1], groups[organ2]}

    for start, _end, organ in mentions:
        if organ in (organ1, organ2) or start >= later:
            continue
        if groups.get(organ) in own_groups:
            continue  # fellow list-member of organ1/organ2, not a third party
        return False  # a genuine third organ precedes the later of the two
    return True


def _filter_same_sentence_crosstalk(papers: list[dict], organ1_patterns: list,
                                    organ2_patterns: list, crosstalk_patterns: list,
                                    organ1_name: str | None = None,
                                    organ2_name: str | None = None,
                                    all_organ_patterns: dict | None = None) -> list[dict]:
    """
    Keep only papers where both organs' aliases AND at least one crosstalk
    keyword (network, axis, interplay, ...) appear together in the same
    sentence — not just somewhere across the same title/abstract. The
    PubMed query already requires both organs and a crosstalk keyword to
    match somewhere in the document (cheap pre-filter, fewer papers
    fetched); this is the strict local check that actually enforces
    three-way co-location.

    When organ1_name/organ2_name/all_organ_patterns are given, a qualifying
    sentence is further required to pass _passes_multi_organ_check() --
    guarding against sentences that mention a third organ and are really
    describing that third organ's relationship to organ1/organ2, not a
    direct organ1<->organ2 link (see module comment above).
    """
    kept = []
    for paper in papers:
        text = (paper.get("title", "") + ". " + paper.get("abstract", "")).lower()
        sentences = re.split(r"(?<=[.!?;])\s+", text)
        for sent in sentences:
            if not (_any_match(organ1_patterns, sent)
                    and _any_match(organ2_patterns, sent)
                    and _any_match(crosstalk_patterns, sent)):
                continue
            if (all_organ_patterns is not None and organ1_name and organ2_name
                    and not _passes_multi_organ_check(sent, organ1_name, organ2_name,
                                                       all_organ_patterns)):
                continue
            kept.append(paper)
            break
    return kept


def _filter_by_condition(papers: list[dict], condition_filter: LLMConditionFilter) -> list[dict]:
    """
    Keep only papers that the LLM condition filter assigned to this case: the
    paper carries the case's label (Healthy / Obesity-metabolic-diabetes) AND
    the evidence sentence the LLM quoted passed the regex test (it really occurs
    in the title/abstract). Papers without a valid label -- not classified yet,
    hallucinated evidence, failed request -- are not used.

    Applied locally to papers that have ALREADY been fetched and passed
    through _filter_same_sentence_crosstalk -- no new PubMed query.
    (This replaces the keyword filter of the original pipeline; the callers
    are unchanged, `condition_filter` simply took the place of the compiled
    keyword patterns.)
    """
    return [paper for paper in papers if condition_filter.accepts(paper)]


def build_condition_results(condition: str, base_results: dict, pairs: list[tuple[str, str]],
                            labels: dict) -> dict:
    """
    Build a condition-specific results dict from the already-searched "all"
    results: for each pair in `pairs`, filter its already-extracted papers
    down to those the LLM condition filter assigned to `condition` (valid
    label + verified evidence sentence), then recompute key players from the
    smaller, condition-filtered paper set.

    Pairs with no cached entry in base_results, or that have zero papers
    after the condition filter, still get an entry (n_papers_found=0) so
    downstream steps (key player counts, LLM classification/description)
    see a consistent pair set matching `pairs`.
    """
    condition_patterns = LLMConditionFilter(condition, labels)

    base_by_pair: dict[tuple[str, str], dict] = {}
    for entry in base_results.values():
        o1, o2 = entry.get("organ1", ""), entry.get("organ2", "")
        if o1 and o2:
            base_by_pair[(min(o1, o2), max(o1, o2))] = entry

    results: dict = {}
    for (o1, o2) in pairs:
        base_entry = base_by_pair.get((o1, o2))
        if base_entry is None:
            print(f"  [!] no cached 'all' entry for {o1} <-> {o2} -- run the base search first")
            continue

        all_papers = base_entry.get("papers", [])
        cond_papers = _filter_by_condition(all_papers, condition_patterns)
        key_players = extract_key_players(cond_papers)

        print(f"  {o1} <-> {o2}: {len(all_papers)} papers -> {len(cond_papers)} after "
              f"'{condition}' filter")

        results[f"{o1}|{o2}"] = {
            "organ1":                 o1,
            "organ2":                 o2,
            "pubmed_query":           base_entry.get("pubmed_query", ""),
            "strategy_used":          f"{base_entry.get('strategy_used', '')}+condition:{condition}",
            "n_papers_found":         len(cond_papers),
            "n_papers_found_raw":     len(all_papers),
            "papers":                 cond_papers,
            "key_players":            key_players,
            "search_date":            datetime.now().isoformat(),
        }

    return results


# ── PubMed search ─────────────────────────────────────────────────────────────

def search_pubmed(query: str, max_results: int, years_back: int) -> list[str]:
    """Return up to max_results PMIDs for query.

    Two limits of this call (found 2026-10-23, see pubmed_final_query_check.py): esearch never returns more than
    10,000 IDs, whatever max_results is; and `mindate` is ignored because no `maxdate` is sent, so years_back has no effect.
    """
    min_date = (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y/%m/%d")
    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
        "mindate": min_date,
        "datetype": "pdat",
    }
    # POST avoids the URL-length ceiling GET requests hit once the pair's
    # combined organ+metabolic+crosstalk clauses push the query past a few
    # thousand characters — NCBI returns HTTP 414 on GET in that case, which
    # silently degrades that pair's results to zero rather than erroring loudly.
    resp = _ncbi_get("esearch.fcgi", params, method="POST")
    if resp is None:
        return []
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


# ── Main search loop ──────────────────────────────────────────────────────────

def run_search(organ_pairs, output_path, resume=True, force_empty=False):
    output_path = Path(output_path)
    results: dict = {}

    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            results = json.load(f)
        n_cached = len(results)
        n_empty  = sum(1 for v in results.values() if v.get("n_papers_found", 0) == 0)
        print(f"[i] Resuming: {n_cached} cached ({n_empty} empty).")
        if force_empty:
            empty_keys = [k for k, v in results.items() if v.get("n_papers_found", 0) == 0]
            for k in empty_keys:
                del results[k]
            print(f"[i] Cleared {len(empty_keys)} empty entries for re-search.")

    crosstalk_patterns = _compile_patterns(CROSSTALK_KEYWORDS)

    total = len(organ_pairs)
    for idx, (o1, o2) in enumerate(organ_pairs):
        edge_key = f"{o1}|{o2}"
        sym_key  = f"{o2}|{o1}"

        if edge_key in results or sym_key in results:
            n = (results.get(edge_key) or results.get(sym_key, {})).get("n_papers_found", 0)
            print(f"  [{idx+1}/{total}] Cached ({n} papers): {o1} <-> {o2}")
            continue

        query = build_query(o1, o2)
        print(f"\n  [{idx+1}/{total}] {o1} <-> {o2}")
        print(f"    Query: {query[:120]}...")
        time.sleep(DELAY)

        pmids = search_pubmed(query, MAX_PAPERS, YEARS_BACK)
        print(f"    Found {len(pmids)} PMIDs — fetching abstracts...", flush=True)
        time.sleep(DELAY)

        papers_raw = fetch_abstracts(pmids, delay=DELAY)
        organ1_patterns = _compile_patterns(ORGAN_ALIASES.get(o1, [o1]))
        organ2_patterns = _compile_patterns(ORGAN_ALIASES.get(o2, [o2]))
        papers = _filter_same_sentence_crosstalk(papers_raw, organ1_patterns,
                                                 organ2_patterns, crosstalk_patterns,
                                                 organ1_name=o1, organ2_name=o2,
                                                 all_organ_patterns=ALL_ORGAN_PATTERNS)
        key_players = extract_key_players(papers)

        n = len(papers)
        print(
            f"    => {len(papers_raw)} papers retrieved, {n} kept "
            f"(both organs + crosstalk term in same sentence)"
            f" | hormones: {len(key_players['hormones'])}"
            f" | metabolites: {len(key_players['metabolites'])}"
            f" | proteins: {len(key_players['proteins'])}"
        )

        results[edge_key] = {
            "organ1":          o1,
            "organ2":          o2,
            "pubmed_query":    query,
            "strategy_used":   "metabolic-single-query",
            "n_papers_found":  n,
            "n_papers_found_raw": len(papers_raw),
            "papers":          papers,
            "key_players":     key_players,
            "search_date":     datetime.now().isoformat(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    total_papers = sum(v.get("n_papers_found", 0) for v in results.values())
    empty = sum(1 for v in results.values() if v.get("n_papers_found", 0) == 0)
    print(f"\n[ok] Done. {len(results)} edges | {total_papers} total papers | {empty} empty.")
    print(f"     Saved: {output_path}")
    return results


# ── Visualization ─────────────────────────────────────────────────────────────

def _lit_sankey_html(variant_id: str, title: str, sankey_json: str) -> str:
    """Canvas-drawn circular Sankey card for one variant."""
    canvas_id = f"lit-sankey-{variant_id}"
    return f"""
    <div class="ls-card ls-sticky">
    <div class="ls-card-title">{title}</div>
    <div style="text-align:center">
      <canvas id="{canvas_id}" style="display:block"></canvas>
    </div>
    <script>
    (function() {{
      const D = {sankey_json};
      const canvas = document.getElementById('{canvas_id}');
      if (!D.links.length) {{ canvas.style.display = 'none'; return; }}
      const ctx = canvas.getContext('2d');

      const organs = D.organs, links = D.links;
      const SIZE = 640, CX = SIZE / 2, CY = SIZE / 2, R = 200, ARC_W = 14, LABEL_GAP = 10;
      const GAP_DEG = organs.length > 1 ? 1.4 : 0;

      const SCALE = 2;
      canvas.width = SIZE * SCALE; canvas.height = SIZE * SCALE;
      canvas.style.width = '100%'; canvas.style.maxWidth = '560px'; canvas.style.height = 'auto'; canvas.style.margin = '0 auto';
      ctx.scale(SCALE, SCALE);
      ctx.clearRect(0, 0, SIZE, SIZE);

      const totals = {{}};
      organs.forEach(o => totals[o] = 0);
      links.forEach(l => {{ totals[l.source] += l.value; totals[l.target] += l.value; }});
      const grand = organs.reduce((s, o) => s + totals[o], 0) || 1;
      const usableDeg = 360 - GAP_DEG * organs.length;

      const toRad = d => d * Math.PI / 180;
      const pointOnCircle = (deg, radius) => {{
        const rad = toRad(deg);
        return [CX + radius * Math.cos(rad), CY + radius * Math.sin(rad)];
      }};
      const colorFor = o => D.organColors[o] || '#2563eb';

      let angle = -90;
      const organRange = {{}};
      organs.forEach(o => {{
        const span = (totals[o] / grand) * usableDeg;
        organRange[o] = {{ start: angle, end: angle + span }};
        angle += span + GAP_DEG;
      }});

      const sortedLinks = [...links].sort((a, b) => (a.source + a.target).localeCompare(b.source + b.target));
      const cursor = {{}};
      organs.forEach(o => cursor[o] = organRange[o].start);

      const ribbonPaths = [];
      sortedLinks.forEach(l => {{
        const span = (l.value / grand) * usableDeg;
        const a0 = cursor[l.source]; cursor[l.source] += span;
        const b0 = cursor[l.target]; cursor[l.target] += span;
        const a1 = a0 + span, b1 = b0 + span;

        const [x0, y0]   = pointOnCircle(a0, R);
        const [x0b, y0b] = pointOnCircle(a1, R);
        const [x1, y1]   = pointOnCircle(b0, R);
        const [x1b, y1b] = pointOnCircle(b1, R);

        const path = new Path2D();
        path.moveTo(x0, y0);
        path.quadraticCurveTo(CX, CY, x1, y1);
        path.lineTo(x1b, y1b);
        path.quadraticCurveTo(CX, CY, x0b, y0b);
        path.closePath();

        const grad = ctx.createLinearGradient(x0, y0, x1, y1);
        grad.addColorStop(0, colorFor(l.source) + '99');
        grad.addColorStop(1, colorFor(l.target) + '55');
        ctx.fillStyle = grad;
        ctx.fill(path);

        ribbonPaths.push({{ path, link: l }});
      }});

      organs.forEach(o => {{
        const {{ start, end }} = organRange[o];
        if (end <= start) return;
        ctx.beginPath();
        ctx.arc(CX, CY, R, toRad(start), toRad(end));
        ctx.lineWidth = ARC_W;
        ctx.strokeStyle = colorFor(o);
        ctx.stroke();

        const mid = (start + end) / 2;
        const [lx, ly] = pointOnCircle(mid, R + ARC_W / 2 + LABEL_GAP);
        const flip = mid > 90 && mid < 270;
        ctx.save();
        ctx.translate(lx, ly);
        ctx.rotate(toRad(mid) + (flip ? Math.PI : 0));
        ctx.fillStyle = '#e2e8f0';
        ctx.font = '14px system-ui';
        ctx.textAlign = flip ? 'right' : 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(o, 0, 0);
        ctx.restore();
      }});

      const tip = document.createElement('div');
      tip.style.cssText = 'position:fixed;pointer-events:none;display:none;background:#1e293b;' +
        'border:1px solid #475569;border-radius:6px;padding:6px 10px;font-size:0.76rem;' +
        'color:#e2e8f0;z-index:999';
      document.body.appendChild(tip);

      canvas.addEventListener('mousemove', e => {{
        const r = canvas.getBoundingClientRect();
        const x = (e.clientX - r.left) * (canvas.width / r.width);
        const y = (e.clientY - r.top) * (canvas.height / r.height);
        let hit = null;
        for (const rp of ribbonPaths) {{
          if (ctx.isPointInPath(rp.path, x, y)) {{ hit = rp.link; break; }}
        }}
        if (hit) {{
          tip.innerHTML = `<strong>${{hit.source}} ↔ ${{hit.target}}</strong><br>${{hit.value.toLocaleString()}} co-occurring papers`;
          tip.style.display = 'block';
          tip.style.left = (e.clientX + 14) + 'px';
          tip.style.top = (e.clientY - 10) + 'px';
        }} else {{
          tip.style.display = 'none';
        }}
      }});
      canvas.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});
    }})();
    </script>
    </div>
"""


def _lit_bar_rows(items: list[tuple[str, int, str]], label_px: int, max_row_px: int) -> str:
    """
    Horizontal bar plot as plain HTML rows. items = (label, value, css background); the
    absolute value is printed right after each bar. The row height shrinks with the number of
    rows (CSS only, see .ls-bars) so that even ~90 connections fit into the window.
    """
    if not items:
        return ""
    top = max(v for _, v, _ in items) or 1
    rows = "".join(
        f'<div class="ls-bar-row"><span class="ls-bar-lbl" title="{label}">{label}</span>'
        f'<span class="ls-bar-track"><span class="ls-bar" '
        f'style="width:{(max(v / top * 80, 0.8) if v > 0 else 0):.1f}%;background:{bg}"></span>'
        f'<span class="ls-bar-val">{v:,}</span></span></div>'
        for label, v, bg in items)
    return f'<div class="ls-bars" style="--n:{len(items)};--lbl-w:{label_px}px;--rh-max:{max_row_px}px">{rows}</div>'


def _lit_variant_html(variant_id: str, v: dict) -> str:
    """One condition (or All papers): papers per organ | papers per connection | circular Sankey."""
    no_links = '<p style="color:#64748b;font-size:0.8rem">No cross-mention evidence yet.</p>'
    return f"""
    <div class="ls-grid">
      <div class="ls-card"><div class="ls-card-title">Papers per organ</div>{v["organ_bars"]}</div>
      <div class="ls-card"><div class="ls-card-title">Papers per connection</div>{v["link_bars"] or no_links}</div>
      {_lit_sankey_html(variant_id, "Sankey diagram", v["sankey_json"])}
    </div>
"""


def _literature_stats_section(organs: list[str], results_by_pair: dict, allowed_pairs: set) -> dict:
    """
    Literature-statistics data for one variant (all/healthy/obese): the
    circular-Sankey data plus per-organ and per-connection paper counts.
    Ribbon width = same-sentence co-occurring paper count for that organ pair
    (further condition-filtered for the healthy/obese variants -- see
    condition_variants in build_viz()).

    This pipeline searches per PAIR, not per organ, so "papers per organ" is
    derived as the union of PMIDs across every pair touching that organ (a
    paper supporting two of an organ's connections is only counted once).

    Returns {"sankey_json": str, "organ_bars": html, "link_bars": html} (the two bar plots).
    """
    from Visualisation.networkBuilderUtils import ORGAN_COLORS, DEFAULT_NODE_COLOR

    organ_pmids: dict[str, set] = {o: set() for o in organs}
    for pair in allowed_pairs:
        o1, o2 = pair
        data = results_by_pair.get(pair, {})
        if data.get("n_papers_found", 0) <= 0:
            continue
        pmids = {p.get("pmid") for p in data.get("papers", []) if p.get("pmid")}
        organ_pmids.setdefault(o1, set()).update(pmids)
        organ_pmids.setdefault(o2, set()).update(pmids)
    organ_papers = {o: len(organ_pmids.get(o, set())) for o in organs}

    links = []
    for pair in allowed_pairs:
        o1, o2 = pair
        n = results_by_pair.get(pair, {}).get("n_papers_found", 0)
        if n <= 0:
            continue
        # The organ with the larger total paper count is the ribbon "source".
        if organ_papers.get(o1, 0) < organ_papers.get(o2, 0):
            o1, o2 = o2, o1
        links.append({"source": o1, "target": o2, "value": n})
    links.sort(key=lambda l: (l["source"], l["target"]))      # allowed_pairs is a set: fixed order

    def _color(o: str) -> str:
        return ORGAN_COLORS.get(o, DEFAULT_NODE_COLOR)

    # Bars: organs in their own colour (as in the Sankey), connections in black and white (white bars
    # on the dark card). Equal counts are ordered by name.
    organ_bars = _lit_bar_rows(
        [(o, organ_papers[o], _color(o)) for o in sorted(organs, key=lambda o: (-organ_papers[o], o))],
        label_px=110, max_row_px=24)
    link_bars = _lit_bar_rows(
        [(f'{l["source"]} ↔ {l["target"]}', l["value"], "#f8fafc")
         for l in sorted(links, key=lambda l: (-l["value"], l["source"], l["target"]))],
        label_px=165, max_row_px=20)

    sankey_organs = [o for o in sorted(organs, key=lambda o: -organ_papers[o])
                     if any(l["source"] == o or l["target"] == o for l in links)]
    sankey_json = json.dumps({
        "organs":      sankey_organs,
        "organColors": {o: ORGAN_COLORS.get(o, DEFAULT_NODE_COLOR) for o in sankey_organs},
        "links":       links,
    }, ensure_ascii=False)
    return {"sankey_json": sankey_json, "organ_bars": organ_bars, "link_bars": link_bars}


def _lit_stats_tab_wrap(variants: dict) -> str:
    """
    Literature Statistics layout: the intro text, one button per view (All papers + the conditions)
    and, below, ONLY the selected view as three columns: papers per organ (bar plot, left),
    papers per connection (bar plot, middle) and the circular Sankey diagram (right).

    variants: key -> {"sankey_json", "organ_bars", "link_bars"}, or
    {"message": html} for a condition whose pipeline has not been run.
    """
    labels = dict(CONDITION_NAME)                       # slug -> name ("all" -> "All papers")
    keys = [k for k in ["all"] + CONDITION_SLUGS if k in variants]

    buttons = "".join(
        f'<button class="lit-stats-selector-btn{" active" if k == "all" else ""}" '
        f'data-variant="{k}" onclick="showLitStatsVariant(\'{k}\')">{labels[k]}</button>'
        for k in keys)

    panes = ""
    for k in keys:
        v = variants[k]
        inner = v["message"] if "message" in v else _lit_variant_html(k, v)
        panes += (f'<div class="lit-stats-variant" id="lit-stats-variant-{k}" '
                  f'style="display:{"block" if k == "all" else "none"}">{inner}</div>')

    return f"""
    <style>
      .ls-grid {{ display:grid; grid-template-columns:minmax(220px,3fr) minmax(300px,4fr) minmax(300px,5fr);
                  gap:16px; align-items:start; }}
      @media (max-width:1000px) {{ .ls-grid {{ grid-template-columns:1fr 1fr; }}
                                   .ls-grid .ls-sticky {{ grid-column:1 / -1; }} }}
      @media (max-width:640px) {{ .ls-grid {{ grid-template-columns:1fr; }} }}
      .ls-sticky {{ position:sticky; top:0; }}
      .ls-card {{ background:#0f172a; border:1px solid #334155; border-radius:8px;
                  padding:8px 10px 10px; min-width:0; }}
      .ls-card-title {{ font-size:0.74rem; font-weight:600; color:#94a3b8; margin-bottom:8px;
                        text-transform:uppercase; letter-spacing:0.03em; }}
      /* bar plots: row height = (window height - 280px) / number of rows, between 8px and --rh-max */
      .ls-bars {{ --rh:clamp(8px, calc((100vh - 280px) / var(--n)), var(--rh-max)); }}
      .ls-bar-row {{ display:flex; align-items:center; height:var(--rh); line-height:1;
                     font-size:clamp(8px, calc(var(--rh) * 0.8), 12px); }}
      .ls-bar-lbl {{ flex:0 0 var(--lbl-w); padding-right:6px; text-align:right; color:#e2e8f0;
                     white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
      .ls-bar-track {{ flex:1; min-width:0; height:100%; display:flex; align-items:center; gap:4px; }}
      .ls-bar {{ flex:none; height:68%; border-radius:0 2px 2px 0; }}
      .ls-bar-val {{ color:#94a3b8; white-space:nowrap; }}
      .lit-stats-selector-btn {{
        background:#1e293b;border:1px solid #334155;color:#94a3b8;
        border-radius:6px;padding:5px 12px;font-size:0.76rem;cursor:pointer;
      }}
      .lit-stats-selector-btn:hover {{ background:#334155;color:#e2e8f0; }}
      .lit-stats-selector-btn.active {{ background:#6366f1;border-color:#6366f1;color:#fff; }}
    </style>
    {_info_text('literature_stats_intro')}
    <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px">{buttons}</div>
    {panes}
    <script>
    function showLitStatsVariant(variant) {{
      document.querySelectorAll('.lit-stats-variant').forEach(el => {{
        el.style.display = (el.id === 'lit-stats-variant-' + variant) ? 'block' : 'none';
      }});
      document.querySelectorAll('.lit-stats-selector-btn').forEach(btn => {{
        btn.classList.toggle('active', btn.dataset.variant === variant);
      }});
    }}
    </script>
"""


def _literature_stats_tab_content(organs: list[str], results_by_pair: dict, allowed_pairs: set,
                                   condition_data: dict[str, tuple | None] | None) -> str:
    """
    Literature Statistics tab: All papers + one variant per condition, computed from
    different paper sets (see condition_data in _build_info_tabs()), laid out
    by _lit_stats_tab_wrap().
    """
    condition_data = condition_data or {}

    variants = {"all": _literature_stats_section(organs, results_by_pair, allowed_pairs)}
    for cond in CONDITION_SLUGS:
        data = condition_data.get(cond)
        if data is None:
            variants[cond] = {"message": (
                f'<p class="info-p" style="color:#64748b">'
                f'No condition-filtered literature statistics for "{CONDITION_NAME[cond]}" are available '
                f'(no verified papers carry this label yet; the labels come from '
                f'<code>python run_metabolic_lit_search.py --classify-conditions</code>).</p>')}
        else:
            cond_organs, cond_results_by_pair, cond_pairs = data
            variants[cond] = _literature_stats_section(cond_organs, cond_results_by_pair, cond_pairs)
    return _lit_stats_tab_wrap(variants)


def _venn_svg(n_only_a: int, n_only_b: int, n_both: int,
              label_a: str, color_a: str,
              label_b: str, color_b: str,
              n_outer: int | None = None,
              outer_label: str = "All",
              subtitle_a: str = "", subtitle_b: str = "") -> str:
    """
    Euler diagram: outer rounded-rect (All), two overlapping circles (A and B).
    All counts are integers; positions are fixed (non-proportional) for clarity.
    """
    W, H = 420, 210
    cx_a, cx_b, cy, r = 155, 265, 108, 82

    # Region label x positions
    lx_a   = cx_a - r // 2 - 12   # left of left circle
    lx_ab  = (cx_a + cx_b) // 2   # centre of overlap
    lx_b   = cx_b + r // 2 + 12   # right of right circle
    ly_num = cy + 5
    ly_lbl = cy + 22

    outer_text = ""
    if n_outer is not None:
        outer_text = (
            f'<text x="22" y="{H - 14}" fill="#475569" font-size="10" font-family="inherit">'
            f'{outer_label} only: {n_outer:,}</text>'
        )

    total_a = n_only_a + n_both
    total_b = n_only_b + n_both

    sub_a = subtitle_a or f"total {total_a:,}"
    sub_b = subtitle_b or f"total {total_b:,}"

    return (
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg"'
        f' style="width:100%;max-width:{W}px;display:block;margin:10px 0">'
        # Background (All)
        f'<rect x="4" y="4" width="{W-8}" height="{H-8}" rx="12"'
        f' fill="#0f172a" stroke="#334155" stroke-width="1.5"/>'
        f'<text x="16" y="20" fill="#475569" font-size="10" font-weight="600"'
        f' font-family="inherit" text-anchor="start">{outer_label}</text>'
        # Left circle (A)
        f'<circle cx="{cx_a}" cy="{cy}" r="{r}"'
        f' fill="{color_a}" fill-opacity="0.18" stroke="{color_a}" stroke-width="1.5"/>'
        # Right circle (B)
        f'<circle cx="{cx_b}" cy="{cy}" r="{r}"'
        f' fill="{color_b}" fill-opacity="0.18" stroke="{color_b}" stroke-width="1.5"/>'
        # Label A (top of left circle)
        f'<text x="{cx_a}" y="{cy - r - 10}" text-anchor="middle" fill="{color_a}"'
        f' font-size="11" font-weight="700" font-family="inherit">{label_a}</text>'
        f'<text x="{cx_a}" y="{cy - r + 2}" text-anchor="middle" fill="{color_a}"'
        f' font-size="9" font-family="inherit" fill-opacity="0.75">{sub_a}</text>'
        # Label B (top of right circle)
        f'<text x="{cx_b}" y="{cy - r - 10}" text-anchor="middle" fill="{color_b}"'
        f' font-size="11" font-weight="700" font-family="inherit">{label_b}</text>'
        f'<text x="{cx_b}" y="{cy - r + 2}" text-anchor="middle" fill="{color_b}"'
        f' font-size="9" font-family="inherit" fill-opacity="0.75">{sub_b}</text>'
        # Count: A only
        f'<text x="{lx_a}" y="{ly_num}" text-anchor="middle" fill="#e2e8f0"'
        f' font-size="18" font-weight="700" font-family="inherit">{n_only_a:,}</text>'
        f'<text x="{lx_a}" y="{ly_lbl}" text-anchor="middle" fill="#94a3b8"'
        f' font-size="9" font-family="inherit">only</text>'
        # Count: overlap
        f'<text x="{lx_ab}" y="{ly_num}" text-anchor="middle" fill="#e2e8f0"'
        f' font-size="18" font-weight="700" font-family="inherit">{n_both:,}</text>'
        f'<text x="{lx_ab}" y="{ly_lbl}" text-anchor="middle" fill="#94a3b8"'
        f' font-size="9" font-family="inherit">shared</text>'
        # Count: B only
        f'<text x="{lx_b}" y="{ly_num}" text-anchor="middle" fill="#e2e8f0"'
        f' font-size="18" font-weight="700" font-family="inherit">{n_only_b:,}</text>'
        f'<text x="{lx_b}" y="{ly_lbl}" text-anchor="middle" fill="#94a3b8"'
        f' font-size="9" font-family="inherit">only</text>'
        f'{outer_text}'
        f'</svg>'
    )


def _llm_prompts_html() -> str:
    """HTML of the "LLM Prompts" tab: every prompt, generated from the pipeline code."""
    from Literature_Search.prompt_catalog import build_prompt_catalog, prompts_html
    return prompts_html(build_prompt_catalog(CONDITION_LLM_MODEL, CONNECTION_TYPE_LLM_MODEL,
                                             DESCRIPTION_LLM_MODEL, CONNECTION_TYPES, LLM_MAX_PAPERS))


def paper_search_payload(results_by_pair: dict, condition_labels: dict) -> dict:
    """Data of the dashboard's Paper Search tab: every retrieved paper once, [pmid, title, year, doi, label index
    (-1 = no usable LLM label), author keywords], plus the paper indices of each organ pair ("A|B", sorted).
    Which papers belong to a condition view is decided by the label index, as everywhere else."""
    import html as _html
    slugs = [c["slug"] for c in CONDITIONS]
    label_index = {c["label"]: i for i, c in enumerate(CONDITIONS)}
    table: list = []
    order: dict[str, int] = {}
    pairs: dict[str, list[int]] = {}
    for (o1, o2), entry in sorted(results_by_pair.items()):
        ids = []
        for p in entry.get("papers", []):
            pid = str(p.get("pmid", "")).strip()
            if not pid:
                continue
            if pid not in order:
                rec = condition_labels.get(pid)
                li = label_index.get(rec["label"], -1) if rec and rec.get("status") == "ok" else -1
                kws = "; ".join(k for k in (_html.unescape(str(x)).strip() for x in (p.get("keywords") or [])) if k)
                order[pid] = len(table)
                table.append([pid, _html.unescape(p.get("title") or "").strip(), str(p.get("year") or ""),
                              p.get("doi") or "", li, kws])
            ids.append(order[pid])
        pairs[f"{min(o1, o2)}|{max(o1, o2)}"] = ids
    return {"slugs": slugs, "papers": table, "pairs": pairs}


def _description_text_values(model: str) -> dict:
    """Placeholder values of the dashboard's "Descriptions" text for the description model."""
    from Literature_Search.llm_descriptions import description_settings
    s = description_settings(model)
    grounding = ""
    if s["claude"]:
        grounding = (
            '<p class="info-p"><strong>Sources are checked for hallucinations.</strong> The papers are picked by '
            'index from the PubMed records already retrieved; the model never types a title, PMID or year, so the '
            'reference list always consists of real retrieved papers. The summary is written with the Claude '
            '<em>citations</em> feature: the API attaches to each sentence the passage of the paper that supports it, '
            'and the reference numbers in the text are generated from these citations. A summary is accepted only if '
            'every cited passage occurs word for word in the title or abstract of the cited paper and every sentence '
            'carries a citation; otherwise it is regenerated (up to 3 times), uncited sentences and uncited trailing '
            'clauses are removed, and if nothing citable is left no description is shown.</p>')
    if s["claude"]:
        organ_method = (
            '<strong style="color:#e2e8f0">Select and summarise</strong>: as for the edge descriptions, a random pool of up to '
            f'{s["pool"]} of these papers is drawn, the model selects the {s["top"]} most relevant ones and writes the '
            'cited summary from them. It covers the organ\'s primary metabolic substrates, energy production pathways '
            'and key hormonal regulation signals.')
    else:
        organ_method = (
            '<strong style="color:#e2e8f0">Summarise</strong>: the first six papers (abstracts cut to 400 characters) are '
            'given to the LLM, which writes 5 sentences covering the organ\'s primary metabolic substrates, energy '
            'production pathways and key hormonal regulation signals.')
    return {"desc_pool": s["pool"], "desc_top": s["top"], "desc_sentences": s["sentences"],
            "desc_grounding": grounding, "desc_organ_method": organ_method,
            "desc_organ_sentences": "5–7" if s["claude"] else "5"}


def _network_analysis_tab_html(results_by_pair: dict,
                                condition_data: dict,
                                all_organs: list[str]) -> str:
    """Network Analysis tab: one row per view (All papers + every condition), how the condition
    views share organ pairs, and the per-organ degree in every view."""

    def _net_stats(rbp: dict) -> tuple[set, set]:
        """Return (edges_set, nodes_set) for pairs with >=1 paper."""
        edges, nodes = set(), set()
        for v in rbp.values():
            if v.get("n_papers_found", 0) > 0:
                o1, o2 = v.get("organ1", ""), v.get("organ2", "")
                if o1 and o2:
                    edges.add((min(o1, o2), max(o1, o2)))
                    nodes.update([o1, o2])
        return edges, nodes

    views = [("all", *_net_stats(results_by_pair))]
    for slug in CONDITION_SLUGS:
        cd = condition_data.get(slug)
        views.append((slug, *(_net_stats(cd[1]) if cd else (set(), set()))))
    edges_of = {s_: e for s_, e, _ in views}
    edges_all = edges_of["all"]
    max_edges = len(all_organs) * (len(all_organs) - 1) // 2

    def _density(e: int, n: int) -> str:
        return f"{100*e/(n*(n-1)/2):.1f}%" if n > 1 else "—"

    def _iso_cell(iso: list) -> str:
        return ", ".join(iso) if iso else '<em style="color:#475569">none</em>'

    td_r = 'style="padding:4px 10px 4px 0;text-align:right;color:#e2e8f0"'
    summary_rows = ""
    for s_, edges, nodes in views:
        iso = sorted(o for o in all_organs if o not in nodes)
        summary_rows += (
            f'<tr><td style="padding:4px 10px 4px 0;color:{CONDITION_COLOR[s_]};font-weight:600">{CONDITION_NAME[s_]}</td>'
            f'<td {td_r}>{len(nodes)}</td>'
            f'<td style="padding:4px 10px 4px 0;text-align:right;color:#e2e8f0;font-weight:700">{len(edges)}</td>'
            f'<td style="padding:4px 10px 4px 0;text-align:right;color:#64748b">{max_edges}</td>'
            f'<td {td_r}>{_density(len(edges), len(nodes))}</td>'
            f'<td style="padding:4px 0;color:#94a3b8;font-size:11px">{_iso_cell(iso)}</td></tr>')

    # pairwise sharing matrix (diagonal = connections of that view)
    th_s = 'style="padding:3px 6px 6px 0;font-weight:600;text-align:right;font-size:10px;border-bottom:1px solid #334155;color:%s"'
    matrix_head = "".join(f"<th {th_s % CONDITION_COLOR[s_]}>{CONDITION_NAME[s_].replace('/', '/<wbr>')}</th>"
                          for s_ in CONDITION_SLUGS)
    matrix_rows = ""
    for a_ in CONDITION_SLUGS:
        cells = "".join(
            f'<td style="padding:3px 6px 3px 0;text-align:right;color:{"#e2e8f0" if a_ == b_ else "#94a3b8"};'
            f'font-weight:{"700" if a_ == b_ else "400"}">{len(edges_of[a_] & edges_of[b_])}</td>'
            for b_ in CONDITION_SLUGS)
        matrix_rows += (f'<tr><td style="padding:3px 10px 3px 0;color:{CONDITION_COLOR[a_]};white-space:nowrap">'
                        f'{CONDITION_NAME[a_]}</td>{cells}</tr>')

    # per-organ degree in every view
    def _degrees(edges: set, organ_list: list) -> dict[str, int]:
        deg = {o: 0 for o in organ_list}
        for o1, o2 in edges:
            if o1 in deg: deg[o1] += 1
            if o2 in deg: deg[o2] += 1
        return deg

    deg = {s_: _degrees(edges_of[s_], all_organs) for s_ in edges_of}
    top = max(deg["all"].values(), default=1) or 1
    degree_rows = ""
    for organ in sorted(all_organs, key=lambda o: -deg["all"][o]):
        cells = "".join(
            f'<td style="padding:3px 8px 3px 0;white-space:nowrap">'
            f'<span style="display:inline-block;min-width:18px;text-align:right;color:#e2e8f0;font-weight:600">'
            f'{deg[s_][organ]}</span> '
            f'<span style="display:inline-block;width:{int(deg[s_][organ] / top * 40)}px;height:8px;'
            f'background:{CONDITION_COLOR[s_]};border-radius:2px;vertical-align:middle"></span></td>'
            for s_, _, _ in views)
        degree_rows += (f'<tr><td style="padding:3px 10px 3px 0;color:#cbd5e1;white-space:nowrap">{organ}</td>'
                        f'{cells}</tr>')
    degree_head = "".join(
        f'<th style="padding:3px 8px 6px 0;font-weight:600;text-align:left;font-size:11px;'
        f'border-bottom:1px solid #334155;color:{CONDITION_COLOR[s_]}">{CONDITION_NAME[s_]}</th>'
        for s_, _, _ in views)

    th = 'style="padding:3px 10px 6px 0;color:#64748b;font-weight:600;text-align:right;border-bottom:1px solid #334155"'
    th_l = 'style="padding:3px 10px 6px 0;color:#64748b;font-weight:600;text-align:left;border-bottom:1px solid #334155"'

    # Layout: network summary (left) | connections shared by each pair of views (right); per-organ degree below
    return f"""
        <style>
          .na-grid {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:28px; align-items:start; }}
          @media (max-width:1100px) {{ .na-grid {{ grid-template-columns:minmax(0,1fr); }} }}
        </style>
        <div class="na-grid">
          <div>
        {_info_text('network_summary_intro')}
        <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:14px">
          <thead>
            <tr>
              <th {th_l}>View</th>
              <th {th}>Organs</th>
              <th {th}>Edges</th>
              <th {th}>Max edges</th>
              <th {th}>Density</th>
              <th style="padding:3px 0 6px;color:#64748b;font-weight:600;text-align:left;border-bottom:1px solid #334155">Isolated organs</th>
            </tr>
          </thead>
          <tbody>{summary_rows}</tbody>
        </table>
          </div>
          <div>
        {_info_text('edge_overlap_intro', n_conditions=len(CONDITION_SLUGS), n_all_edges=len(edges_all))}
        <p class="info-p" style="margin-bottom:6px">Connections shared by each pair of condition views
        (diagonal: connections of that view):</p>
        <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:14px">
          <thead><tr><th {th_l}></th>{matrix_head}</tr></thead>
          <tbody>{matrix_rows}</tbody>
        </table>
          </div>
        </div>
        {_info_text('per_organ_degree_intro')}
        <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:6px">
          <thead><tr><th {th_l}>Organ</th>{degree_head}</tr></thead>
          <tbody>{degree_rows}</tbody>
        </table>
"""


KP_ENRICHMENT_XLSX = Path(__file__).resolve().parent / "key_player_analysis" / "tables" / "key_player_condition_enrichment.xlsx"


_KP_CAT_LABEL = {"hormones": "Hormones", "metabolites": "Metabolites", "proteins": "Proteins / Transporters"}
_KP_TOP = 15                      # bars per side and category (as in the enrichment figures of generate_key_player_analysis.py)


def _kp_pct(v: float) -> str:
    return f"{v:.2f}" if v < 1 else f"{v:.1f}"


def _kp_panel(df, cat: str, name_a: str, name_b: str, col_a: str, col_b: str) -> str:
    """One diverging bar plot (one key player category) of the Key Player Enrichment tab: top bars per side among the terms with
    Benjamini-Hochberg q < 0.05, sorted by log2 odds ratio (largest at the top). To the right of the zero line: enriched in A (the
    condition), to the left: enriched in B (the rest). The numbers are the % of papers that mention the term (A vs B)."""
    import html as _html
    d = df[(df.category == cat) & (df["significant_q<0.05"])]
    up = d[d.log2_odds_ratio > 0]
    down = d[d.log2_odds_ratio < 0]
    sel = [("a", r) for r in up.nlargest(_KP_TOP, "log2_odds_ratio").itertuples()] + \
          [("b", r) for r in down.nsmallest(_KP_TOP, "log2_odds_ratio").sort_values("log2_odds_ratio", ascending=False).itertuples()]
    head = (f'<div class="kp-panel-title">{_KP_CAT_LABEL[cat]}</div>'
            f'<div class="kp-panel-sub">{len(up)} enriched in {_html.escape(name_a)} · {len(down)} enriched in {_html.escape(name_b)} '
            f'({int((df.category == cat).sum())} tested)</div>')
    if not sel:
        return f'<div class="kp-panel">{head}<p class="kp-none">no significant terms (q &lt; 0.05)</p></div>'
    left = max([0.4] + [-r.log2_odds_ratio for s, r in sel if s == "b"])               # length of the longest bar to the left (log2 units)
    right = max([0.4] + [r.log2_odds_ratio for s, r in sel if s == "a"])
    tot = (left + right) * 1.7                                                         # room right of the bars for the numbers
    zero = 100 * left / tot
    rows = ""
    for side, r in sel:
        v = r.log2_odds_ratio
        w = 100 * abs(v) / tot
        if side == "a":
            bar_l, lab_l, col = zero, zero + w + 0.8, col_a
        else:
            bar_l, lab_l, col = zero - w, zero + 0.8, col_b
        tip = (f"{r.term}: {r.n_papers_A:,} of {int(r.N_A):,} papers ({r.pct_A:.2f}%) vs {r.n_papers_B:,} of {int(r.N_B):,} ({r.pct_B:.2f}%); "
               f"log2 odds ratio {v:+.2f}; q = {r.q_value_BH:.1e}")
        rows += (f'<div class="kp-row" title="{_html.escape(tip, quote=True)}"><span class="kp-name">{_html.escape(str(r.term))}</span>'
                 f'<span class="kp-track"><span class="kp-zero" style="left:{zero:.2f}%"></span>'
                 f'<span class="kp-bar" style="left:{bar_l:.2f}%;width:{w:.2f}%;background:{col}"></span>'
                 f'<span class="kp-val" style="left:{lab_l:.2f}%">{_kp_pct(r.pct_A)}% vs {_kp_pct(r.pct_B)}%</span></span></div>')
    step = 1 if left + right <= 8 else 2 if left + right <= 16 else 4
    ticks = ""
    for t_ in range(-int(left // step) * step, int(right // step) * step + 1, step):
        ticks += f'<span class="kp-tick" style="left:{zero + 100 * t_ / tot:.2f}%">{t_}</span>'
    axis = (f'<div class="kp-row kp-axisrow"><span class="kp-name"></span><span class="kp-track kp-axis">{ticks}</span></div>'
            f'<div class="kp-axis-label">log2 odds ratio (← enriched in {_html.escape(name_b)} | enriched in {_html.escape(name_a)} →)</div>')
    return f'<div class="kp-panel">{head}{rows}{axis}</div>'


def _key_player_enrichment_tab() -> str:
    """"Key Player Enrichment" tab (Result Statistics): like Literature Statistics a row of buttons and only the selected comparison below it, here as
    three diverging bar plots (hormones | metabolites | proteins/transporters) drawn from the sheets of key_player_condition_enrichment.xlsx
    (<condition>_vs_rest for the seven condition labels and healthy_vs_obesity)."""
    import html as _html
    intro = _info_text("key_player_enrichment_intro")
    try:
        import pandas as pd
        xls = pd.ExcelFile(KP_ENRICHMENT_XLSX)
    except (OSError, ValueError, ImportError) as err:
        return intro + (f'<p class="info-p" style="color:#f87171">The enrichment table is not available ({_html.escape(str(err))}); '
                        f'run <code>python generate_key_player_analysis.py</code> to create it.</p>')
    gray = "#64748b"
    views = [(s, CONDITION_NAME[s], f"{s}_vs_rest", CONDITION_NAME[s], "rest", CONDITION_COLOR[s], gray) for s in CONDITION_SLUGS]
    views.append(("healthy_vs_obesity", "Healthy vs Obesity", "healthy_vs_obesity", "Healthy", "Obesity/metabolic/diabetes",
                  CONDITION_COLOR["healthy"], CONDITION_COLOR["obesity"]))
    buttons, panes = "", ""
    for i, (key, label, sheet, name_a, name_b, col_a, col_b) in enumerate(views):
        if sheet not in xls.sheet_names:
            continue
        df = xls.parse(sheet)
        n_a, n_b = int(df["N_A"].iloc[0]), int(df["N_B"].iloc[0])
        first = not buttons
        buttons += (f'<button class="kp-btn{" active" if first else ""}" data-kp="{key}" '
                    f'onclick="showKpPlots(\'{key}\')">{_html.escape(label)}</button>')
        title = (f'<div class="kp-head"><strong style="color:{col_a}">{_html.escape(name_a)}</strong> '
                 f'vs {_html.escape(name_b)} <span class="kp-n">n = {n_a:,} vs {n_b:,} papers</span></div>')
        grid = "".join(_kp_panel(df, c, name_a, name_b, col_a, col_b) for c in ("hormones", "metabolites", "proteins"))
        panes += (f'<div class="kp-variant" id="kp-variant-{key}" style="display:{"block" if first else "none"}">'
                  f'{title}<div class="kp-grid">{grid}</div></div>')
    css = """
    <style>
      .kp-btns { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:14px; }
      .kp-btn { background:#1e293b; border:1px solid #334155; color:#94a3b8; border-radius:6px; padding:5px 12px; font-size:0.76rem; cursor:pointer; }
      .kp-btn:hover { background:#334155; color:#e2e8f0; }
      .kp-btn.active { background:#6366f1; border-color:#6366f1; color:#fff; }
      .kp-head { font-size:0.9rem; color:#e2e8f0; margin-bottom:10px; }
      .kp-n { color:#64748b; font-size:0.76rem; margin-left:8px; }
      .kp-grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:16px; align-items:start; }
      @media (max-width:1100px) { .kp-grid { grid-template-columns:minmax(0, 1fr); } }
      .kp-panel { background:#0f172a; border:1px solid #334155; border-radius:8px; padding:8px 12px 10px; min-width:0; }
      .kp-panel-title { font-size:0.82rem; font-weight:700; color:#e2e8f0; }
      .kp-panel-sub { font-size:0.68rem; color:#94a3b8; margin:2px 0 8px; }
      .kp-none { color:#64748b; font-size:0.8rem; }
      .kp-row { display:flex; align-items:center; height:15px; font-size:10.5px; line-height:1; }
      .kp-name { flex:0 0 42%; padding-right:8px; text-align:right; color:#e2e8f0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
      .kp-track { position:relative; flex:1; height:100%; }
      .kp-zero { position:absolute; top:0; bottom:0; width:1px; background:#475569; }
      .kp-bar { position:absolute; top:3px; bottom:3px; border-radius:1px; }
      .kp-val { position:absolute; top:50%; transform:translateY(-50%); color:#94a3b8; white-space:nowrap; font-size:9.5px; }
      .kp-axisrow { height:13px; margin-top:2px; }
      .kp-axis { border-top:1px solid #475569; }
      .kp-tick { position:absolute; top:1px; transform:translateX(-50%); color:#94a3b8; font-size:9px; }
      .kp-axis-label { text-align:center; color:#64748b; font-size:9.5px; margin-top:2px; }
    </style>
    <script>
    function showKpPlots(key) {
      document.querySelectorAll('.kp-variant').forEach(el => { el.style.display = (el.id === 'kp-variant-' + key) ? 'block' : 'none'; });
      document.querySelectorAll('.kp-btn').forEach(b => b.classList.toggle('active', b.dataset.kp === key));
    }
    </script>
    """
    return css + intro + f'<div class="kp-btns">{buttons}</div>' + panes


def _condition_label_rows(labels: dict) -> str:
    """Info-panel table rows: condition label | meaning | papers with a verified label."""
    from Literature_Search.llm_condition_filter import LABEL_DESCRIPTIONS
    counts = Counter(r.get("label") for r in labels.values() if r.get("status") == "ok")
    return "".join(
        f'<tr><td style="padding:3px 8px 3px 0;color:#94a3b8;white-space:nowrap">'
        f'<strong style="color:#e2e8f0">{name}</strong></td>'
        f'<td style="padding:3px 8px 3px 0;color:#94a3b8">{LABEL_DESCRIPTIONS[name]}</td>'
        f'<td style="padding:3px 0 3px 8px;color:#f8fafc;text-align:right;white-space:nowrap">'
        f'{counts.get(name, 0):,}</td></tr>'
        for name in CONDITION_LABEL_NAMES
    )


def _condition_filter_stats_html(labels: dict, n_unique_papers: int) -> str:
    """Info-panel paragraph with the outcome of the LLM condition filtering."""
    if not labels:
        return ('<p class="info-p" style="color:#94a3b8">No papers have been classified by the '
                'condition filter yet.</p>')
    st = Counter(r.get("status") for r in labels.values())
    n = len(labels)
    ok = st.get("ok", 0)
    halluc = st.get("hallucinated", 0) + st.get("too_short", 0)
    unusable = n - ok - halluc

    def pct(x: int) -> str:
        return f"{100 * x / n:.1f}%" if n else "—"

    return (
        '<p class="info-p"><strong>Outcome:</strong> '
        f'{n:,} of {n_unique_papers:,} unique papers classified; '
        f'<strong>{ok:,}</strong> ({pct(ok)}) carry a label with a verified evidence sentence, '
        f'<strong>{halluc:,}</strong> ({pct(halluc)}) were rejected as hallucinated (quoted sentence '
        f'not found in the title/abstract, or too short to verify), {unusable:,} ({pct(unusable)}) were unusable '
        f'(invalid or failed answer). Papers without a verified label appear only in the '
        f'All papers view.</p>'
    )


def _build_info_tabs(organs: list[str], results_by_pair: dict, allowed_pairs: set,
                      condition_data: dict[str, tuple | None] | None = None,
                      oo_data: dict | None = None) -> list[dict]:
    """
    Build info panel content dynamically from the current configuration.

    condition_data (optional): {slug: (organs, results_by_pair, pairs) | None for every
    condition view} -- when present, the Literature Statistics tab
    gets an All-papers + per-condition selector using this data for the conditions
    (see build_condition_results() / run_metabolic_lit_search.py
    --condition). A condition mapped to None (pipeline not yet run for it)
    is shown as unavailable rather than omitted, so the selector's presence
    doesn't silently depend on which steps happen to have been run.

    oo_data (optional): output of _compute_oo_data -- adds an "Edge
    Statistics" tab with OO coefficient distribution and elbow threshold.
    """
    condition_data = condition_data or {}

    # ── Compute paper-funnel stats (with duplicate accounting) ──────────────
    _n_raw      = sum(v.get("n_papers_found_raw", v.get("n_papers_found", 0))
                      for v in results_by_pair.values())
    _n_filtered = sum(v.get("n_papers_found", 0) for v in results_by_pair.values())
    _pairs_filtered = sum(1 for v in results_by_pair.values() if v.get("n_papers_found", 0) > 0)

    def _unique_pmids(rbp: dict) -> set:
        pmids = set()
        for v in rbp.values():
            for p in v.get("papers", []):
                pid = p.get("pmid")
                if pid:
                    pmids.add(pid)
        return pmids

    _pmids_filtered     = _unique_pmids(results_by_pair)
    _n_unique_filtered  = len(_pmids_filtered)
    _n_dup_filtered     = _n_filtered - _n_unique_filtered

    def _cond_stats(cond_name: str) -> tuple[int, int, int, int]:
        """Returns (total_papers, unique_papers, pairs_with_papers, dup_count)."""
        cd = condition_data.get(cond_name)
        if cd is None:
            return 0, 0, 0, 0
        _, cond_rbp, _ = cd
        total  = sum(v.get("n_papers_found", 0) for v in cond_rbp.values())
        unique = len(_unique_pmids(cond_rbp))
        pairs  = sum(1 for v in cond_rbp.values() if v.get("n_papers_found", 0) > 0)
        return total, unique, pairs, total - unique

    cond_stat = {slug: _cond_stats(slug) for slug in CONDITION_SLUGS}   # (total, unique, pairs, dup)

    def _pct(num: int, denom: int) -> str:
        return f"{100*num/denom:.0f}%" if denom else "—"
    def _dup_label(dup: int, total: int) -> str:
        return f"{dup:,} ({100*dup/total:.0f}%)" if total else "—"

    _pct_filt = _pct(_n_filtered, _n_raw)

    # unique papers of the PubMed step (count-only run of recount_step1_window.py; the paper lists of this step were not kept)
    try:
        _step1_unique = json.loads((Path(OUTPUT_JSON).parent / "pubmed_step1_final.json").read_text(encoding="utf-8")).get("window unique papers")
    except (OSError, ValueError):
        _step1_unique = None
    _step1_uniq_cell = (f'<td class="frow-uniq">{_step1_unique:,}</td>' if _step1_unique
                        else '<td class="frow-uniq" style="color:#64748b;font-style:italic">n/a</td>')

    _funnel_rows = f"""
        <tr>
          <td class="frow-step">PubMed fetch</td>
          <td class="frow-val">{_n_raw:,}</td>
          {_step1_uniq_cell}
          <td class="frow-pairs">{len(results_by_pair)}</td>
          <td class="frow-pct">—</td>
        </tr>
        <tr>
          <td class="frow-step">Same-sentence filter</td>
          <td class="frow-val">{_n_filtered:,}</td>
          <td class="frow-uniq">{_n_unique_filtered:,}</td>
          <td class="frow-pairs">{_pairs_filtered}</td>
          <td class="frow-pct">{_pct_filt}</td>
        </tr>""" + "".join(f"""
        <tr>
          <td class="frow-step">{CONDITION_NAME[s_]} view</td>
          <td class="frow-val">{cond_stat[s_][0]:,}</td>
          <td class="frow-uniq">{cond_stat[s_][1]:,}</td>
          <td class="frow-pairs">{cond_stat[s_][2]}</td>
          <td class="frow-pct">{_pct(cond_stat[s_][0], _n_filtered)}</td>
        </tr>""" for s_ in CONDITION_SLUGS)

    _dup_rows = f"""
            <tr>
              <td class="frow-step">All papers (same-sentence)</td>
              <td class="frow-val">{_n_filtered:,}</td>
              <td class="frow-uniq">{_n_unique_filtered:,}</td>
              <td class="frow-pct">{_dup_label(_n_dup_filtered, _n_filtered)}</td>
            </tr>""" + "".join(f"""
            <tr>
              <td class="frow-step">{CONDITION_NAME[s_]} view</td>
              <td class="frow-val">{cond_stat[s_][0]:,}</td>
              <td class="frow-uniq">{cond_stat[s_][1]:,}</td>
              <td class="frow-pct">{_dup_label(cond_stat[s_][3], cond_stat[s_][0])}</td>
            </tr>""" for s_ in CONDITION_SLUGS)

    kw_block = " OR ".join(
        f'"{kw}"' if (" " in kw or "-" in kw) else kw
        for kw in METABOLIC_KEYWORDS
    )
    crosstalk_kw_block = " OR ".join(
        f'"{kw}"' if (" " in kw or "-" in kw) else kw
        for kw in CROSSTALK_KEYWORDS
    )
    # only the labels of the papers that are in the network now (the label file also holds papers that were
    # removed later, e.g. by the 20-year window)
    _current_pmids = {str(p) for p in _pmids_filtered}
    cond_labels = {p: r for p, r in load_condition_labels(CONDITION_LABELS, CONDITION_LLM_MODEL).items()
                   if str(p) in _current_pmids}
    cond_label_rows = _condition_label_rows(cond_labels)
    cond_filter_stats = _condition_filter_stats_html(cond_labels, _n_unique_filtered)
    from Literature_Search.llm_connection_type import VOTE_RUNS as vote_runs, MAX_ABSTRACT_CHARS as _type_abstract_chars
    vote_majority = vote_runs // 2 + 1
    from Literature_Search.pubmed_search import HORMONES as _H, METABOLITES as _M, PROTEINS as _P
    # search date / publication window of the PubMed step and the size of the PET cohort networks
    try:
        _step1 = json.loads((Path(OUTPUT_JSON).parent / "pubmed_step1_final.json").read_text(encoding="utf-8"))
        _d = datetime.strptime(_step1["search date"], "%Y/%m/%d")
        _w0, _w1 = (datetime.strptime(x, "%Y-%m-%d") for x in _step1["window"].split(".."))
        _search_date = f"{_d.day} {_d:%B %Y}"
        _window_range = f"{_w0.day} {_w0:%b %Y} – {_w1.day} {_w1:%b %Y}"
    except (OSError, ValueError, KeyError):
        _search_date, _window_range = "n/a", f"in the last {YEARS_BACK} years"
    _n_h_edges = len(load_edge_filter(HEALTHY_FILTER_CSV)) if HEALTHY_FILTER_CSV.exists() else 0
    _n_o_edges = len(load_edge_filter(OBESE_FILTER_CSV)) if OBESE_FILTER_CSV.exists() else 0
    ct_rows = "".join(
        f'<tr><td style="padding:3px 8px 3px 0;color:#94a3b8;white-space:nowrap">'
        f'<strong style="color:#e2e8f0">{v["label"]}</strong></td>'
        f'<td style="padding:3px 0;color:#94a3b8">{v["description"]}</td></tr>'
        for v in CONNECTION_TYPES.values()
    )
    return [
        {
            "id": "search",
            "label": "Literature Search",
            "content": _info_text(
                "search",
                n_pairs=len(allowed_pairs), n_organs=len(organs),
                years_back=YEARS_BACK, max_papers="10,000",
                n_pairs_searched=len(results_by_pair),
                funnel_rows=_funnel_rows, duplicate_rows=_dup_rows,
                kw_block=kw_block, crosstalk_kw_block=crosstalk_kw_block,
                cond_llm_model=CONDITION_LLM_MODEL, cond_label_rows=cond_label_rows,
                cond_llm_backend=("the Claude API" if str(CONDITION_LLM_MODEL).startswith("claude-") else "Ollama"),
                cond_filter_stats=cond_filter_stats,
                search_date=_search_date, window_range=_window_range,
                n_hormones=f"{len(_H):,}", n_metabolites=f"{len(_M):,}", n_proteins=f"{len(_P):,}",
                n_h_edges=_n_h_edges, n_o_edges=_n_o_edges,
            ),
        },
        {
            "id": "literature_stats",
            "label": "Literature Statistics",
            "group": "stats",
            "content": _literature_stats_tab_content(organs, results_by_pair, allowed_pairs, condition_data),
        },
        {
            "id": "llm",
            "label": "Connection Types",
            "content": _info_text(
                "connection_types",
                llm_model=CONNECTION_TYPE_LLM_MODEL,
                llm_backend=("the Claude API" if str(CONNECTION_TYPE_LLM_MODEL).startswith("claude-") else "Ollama"),
                llm_max_papers=LLM_MAX_PAPERS, llm_abstract_chars=_type_abstract_chars,
                vote_runs=vote_runs, vote_majority=vote_majority,
                connection_type_rows=ct_rows,
            ),
        },
        {
            "id": "descriptions",
            "label": "Descriptions",
            "content": _info_text("descriptions", llm_model=DESCRIPTION_LLM_MODEL,
                                  **_description_text_values(DESCRIPTION_LLM_MODEL)),
        },
        {
            "id": "llm_prompts",
            "label": "LLM Prompts",
            "content": _info_text("llm_prompts_intro") + _llm_prompts_html(),
        },
        {
            "id": "network_analysis",
            "label": "Network Analysis",
            "group": "stats",
            "content": _network_analysis_tab_html(results_by_pair, condition_data, organs),
        },
    ] + ([
        {
            "id": "oo_threshold",
            "label": "Edge Statistics",
            "group": "stats",
            "content": _oo_threshold_tab_html(oo_data),
        },
    ] if oo_data else []) + [
        {
            "id": "key_player_enrichment",
            "label": "Key Player Enrichment",
            "group": "stats",
            "content": _key_player_enrichment_tab(),
        },
    ]


def _index_results_by_pair(results: dict) -> dict[tuple[str, str], dict]:
    by_pair: dict[tuple[str, str], dict] = {}
    for v in results.values():
        o1, o2 = v.get("organ1", ""), v.get("organ2", "")
        if o1 and o2:
            by_pair[(min(o1, o2), max(o1, o2))] = v
    return by_pair


def _compute_edge_display_data(o1: str, o2: str, edge_data: dict,
                                llm_types: dict, llm_descs: dict,
                                get_connection_types, CONNECTION_TYPES) -> dict:
    """
    Build the display-ready dict for one edge (key players, connection
    type, LLM description, papers) from a given set of literature/LLM
    sources. Reused for both the "all" baseline and each condition variant
    (healthy/obese) -- same shape, different source data.
    """
    papers  = edge_data.get("papers", [])
    _kp_raw = edge_data.get("key_players", {})

    def _remerge(raw_counts, synonyms):
        merged = _merge_synonyms(raw_counts, synonyms)
        ranked = [t for t, _ in sorted(merged.items(), key=lambda x: -x[1])]
        return ranked, merged

    _h_list, _h_counts = _remerge(_kp_raw.get("hormones_counts", {}), HORMONE_SYNONYMS)
    _m_list, _m_counts = _remerge(_kp_raw.get("metabolites_counts", {}), METABOLITE_SYNONYMS)
    _p_list, _p_counts = _remerge(_kp_raw.get("proteins_counts", {}), PROTEIN_SYNONYMS)

    type_labels = get_connection_types(llm_types, o1, o2, CONNECTION_TYPES)
    conn_type        = type_labels[0] if type_labels else ""
    conn_type_others = type_labels[1:]

    llm_entry = (
        llm_descs.get(f"{o1}|{o2}")
        or llm_descs.get(f"{o2}|{o1}")
        or {}
    )
    # When an LLM summary exists, its citation numbers [1]-[5] index into
    # its OWN selected_papers list, not the full evidence list -- "papers"
    # below must match whichever text is actually shown, or citation links
    # resolve to the wrong paper / nothing.
    llm_description = llm_entry.get("description", "")
    citation_papers = llm_entry.get("papers") if llm_description else None

    return {
        "pubmed_query":              edge_data.get("pubmed_query", ""),
        "n_papers_found":            edge_data.get("n_papers_found", 0),
        "papers":                    citation_papers if citation_papers else papers,
        "connection_type":        conn_type,
        "connection_type_others": conn_type_others,
        "search_date":               edge_data.get("search_date", ""),
        "ai_description":            llm_description,
        "key_players_hormones":           _h_list,
        "key_players_metabolites":        _m_list,
        "key_players_proteins":           _p_list,
        "key_players_counts_hormones":    _h_counts,
        "key_players_counts_metabolites": _m_counts,
        "key_players_counts_proteins":    _p_counts,
    }


def _compute_oo_data(results_by_pair: dict, allowed_pairs: set,
                      cond_pats: dict | None = None) -> dict:
    """
    Compute Otsuka-Ochiai coefficients and elbow threshold for the
    reference-metabolic network, for the "all" view and (optionally) each
    condition-filtered view.

    Per-organ paper count = union of PMIDs across every pair touching that
    organ (same derivation used for the Sankey diagram).

    Returns a dict keyed by view ("all" + every condition slug), each
    containing:
        pairs:     sorted list of (organ1, organ2, oo_value)
        threshold: float  (kneedle elbow on sorted OO values, or 0 if N<3)
        n_edges:   int    (pairs with oo_value >= threshold and > 0)
    """
    from threshold_utils import kneedle_elbow

    def _oo_for_papers(pairs_papers: dict[tuple, list]) -> dict:
        """pairs_papers: {(o1,o2): [paper, ...]}  →  result dict."""
        organ_pmids: dict[str, set] = {}
        for (o1, o2), papers in pairs_papers.items():
            pmids = {p.get("pmid") for p in papers if p.get("pmid")}
            organ_pmids.setdefault(o1, set()).update(pmids)
            organ_pmids.setdefault(o2, set()).update(pmids)
        organ_n = {o: len(s) for o, s in organ_pmids.items()}

        oo_list = []
        for (o1, o2), papers in pairs_papers.items():
            n_co = len([p for p in papers if p.get("pmid")])
            n1 = organ_n.get(o1, 0)
            n2 = organ_n.get(o2, 0)
            denom = (n1 * n2) ** 0.5 if n1 > 0 and n2 > 0 else 0
            oo = (n_co / denom) if denom > 0 else 0.0
            oo_list.append((o1, o2, oo))
        oo_list.sort(key=lambda t: -t[2])

        vals = [t[2] for t in oo_list]
        nonzero = sorted(v for v in vals if v > 0)
        thresh = kneedle_elbow(nonzero) if len(nonzero) >= 3 else 0.0
        n_edges = sum(1 for v in vals if v >= thresh and v > 0)
        return {"pairs": oo_list, "threshold": thresh, "n_edges": n_edges}

    # "all" condition: use already-filtered (same-sentence) papers
    all_pp = {pair: results_by_pair.get(pair, {}).get("papers", [])
              for pair in allowed_pairs}
    result = {"all": _oo_for_papers(all_pp)}

    if cond_pats:
        for cond, pats in cond_pats.items():
            cond_pp = {pair: _filter_by_condition(papers, pats)
                       for pair, papers in all_pp.items()}
            result[cond] = _oo_for_papers(cond_pp)

    return result


def _oo_threshold_tab_html(oo_data: dict) -> str:
    """
    Build the "Edge Statistics" info-panel tab: left column explains the OO
    coefficient and kneedle threshold, right column shows the OO distribution
    and elbow threshold for all conditions at once (compact, hi-DPI canvases).
    """
    import json as _json

    condition_labels = CONDITION_NAME
    colors = CONDITION_COLOR

    cards = ""
    chart_data = {}
    for cond in ["all"] + CONDITION_SLUGS:
        if cond not in oo_data:
            continue
        d = oo_data[cond]
        oo_list = d["pairs"]
        thresh  = d["threshold"]
        n_edges = d["n_edges"]
        n_total = len(oo_list)
        n_pos   = sum(1 for _, _, v in oo_list if v > 0)
        col     = colors[cond]
        chart_data[cond] = {
            "pairs": [{"label": f"{o1[:3]}–{o2[:3]}", "oo": round(oo, 5)}
                      for o1, o2, oo in oo_list],
            "threshold": round(thresh, 6),
            "color": col,
        }
        cards += f"""
      <div class="oo-card" style="border-left:3px solid {col}">
        <div class="oo-card-head">
          <span class="oo-card-title" style="color:{col}">{condition_labels[cond]}</span>
          <span class="oo-chip"><b>{n_total}</b> pairs</span>
          <span class="oo-chip"><b>{n_pos}</b> with &ge;1 paper</span>
          <span class="oo-chip"><b>{thresh:.4f}</b> elbow (OO)</span>
          <span class="oo-chip"><b>{n_edges}</b> edges &ge; elbow</span>
        </div>
        <div class="oo-chart-wrap"><canvas id="oo-chart-{cond}" class="oo-canvas"></canvas></div>
      </div>
"""

    chart_json = _json.dumps(chart_data)

    return f"""
    <style>
      .oo-split {{
        display: grid;
        grid-template-columns: minmax(260px, 5fr) minmax(320px, 7fr);
        gap: 28px;
        align-items: start;
      }}
      @media (max-width: 900px) {{ .oo-split {{ grid-template-columns: 1fr; }} }}
      .oo-right {{ display: flex; flex-direction: column; gap: 10px; min-width: 0; max-height: calc(100vh - 175px); overflow-y: auto; padding-right: 4px; }}
      .oo-card {{
        flex: 0 0 150px; min-height: 0; display: flex; flex-direction: column;
        background: #0f172a; border: 1px solid #334155; border-radius: 8px;
        padding: 8px 10px 6px;
      }}
      .oo-card-head {{
        display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px;
        margin-bottom: 4px;
      }}
      .oo-card-title {{ font-size: 13px; font-weight: 700; margin-right: 4px; }}
      .oo-chip {{ font-size: 11px; color: #94a3b8; white-space: nowrap; }}
      .oo-chip b {{ color: #e2e8f0; font-weight: 600; }}
      .oo-chart-wrap {{ flex: 1 1 0; min-height: 0; position: relative; }}
      .oo-canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; display: block; }}
    </style>
    <div class="oo-split">
      <div class="oo-left">
        {_info_text('edge_statistics_text')}
      </div>
      <div class="oo-right">
        {cards}
      </div>
    </div>
    <script>
    (function() {{
      const DATA = {chart_json};
      const sharedMax = Math.max(0.01, ...Object.values(DATA).map(
        d => Math.max(...d.pairs.map(p => p.oo), d.threshold * 1.1)));

      function draw(cond) {{
        const D = DATA[cond];
        const canvas = document.getElementById('oo-chart-' + cond);
        if (!canvas) return;
        const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
        if (!cssW || !cssH) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width  = Math.round(cssW * dpr);
        canvas.height = Math.round(cssH * dpr);
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cssW, cssH);

        const pad = {{l: 40, r: 8, t: 12, b: 20}};
        const plotW = cssW - pad.l - pad.r, plotH = cssH - pad.t - pad.b;
        const pairs = D.pairs, thresh = D.threshold;
        const slot = plotW / pairs.length;
        const bw = Math.max(1, slot * 0.78);

        ctx.font = '9px system-ui, sans-serif';
        ctx.textBaseline = 'middle';
        ctx.textAlign = 'right';
        ctx.lineWidth = 1;
        [0, 0.5, 1].forEach(f => {{
          const y = Math.round(pad.t + plotH - f * plotH) + 0.5;
          ctx.strokeStyle = '#1e293b';
          ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(cssW - pad.r, y); ctx.stroke();
          ctx.fillStyle = '#94a3b8';
          ctx.fillText((sharedMax * f).toFixed(2), pad.l - 4, y);
        }});

        pairs.forEach((p, i) => {{
          const bh = (p.oo / sharedMax) * plotH;
          ctx.fillStyle = (p.oo + 5e-6 >= thresh && p.oo > 0) ? D.color : '#475569';
          ctx.fillRect(pad.l + i * slot, pad.t + plotH - bh, bw, bh);
        }});

        ctx.strokeStyle = '#64748b';
        ctx.beginPath();
        ctx.moveTo(pad.l + 0.5, pad.t); ctx.lineTo(pad.l + 0.5, pad.t + plotH + 0.5);
        ctx.lineTo(cssW - pad.r, pad.t + plotH + 0.5); ctx.stroke();

        if (thresh > 0) {{
          const ty = pad.t + plotH - (thresh / sharedMax) * plotH;
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = '#f87171'; ctx.lineWidth = 1.2;
          ctx.beginPath(); ctx.moveTo(pad.l, ty); ctx.lineTo(cssW - pad.r, ty); ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = '#f87171'; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
          ctx.fillText('elbow ' + thresh.toFixed(4), cssW - pad.r - 2, ty - 2);
        }}

        ctx.fillStyle = '#94a3b8'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
        ctx.fillText('organ pairs sorted by OO coefficient (desc); y-axis = OO', pad.l, pad.t + plotH + 6);
      }}

      function drawAll() {{ Object.keys(DATA).forEach(draw); }}

      Object.keys(DATA).forEach(cond => {{
        const canvas = document.getElementById('oo-chart-' + cond);
        if (!canvas) return;
        canvas.addEventListener('mousemove', ev => {{
          const D = DATA[cond];
          const r = canvas.getBoundingClientRect();
          const slot = (r.width - 48) / D.pairs.length;
          const i = Math.floor((ev.clientX - r.left - 40) / slot);
          const p = D.pairs[i];
          canvas.title = p ? p.label + '  OO = ' + p.oo.toFixed(4) : '';
        }});
        if (window.ResizeObserver) new ResizeObserver(() => draw(cond)).observe(canvas);
      }});
      window.addEventListener('resize', drawAll);
      drawAll();
    }})();
    </script>
"""


def _cohort_pairs_to_csv(pairs: set, organs: set) -> str:
    """Render a set of canonical organ pairs as a 0/1 adjacency CSV."""
    organs_sorted = sorted(organs)
    lines = ["," + ",".join(organs_sorted)]
    for row in organs_sorted:
        cells = [row]
        for col in organs_sorted:
            if row == col:
                cells.append("0")
            elif (min(row, col), max(row, col)) in pairs:
                cells.append("1")
            else:
                cells.append("0")
        lines.append(",".join(cells))
    return "\n".join(lines)


def _inject_cohort_comparison(html_path: Path,
                               healthy_cohort: set, obese_cohort: set, lung_cohort: set,
                               all_organs: set) -> None:
    """
    Post-process the generated HTML to add one-click cohort comparison buttons.

    Embeds the healthy-, overweight- and lung-cancer-cohort adjacency matrices as JS constants so
    the user can trigger the reference-comparison coloring (green = SHARED,
    blue = ONLY_LIT, orange = ONLY_COHORT) with one click.  The buttons call
    window.loadExternalComparison (exposed by networkBuilderUtils).  The dashboard
    has no file-upload option; these three buttons are the only way to load a reference.
    """
    h_csv = _cohort_pairs_to_csv(healthy_cohort, all_organs).replace("`", "\\`")
    o_csv = _cohort_pairs_to_csv(obese_cohort,   all_organs).replace("`", "\\`")
    l_csv = _cohort_pairs_to_csv(lung_cohort,    all_organs).replace("`", "\\`")

    extra_js = f"""<script>
(function(){{
  var _HC = `{h_csv}`;
  var _OC = `{o_csv}`;
  var _LC = `{l_csv}`;
  // Use loadExternalComparison (exposed by networkBuilderUtils) to load the
  // built-in cohort matrix (there is no file-upload option in the dashboard).
  window.loadHealthyCohort    = function(){{ if (window.loadExternalComparison) window.loadExternalComparison(_HC, 'healthy_cohort'); }};
  window.loadObeseCohort      = function(){{ if (window.loadExternalComparison) window.loadExternalComparison(_OC, 'obese_cohort'); }};
  window.loadLungCancerCohort = function(){{ if (window.loadExternalComparison) window.loadExternalComparison(_LC, 'lung_cancer_cohort'); }};
}})();
</script>"""

    btn_h = (
        '<button id="healthy-cohort-btn" onclick="loadHealthyCohort()" class="ctrl-btn" '
        'title="Compare current Healthy view vs the predefined healthy-cohort edges">'
        '&#9878; Healthy cohort</button>'
    )
    btn_o = (
        '<button id="obese-cohort-btn" onclick="loadObeseCohort()" class="ctrl-btn" '
        'title="Compare current Overweight view vs the predefined overweight-cohort edges">'
        '&#9878; Overweight cohort</button>'
    )
    btn_l = (
        '<button id="lung-cancer-cohort-btn" onclick="loadLungCancerCohort()" class="ctrl-btn" '
        'title="Compare current view vs the predefined lung-cancer-cohort edges">'
        '&#9878; Lung cancer cohort</button>'
    )

    # "All papers" button — resets to the unfiltered view (clears the condition toggles)
    no_cond_js = """<script>
(function(){
  function _noCondBtn() {
    var btn = document.getElementById('topbar-no-condition-btn');
    if (typeof setDisplayMode === 'function') {
      setDisplayMode('all');
    }
    // Deactivate the condition toggle buttons visually
    document.querySelectorAll('.cond-toggle').forEach(function(el){ el.classList.remove('active'); });
    if (btn) btn.classList.add('active');
  }
  // Hook into the existing toggle buttons to deactivate ours when they activate
  function _patchToggleButtons() {
    document.querySelectorAll('.cond-toggle').forEach(function(el){
      if (el.__ncPatched) return;
      el.__ncPatched = true;
      el.addEventListener('click', function(){
        var ncBtn = document.getElementById('topbar-no-condition-btn');
        if (ncBtn) ncBtn.classList.remove('active');
      });
    });
  }
  document.addEventListener('DOMContentLoaded', function(){
    _patchToggleButtons();
    // Mark No Condition as active by default (page starts in 'all' mode)
    var btn = document.getElementById('topbar-no-condition-btn');
    if (btn) btn.classList.add('active');
  });
  window._noCondBtn = _noCondBtn;
})();
</script>"""

    # CSS patch: style the No Condition active state
    css_patch = """<style>
#topbar-no-condition-btn.active {
  background: #4f46e5 !important;
  border-color: #4f46e5 !important;
  color: #fff !important;
}
</style>"""

    no_cond_btn = (
        '<button id="topbar-no-condition-btn" class="ctrl-btn" onclick="_noCondBtn()" '
        'title="Show all papers (no condition filter)">All papers</button>'
    )

    html = html_path.read_text(encoding="utf-8")

    # 1. Inject the "All papers" button BEFORE the first condition toggle button.
    first_cond_anchor = 'id="cond-btn-'
    if first_cond_anchor in html:
        pos = html.index(first_cond_anchor)
        btn_start = html.rindex('<button', 0, pos)
        html = html[:btn_start] + no_cond_btn + html[btn_start:]

    # 2. Insert the three cohort comparison buttons after the (hidden) clear button (inside comparison-box).
    clear_anchor  = 'id="upload-comparison-clear-btn"'
    if clear_anchor in html:
        pos      = html.index(clear_anchor)
        btn_end  = html.index('</button>', pos) + len('</button>')
        html = html[:btn_end] + btn_h + btn_o + btn_l + html[btn_end:]
    else:
        html = html.replace('</div>\n  </div>\n</div>', btn_h + btn_o + btn_l + '\n  </div>\n  </div>\n</div>', 1)

    # 3. Inject JS and CSS before </body>
    html = html.replace('</body>',
                        css_patch + no_cond_js + extra_js + '\n</body>', 1)
    html_path.write_text(html, encoding="utf-8")


def build_viz(results: dict, output_path: Path):
    """Build the interactive HTML network visualization."""
    try:
        import networkx as nx
        from Visualisation.networkBuilderUtils import export_network_to_cytoscape_dashboard
        from Literature_Search.llm_connection_type import (
            load_connection_type_classifications, get_connection_types,
        )
        from Literature_Search.llm_descriptions import load_llm_descriptions
        from Literature_Search.organ_descriptions import load_organ_descriptions

        allowed_pairs  = set(load_edge_filter(EDGE_FILTER_CSV))
        allowed_organs = {o for pair in allowed_pairs for o in pair}

        # Cohort CSVs (PET partial-correlation cohorts, BMI <= 25 / > 25) are used ONLY for
        # the optional comparison overlay buttons -- they do not define the condition views
        # (those come from the LLM labels).
        healthy_cohort = set(load_edge_filter(HEALTHY_FILTER_CSV)) if HEALTHY_FILTER_CSV.exists() else set()
        obese_cohort   = set(load_edge_filter(OBESE_FILTER_CSV))   if OBESE_FILTER_CSV.exists() else set()
        # FDG-PET lung cancer cohort (one group, SUV mean / aorta, 11 organs): compute_lung_cancer_PC_network.py
        lung_cohort    = set(load_edge_filter(LUNG_FILTER_CSV))    if LUNG_FILTER_CSV.exists() else set()

        llm_types   = load_connection_type_classifications(OUTPUT_LLM_TYPES)
        llm_descs   = load_llm_descriptions(LLM_DESCRIPTIONS)
        organ_descs = load_organ_descriptions(ORGAN_DESCRIPTIONS)

        results_by_pair = _index_results_by_pair(results)

        # LLM condition filters (one per condition view, all treated alike). Each accepts a
        # paper only if the LLM labelled it with that condition's label AND its quoted
        # evidence sentence was verified in the title/abstract.
        condition_labels = load_condition_labels(CONDITION_LABELS, CONDITION_LLM_MODEL)
        if not condition_labels:
            print("[!] No LLM condition labels found -- the condition views will be "
                  "empty. Run: python run_metabolic_lit_search.py --classify-conditions")
        else:
            n_ok = sum(1 for r in condition_labels.values() if r.get("status") == "ok")
            print(f"[i] LLM condition labels: {len(condition_labels)} papers classified, "
                  f"{n_ok} with verified evidence.")
        cond_pats = {slug: LLMConditionFilter(slug, condition_labels) for slug in CONDITION_SLUGS}

        # Condition-specific LLM data (built by --condition runs, one set per view). A view
        # whose files do not exist yet simply shows no connection type / summary
        # (the edge panel then says "No valid description could be generated").
        cond_llm: dict[str, tuple] = {}
        for cond in CONDITION_SLUGS:
            paths = condition_output_paths(cond)
            c_types = (load_connection_type_classifications(paths["llm_types"])
                       if paths["llm_types"].exists() else {})
            c_descs = (load_llm_descriptions(paths["llm_descs"])
                       if paths["llm_descs"].exists() else {})
            cond_llm[cond] = (c_types, c_descs)

        # Which pairs/organs belong to each condition view: a pair is in a view if at least
        # one of its already-fetched papers was assigned to that condition by the LLM
        # filter (valid label + verified evidence). Evaluated on the fly for ALL pairs.
        lit_pairs: dict[str, set] = {slug: set() for slug in CONDITION_SLUGS}
        for pair in allowed_pairs:
            papers = results_by_pair.get(pair, {}).get("papers", [])
            for slug in CONDITION_SLUGS:
                if _filter_by_condition(papers, cond_pats[slug]):
                    lit_pairs[slug].add(pair)
        lit_organs = {slug: {o for p in prs for o in p} for slug, prs in lit_pairs.items()}

        G = nx.Graph()
        for organ in sorted(allowed_organs):
            organ_entry = organ_descs.get(organ, {})
            G.add_node(organ,
                       llm_description=organ_entry.get("description", ""),
                       llm_papers=organ_entry.get("papers", []),
                       conds=[slug for slug in CONDITION_SLUGS if organ in lit_organs[slug]])

        for (o1, o2) in allowed_pairs:
            G.add_edge(o1, o2)
            G.edges[o1, o2]['conds'] = [slug for slug in CONDITION_SLUGS if (o1, o2) in lit_pairs[slug]]

            edge_data = results_by_pair.get((o1, o2), {})
            G.edges[o1, o2]['merged_data'] = _compute_edge_display_data(
                o1, o2, edge_data, llm_types, llm_descs,
                get_connection_types, CONNECTION_TYPES,
            )
            G.edges[o1, o2]['color'] = "#64748b"

            # Condition variants: key players, paper counts and (when generated) the
            # condition-specific connection type and summary, for every view in which
            # this pair has at least one verified paper.
            variants: dict = {}
            for cond in CONDITION_SLUGS:
                cond_papers = _filter_by_condition(
                    edge_data.get("papers", []), cond_pats[cond]
                )
                if not cond_papers:
                    continue
                cond_kp = extract_key_players(cond_papers)
                cond_edge_data = {
                    **edge_data,
                    "papers":         cond_papers,
                    "n_papers_found": len(cond_papers),
                    "key_players":    cond_kp,
                }
                c_types, c_descs = cond_llm[cond]
                variants[cond] = _compute_edge_display_data(
                    o1, o2, cond_edge_data, c_types, c_descs,
                    get_connection_types, CONNECTION_TYPES,
                )
            if variants:
                G.edges[o1, o2]['condition_variants'] = variants

        # Condition data for the Literature Statistics / Network Analysis tabs
        # (computed on the fly from the base results and the LLM labels).
        info_condition_data: dict = {}
        for cond in CONDITION_SLUGS:
            cond_lit_pairs = lit_pairs[cond]
            cond_rbp: dict = {}
            for pair in cond_lit_pairs:
                base   = results_by_pair.get(pair, {})
                papers = _filter_by_condition(base.get("papers", []), cond_pats[cond])
                cond_rbp[pair] = {**base, "papers": papers, "n_papers_found": len(papers)}
            cond_organs = sorted({o for p in cond_lit_pairs for o in p})
            info_condition_data[cond] = (cond_organs, cond_rbp, cond_lit_pairs)

        # Compute OO coefficient distributions for the Edge Statistics tab
        oo_data = _compute_oo_data(results_by_pair, allowed_pairs, cond_pats)

        # Store per-edge OO coefficients as graph edge attributes so
        # networkBuilderUtils can scale edge widths by evidence strength.
        _oo_lookup: dict[tuple, dict] = {}
        for cond, cdata in oo_data.items():
            for (o1, o2, oo_val) in cdata.get("pairs", []):
                key = (o1, o2) if G.has_edge(o1, o2) else (o2, o1)
                _oo_lookup.setdefault(key, {})[cond] = oo_val
        for pair in allowed_pairs:
            o1, o2 = pair
            if G.has_edge(o1, o2):
                vals = _oo_lookup.get((o1, o2), _oo_lookup.get((o2, o1), {}))
                G.edges[o1, o2]['oo_coeff']   = vals.get("all", 0.0)
                G.edges[o1, o2]['oo_by_cond'] = {slug: vals.get(slug, 0.0) for slug in CONDITION_SLUGS}

        oo_thresholds = {
            cond: oo_data[cond]["threshold"]
            for cond in ["all"] + CONDITION_SLUGS
            if cond in oo_data
        }

        export_network_to_cytoscape_dashboard(
            graph=G,
            filename=str(output_path),
            include_legend=False,
            title=VIZ_TITLE,
            info_panel_tabs=_build_info_tabs(sorted(allowed_organs), results_by_pair, allowed_pairs,
                                             condition_data=info_condition_data,
                                             oo_data=oo_data),
            condition_toggles=[{"slug": c["slug"], "label": c["name"], "title": c["label"]}
                               for c in CONDITIONS],
            comparison_upload_label="Reference comparison",   # only switches the comparison machinery on; no upload button
            oo_thresholds=oo_thresholds,
            paper_search=paper_search_payload(results_by_pair, condition_labels),
        )

        # Inject one-click PET cohort overlay buttons (⚖ Healthy / Overweight / Lung cancer cohort):
        # they can be combined with ANY condition view to compare it with the PET network.
        _inject_cohort_comparison(output_path, healthy_cohort, obese_cohort, lung_cohort, allowed_organs)

        # UI refinements (compact layout, reference-network mode with blue/red signed
        # partial-correlation edges, r badge) and the weighted PC cohort matrices:
        # see Visualisation/ui_patches.py
        from Visualisation.ui_patches import (
            apply_ui_patches, inject_weighted_cohort_matrices, patch_cohort_reference_view,
        )
        apply_ui_patches(output_path)
        inject_weighted_cohort_matrices(output_path, OUTPUT_DIR)
        patch_cohort_reference_view(output_path)

        print(f"[ok] Visualization saved: {output_path}")
    except Exception as e:
        import traceback
        print(f"[!] Visualization failed: {e}")
        traceback.print_exc()


# ── Condition pipeline (--condition <view> | every) ─────────────────────────────

def run_condition_pipeline(condition: str, args) -> None:
    """
    Build a condition-filtered results file from the already-cached "all"
    search results (no new PubMed calls), then run LLM connection-type
    classification and LLM connection descriptions against it.

    Entirely separate from the normal flow below: does not touch
    OUTPUT_JSON/OUTPUT_LLM_TYPES/LLM_DESCRIPTIONS and does not rebuild the
    dashboard -- the "all" case stays exactly as it is today.
    """
    if not OUTPUT_JSON.exists():
        print(f"[!] No base results file found at {OUTPUT_JSON}.")
        print(f"    Run without --condition first to build the base 'all' search.")
        sys.exit(1)
    with open(OUTPUT_JSON, encoding="utf-8") as f:
        base_results = json.load(f)
    print(f"[i] Loaded {len(base_results)} cached 'all' edges.")

    # The LLM filter judges the papers of ALL 91 pairs -- the cohort CSVs no longer
    # restrict which pairs a case is built for, so every pair can contribute.
    cond_pairs = load_edge_filter(EDGE_FILTER_CSV)
    labels = load_condition_labels(CONDITION_LABELS, CONDITION_LLM_MODEL)
    if not labels:
        print(f"[!] No LLM condition labels found at {CONDITION_LABELS}.")
        print("    Run first:  python run_metabolic_lit_search.py --classify-conditions")
        sys.exit(1)
    n_unique = len({str(p.get("pmid")) for e in base_results.values() for p in e.get("papers", [])})
    print(f"[i] Condition: {condition} -> LLM label '{CONDITION_TO_LABEL[condition]}' "
          f"| {len(cond_pairs)} pairs")
    print(f"[i] LLM labels available for {len(labels)} of {n_unique} unique papers "
          f"(papers without a verified label are not used)\n")

    out_paths = condition_output_paths(condition)

    print("[i] Building condition-filtered results (filtering already-fetched papers, "
          "no new PubMed queries)...")
    cond_results = build_condition_results(condition, base_results, cond_pairs, labels)
    with open(out_paths["results"], "w", encoding="utf-8") as f:
        json.dump(cond_results, f, indent=2, ensure_ascii=False)
    total_papers = sum(v.get("n_papers_found", 0) for v in cond_results.values())
    print(f"\n[ok] {len(cond_results)} edges | {total_papers} papers after condition filter.")
    print(f"     Saved: {out_paths['results']}")

    if not args.skip_llm_type:
        print(f"\n[i] Running LLM connection-type classification ({condition})...")
        from Literature_Search.llm_connection_type import generate_connection_type_classifications
        generate_connection_type_classifications(
            organ_pairs         = cond_pairs,
            literature_results  = cond_results,
            connection_types    = CONNECTION_TYPES,
            output_path         = out_paths["llm_types"],
            model               = CONNECTION_TYPE_LLM_MODEL,
            max_papers          = LLM_MAX_PAPERS,
            resume              = not args.reset_llm_type,
            reset               = args.reset_llm_type,
        )
    else:
        print("[i] Skipping LLM type classification (using cache).")

    print(f"\n[i] Running LLM connection descriptions ({condition})...")
    from Literature_Search.llm_descriptions import generate_llm_descriptions
    generate_llm_descriptions(
        organ_pairs         = cond_pairs,
        literature_results  = cond_results,
        output_path         = out_paths["llm_descs"],
        model               = DESCRIPTION_LLM_MODEL,
        resume              = not args.reset,
        reset               = args.reset,
    )

    print(f"\n[ok] Condition pipeline done for '{condition}'.")
    print("     LLM types/descriptions are now saved for all pairs with papers in this case.")
    print("     Run --viz-only to rebuild the dashboard with this condition's LLM data.")


# ── LLM condition classification (--classify-conditions) ───────────────────────

def run_condition_classification(args) -> None:
    """
    Label every paper of all 91 pairs with the LLM condition filter (one label
    per paper + the evidence sentence the LLM quoted, verified by regex).

    Resumable: results are appended to CONDITION_LABELS after every paper, so the
    run can be stopped (Ctrl+C) and continued any time. Papers of the healthy /
    obese cohort-network pairs are classified first, then the other pairs from the
    smallest to the largest.
    """
    if not OUTPUT_JSON.exists():
        print(f"[!] No base results file found at {OUTPUT_JSON}.")
        sys.exit(1)
    with open(OUTPUT_JSON, encoding="utf-8") as f:
        results = json.load(f)
    priority: set[str] = set()
    for csv_path in (HEALTHY_FILTER_CSV, OBESE_FILTER_CSV):
        if csv_path.exists():
            priority |= {f"{a}|{b}" for a, b in load_edge_filter(csv_path)}
    print(f"[i] LLM condition filter | model {CONDITION_LLM_MODEL} | {len(results)} pairs | "
          f"{len(priority)} cohort-network pairs first")
    classify_condition_labels(results, CONDITION_LABELS, CONDITION_LLM_MODEL,
                              limit=args.classify_limit, retry_errors=args.retry_errors,
                              priority_pairs=priority, workers=args.workers,
                              max_cost=args.max_cost)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset",          action="store_true",
                        help="Delete search cache and start from scratch.")
    parser.add_argument("--force-empty",    action="store_true",
                        help="Re-search edges that previously returned 0 papers.")
    parser.add_argument("--skip-llm-type",  action="store_true",
                        help="Skip LLM connection-type classification; use cached results.")
    parser.add_argument("--reset-llm-type", action="store_true",
                        help="Delete LLM type cache and reclassify all pairs.")
    parser.add_argument("--viz-only",       action="store_true",
                        help="Skip search and LLM; just rebuild the visualization.")
    parser.add_argument("--condition", choices=CONDITION_SLUGS + ["every"],
                        help="Build a condition-filtered results file from the already-cached "
                             "'all' search (no new PubMed queries) plus its own LLM "
                             "connection-type classification and descriptions. Separate "
                             "pipeline -- does not touch the 'all' case or the dashboard.")
    parser.add_argument("--classify-conditions", action="store_true",
                        help="Label every paper of all 91 pairs with the LLM condition filter "
                             "(resumable; this is the slow step, ~16 s per paper on CPU).")
    parser.add_argument("--classify-limit", type=int, default=None,
                        help="With --classify-conditions: classify only N random unclassified papers.")
    parser.add_argument("--retry-errors", action="store_true",
                        help="With --classify-conditions: redo papers whose request failed.")
    parser.add_argument("--workers", type=int, default=None,
                        help="With --classify-conditions: parallel requests (default 1 for Ollama, "
                             "8 for a Claude API model).")
    parser.add_argument("--max-cost", type=float, default=None,
                        help="With --classify-conditions and a Claude API model: stop when the "
                             "estimated spend of this run reaches this many USD.")
    args = parser.parse_args()

    if args.classify_conditions:
        run_condition_classification(args)
        return

    if args.condition:
        for slug in (CONDITION_SLUGS if args.condition == "every" else [args.condition]):
            run_condition_pipeline(slug, args)
        return

    if args.reset and OUTPUT_JSON.exists():
        OUTPUT_JSON.unlink()
        print(f"[i] Search cache deleted: {OUTPUT_JSON}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EDGE_FILTER_CSV.exists():
        print(f"[!] Edge filter not found: {EDGE_FILTER_CSV}")
        sys.exit(1)
    pairs = load_edge_filter(EDGE_FILTER_CSV)
    print(f"[i] Edge filter: {EDGE_FILTER_CSV.name} → {len(pairs)} pairs selected")
    print("[i] PubMed: at most 10,000 hits per query (esearch limit) | all years searched; the 20-year window is applied afterwards "
          "by apply_date_window.py\n")

    # ── Step 1: PubMed search ─────────────────────────────────────────────────
    if not args.viz_only:
        results = run_search(pairs, OUTPUT_JSON,
                             resume=not args.reset,
                             force_empty=args.force_empty)
    else:
        if not OUTPUT_JSON.exists():
            print(f"[!] No results file found at {OUTPUT_JSON}. Run without --viz-only first.")
            sys.exit(1)
        with open(OUTPUT_JSON, encoding="utf-8") as f:
            results = json.load(f)
        print(f"[i] Loaded {len(results)} cached edges.")

    # ── Step 2: LLM connection-type classification ────────────────────────────
    if not args.viz_only and not args.skip_llm_type:
        print("\n[i] Running LLM connection-type classification...")
        from Literature_Search.llm_connection_type import generate_connection_type_classifications
        generate_connection_type_classifications(
            organ_pairs       = pairs,
            literature_results = results,
            connection_types  = CONNECTION_TYPES,
            output_path       = OUTPUT_LLM_TYPES,
            model             = CONNECTION_TYPE_LLM_MODEL,
            max_papers        = LLM_MAX_PAPERS,
            resume            = not args.reset_llm_type,
            reset             = args.reset_llm_type,
        )
    elif args.skip_llm_type or args.viz_only:
        print("[i] Skipping LLM type classification (using cache).")

    # ── Step 3: Visualization ─────────────────────────────────────────────────
    print("\n[i] Building visualization...")
    build_viz(results, OUTPUT_HTML)


if __name__ == "__main__":
    main()
