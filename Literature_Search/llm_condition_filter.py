"""
LLM-based condition filter.

Replaces the keyword condition filter of the original pipeline. A local Ollama
model (or, if the model name starts with "claude-", a Claude API model such as
claude-haiku-5-5) plays a physician / molecular biologist: for every paper it reads the
title + abstract and assigns ONE primary condition of the studied subjects:

    No condition | Healthy | Obesity/metabolic syndrome/diabetes |
    Cardiovascular disease | Cancers | Infectious diseases | Other diseases
    (psychiatric and genetic disorders belong to 'Other diseases')

Together with the label the model must return the sentence of the title or
abstract that led to its conclusion ("evidence"). A separate, purely
regex-based test then checks that this sentence really occurs in the paper's
title or abstract. If it does not, the answer is treated as a hallucination
and the paper is NOT used for any condition view (it stays only in the unfiltered
"All papers" view).

Views:   every label is a dashboard view (CONDITIONS below), all treated identically;
         the unfiltered "All papers" view uses every paper.

Results are appended to a JSONL cache (one line per paper, keyed by PMID), so
the run can be interrupted and resumed. A record is only reused if it was made
with the same model and PROMPT_VERSION.

CLI (run from the Metabolic_Reference_Atlas/ folder):
    python Literature_Search/llm_condition_filter.py                 # all papers, resume
    python Literature_Search/llm_condition_filter.py --limit 40      # random test sample
    python Literature_Search/llm_condition_filter.py --retry-errors  # redo transient failures
"""

import argparse
import html
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests as _requests

# ── Settings ─────────────────────────────────────────────────────────────────

OLLAMA_URL     = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = 300
OLLAMA_RETRIES = 3
NUM_CTX        = 4096
NUM_PREDICT    = 220
MAX_ABSTRACT_CHARS = 6000
MIN_EVIDENCE_WORDS = 4          # shorter "evidence" cannot be verified meaningfully
PROMPT_VERSION     = "v6"

# Claude API backend: any model name starting with "claude-" is sent to the Anthropic API
# instead of Ollama (needs ANTHROPIC_API_KEY and credit in the Anthropic Console).
ANTHROPIC_MAX_TOKENS = 400         # the answer is one sentence + a label (~60 tokens)
ANTHROPIC_EFFORT     = "low"       # no reasoning needed for a one-step classification
ANTHROPIC_THINKING   = False       # True = let the model think first (slower, more output tokens)
ANTHROPIC_TIMEOUT    = 90          # seconds per request (the SDK retries 429/5xx itself)
ANTHROPIC_RETRIES    = 6
ANTHROPIC_WORKERS    = 8           # parallel requests when --workers is not given
API_PRICES = {                     # USD per million tokens (input, output), Anthropic list prices
    "claude-haiku-5-5":  (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5":   (4.00, 20.00),
}
MAX_CONSECUTIVE_ERRORS = 10        # a run stops after this many failed requests in a row

LABEL_NO_CONDITION = "No condition"
LABEL_HEALTHY      = "Healthy"
LABEL_METABOLIC    = "Obesity/metabolic syndrome/diabetes"
LABEL_CARDIO       = "Cardiovascular disease"
LABEL_CANCER       = "Cancers"
LABEL_INFECTIOUS   = "Infectious diseases"
LABEL_OTHER        = "Other diseases"
LABELS = [LABEL_NO_CONDITION, LABEL_HEALTHY, LABEL_METABOLIC, LABEL_CARDIO, LABEL_CANCER,
          LABEL_INFECTIOUS, LABEL_OTHER]

# short definitions for the dashboard info panel / statistics (the full ones are in SYSTEM_PROMPT)
LABEL_DESCRIPTIONS = {
    LABEL_NO_CONDITION: "The paper does not study subjects with a defined health condition (for example a review "
                        "of general physiology, a method, in-vitro work, or no condition is stated).",
    LABEL_HEALTHY: "The studied subjects are healthy people or normal animals. Healthy subjects that only serve "
                   "as the control group of a disease do not count.",
    LABEL_METABOLIC: "Overweight, obesity, metabolic syndrome, diabetes.",
    LABEL_CARDIO: "Diseases of the heart and blood vessels.",
    LABEL_CANCER: "Cancer and tumours.",
    LABEL_INFECTIOUS: "Infections by bacteria, viruses, fungi or parasites.",
    LABEL_OTHER: "Any other disease or pathology, including psychiatric and psychological disorders "
                 "and inherited or congenital (genetic) disorders.",
}

# Condition views of the dashboard. Every LLM label is a view, all treated identically:
# the papers carrying that label (valid answer + verified evidence sentence). The
# unfiltered "All papers" view is separate (ALL_VIEW). "No condition" here is the LLM
# label (papers without a defined health condition), not the unfiltered view.
CONDITIONS = [
    {"slug": "nocondition",    "label": LABEL_NO_CONDITION, "name": "No condition",               "color": "#94a3b8"},
    {"slug": "healthy",        "label": LABEL_HEALTHY,      "name": "Healthy",                    "color": "#34d399"},
    {"slug": "obesity",        "label": LABEL_METABOLIC,    "name": "Obesity/Metabolic/Diabetes", "color": "#fb923c"},
    {"slug": "cardiovascular", "label": LABEL_CARDIO,       "name": "Cardiovascular",             "color": "#f87171"},
    {"slug": "cancers",        "label": LABEL_CANCER,       "name": "Cancers",                    "color": "#c084fc"},
    {"slug": "infectious",     "label": LABEL_INFECTIOUS,   "name": "Infectious",                 "color": "#facc15"},
    {"slug": "other",          "label": LABEL_OTHER,        "name": "Other diseases",             "color": "#60a5fa"},
]
ALL_VIEW = {"slug": "all", "label": None, "name": "All papers", "color": "#a5b4fc"}
CONDITION_SLUGS = [c["slug"] for c in CONDITIONS]
CONDITION_TO_LABEL = {c["slug"]: c["label"] for c in CONDITIONS}      # view slug -> LLM label
CONDITION_NAME = {c["slug"]: c["name"] for c in [ALL_VIEW] + CONDITIONS}
CONDITION_COLOR = {c["slug"]: c["color"] for c in [ALL_VIEW] + CONDITIONS}

# statuses of one classification
OK            = "ok"             # parsed, label valid, evidence found in title/abstract
HALLUCINATED  = "hallucinated"   # evidence sentence NOT found in title/abstract
TOO_SHORT     = "too_short"      # evidence missing or shorter than MIN_EVIDENCE_WORDS
INVALID_JSON  = "invalid_json"
INVALID_LABEL = "invalid_label"
ERROR         = "error"          # request failed (timeout, connection, ...)
STATUSES = [OK, HALLUCINATED, TOO_SHORT, INVALID_JSON, INVALID_LABEL, ERROR]

_LABEL_ALIASES = {lab.lower(): lab for lab in LABELS}
_LABEL_ALIASES.update({
    "none": LABEL_NO_CONDITION, "no conditions": LABEL_NO_CONDITION,
    "obesity": LABEL_METABOLIC, "overweight": LABEL_METABOLIC, "diabetes": LABEL_METABOLIC,
    "metabolic syndrome": LABEL_METABOLIC, "obesity/metabolic syndrome": LABEL_METABOLIC,
    "obesity/metabolic syndrome/diabetes mellitus": LABEL_METABOLIC,
    "overweight/obesity/metabolic syndrome": LABEL_METABOLIC,
    "cardiovascular": LABEL_CARDIO, "cardiovascular diseases": LABEL_CARDIO,
    "cancer": LABEL_CANCER, "cancers": LABEL_CANCER,
    "infectious disease": LABEL_INFECTIOUS, "infection": LABEL_INFECTIOUS,
    # the two categories merged into "Other diseases": accept them if the model still names them
    "genetic diseases": LABEL_OTHER, "genetic disease": LABEL_OTHER, "genetic": LABEL_OTHER,
    "mental health": LABEL_OTHER, "mental": LABEL_OTHER, "mental health disorders": LABEL_OTHER,
    "other disease": LABEL_OTHER, "other": LABEL_OTHER,
})

# ── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a physician and molecular biologist with deep clinical and experimental expertise. You are very good at distinguishing diseases from each other and at telling healthy physiology from disease.

Read the TITLE and ABSTRACT of a biomedical paper and assign the paper to exactly ONE of the following categories, according to the health condition of the subjects (people, animals) the paper studies:

"No condition" - the paper does not study subjects with a defined health condition (for example a review of general physiology, a method, in-vitro work, or no condition is stated).
"Healthy" - the studied subjects are healthy people or normal animals. Healthy subjects that only serve as the control group of a disease do not count.
"Obesity/metabolic syndrome/diabetes" - overweight, obesity, metabolic syndrome, diabetes.
"Cardiovascular disease" - diseases of the heart and blood vessels.
"Cancers" - cancer and tumours.
"Infectious diseases" - infections by bacteria, viruses, fungi or parasites.
"Other diseases" - any other disease or pathology, including psychiatric and psychological disorders and inherited or congenital (genetic) disorders.

Also return "evidence": ONE sentence copied word for word from the title or abstract that led to your decision. Never write your own words in "evidence".

Reply with ONLY a JSON object, nothing else:
{"evidence": "<one sentence copied verbatim>", "condition": "<exactly one category name>"}"""

# (title, abstract, answer) -- shown to the model as earlier chat turns: one generic
# example per category, in mixed order.
FEW_SHOT = [
    ("Lactate shuttling between tumour cells and stroma drives growth.",
     "Xenograft tumours were grown in nude mice. Lactate export from tumour cells supported stromal metabolism.",
     {"evidence": "Xenograft tumours were grown in nude mice.", "condition": LABEL_CANCER}),
    ("Skeletal muscle glucose uptake during exercise in healthy young men.",
     "Ten healthy young men performed one-legged knee extension exercise. Muscle glucose uptake was measured by PET.",
     {"evidence": "Ten healthy young men performed one-legged knee extension exercise.", "condition": LABEL_HEALTHY}),
    ("Inter-organ crosstalk in energy metabolism: a review.",
     "This review summarises how organs exchange metabolites and hormones to coordinate energy homeostasis.",
     {"evidence": "This review summarises how organs exchange metabolites and hormones to coordinate energy homeostasis.",
      "condition": LABEL_NO_CONDITION}),
    ("Myocardial energy metabolism in patients with heart failure.",
     "Thirty patients with chronic heart failure underwent cardiac PET imaging. Myocardial glucose uptake was increased.",
     {"evidence": "Thirty patients with chronic heart failure underwent cardiac PET imaging.",
      "condition": LABEL_CARDIO}),
    ("High-fat diet-induced obese mice show impaired brown fat thermogenesis.",
     "Mice were fed a high-fat diet for 12 weeks to induce obesity. Thermogenic gene expression in brown fat was reduced.",
     {"evidence": "Mice were fed a high-fat diet for 12 weeks to induce obesity.", "condition": LABEL_METABOLIC}),
    ("Hepatic metabolism during bacterial sepsis in mice.",
     "Mice were infected with Escherichia coli to induce sepsis. Hepatic gluconeogenesis was suppressed during infection.",
     {"evidence": "Mice were infected with Escherichia coli to induce sepsis.", "condition": LABEL_INFECTIOUS}),
    ("Gut microbiota in ulcerative colitis: a review.",
     "Ulcerative colitis is a chronic inflammatory bowel disease. This review discusses how microbial metabolites shape intestinal inflammation.",
     {"evidence": "This review discusses how microbial metabolites shape intestinal inflammation.",
      "condition": LABEL_OTHER}),
]


def _paper_message(title: str, abstract: str) -> str:
    abstract = (abstract or "").strip()
    if len(abstract) > MAX_ABSTRACT_CHARS:
        abstract = abstract[:MAX_ABSTRACT_CHARS].rsplit(" ", 1)[0]
    return f"TITLE: {(title or '').strip()}\nABSTRACT: {abstract or '(not available)'}"


def build_messages(title: str, abstract: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t, a, ans in FEW_SHOT:
        msgs.append({"role": "user", "content": _paper_message(t, a)})
        msgs.append({"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)})
    msgs.append({"role": "user", "content": _paper_message(title, abstract)})
    return msgs


# ── Evidence test (regex) ────────────────────────────────────────────────────

_ELLIPSIS_RE = re.compile(r"\[\s*\.\.\.\s*\]|\.{3,}|…")


def _clean_text(s: str) -> str:
    """Unescape HTML entities and fold unicode so quotes/dashes/greek compare equal."""
    s = html.unescape(s or "")
    return unicodedata.normalize("NFKC", s)


_PROMPT_LABEL_RE = re.compile(r"^\s*(?:TITLE|ABSTRACT)\s*:\s*", re.IGNORECASE)


def strip_prompt_label(evidence: str) -> str:
    """The prompt marks the two parts of a paper with "TITLE:" / "ABSTRACT:"; a model that copies
    the title sometimes keeps that label. It is our own markup, not paper text."""
    return _PROMPT_LABEL_RE.sub("", evidence or "", count=1).strip()


def verify_evidence(evidence: str, title: str, abstract: str) -> tuple[bool, str]:
    """
    Regex test that the evidence sentence really occurs in the title/abstract.

    The sentence is reduced to its word tokens; the test passes if those tokens
    occur in the same order, separated ONLY by whitespace/punctuation, in
    "<title>. <abstract>" (case-insensitive). So harmless formatting changes
    (quote style, dash vs space, dropped comma) are tolerated, but a sentence
    with any changed, added or missing word is a hallucination. An ellipsis in
    the evidence ("A ... B") means "skip text here": every fragment must occur,
    in order. Returns (ok, reason).
    """
    ev = _clean_text(strip_prompt_label(evidence)).strip()
    if not ev:
        return False, "empty evidence"
    fragments = [f for f in _ELLIPSIS_RE.split(ev) if re.search(r"\w", f)]
    tokens_per_fragment = [re.findall(r"\w+", f) for f in fragments]
    n_words = sum(len(t) for t in tokens_per_fragment)
    if n_words < MIN_EVIDENCE_WORDS:
        return False, f"evidence has {n_words} words (< {MIN_EVIDENCE_WORDS})"
    text = _clean_text((title or "") + ". " + (abstract or ""))
    pos = 0
    for tokens in tokens_per_fragment:
        pattern = re.compile(r"\W+".join(re.escape(t) for t in tokens), re.IGNORECASE)
        m = pattern.search(text, pos)
        if not m:
            return False, "sentence not found in title/abstract"
        pos = m.end()
    return True, "found"


# ── Ollama ───────────────────────────────────────────────────────────────────

def check_ollama(model: str) -> bool:
    try:
        resp = _requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        return any(model in m["name"] for m in resp.json().get("models", []))
    except Exception:
        return False


def _call_ollama(messages: list[dict], model: str) -> str:
    payload = {
        "model": model, "messages": messages, "stream": False, "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": 0, "seed": 0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX},
    }
    last_exc: Exception | None = None
    for attempt in range(1, OLLAMA_RETRIES + 1):
        try:
            resp = _requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            msg = resp.json()["message"]
            return (msg.get("content") or msg.get("thinking") or "").strip()
        except (_requests.exceptions.Timeout, _requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            time.sleep(2 * attempt)
    raise last_exc  # type: ignore[misc]


# ── Claude API ───────────────────────────────────────────────────────────────

_RESULT_SCHEMA = {                     # constrains the reply: valid JSON, one of the 7 labels
    "type": "object",
    "properties": {"evidence": {"type": "string"},
                   "condition": {"type": "string", "enum": LABELS}},
    "required": ["evidence", "condition"],
    "additionalProperties": False,
}
_anthropic_client = None


def is_anthropic_model(model: str) -> bool:
    return str(model).startswith("claude-")


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic            # imported lazily: only needed for the API backend
        _anthropic_client = anthropic.Anthropic(timeout=ANTHROPIC_TIMEOUT, max_retries=ANTHROPIC_RETRIES)
    return _anthropic_client


def _call_anthropic(messages: list[dict], model: str) -> tuple[str, int, int]:
    """One classification request. Returns (reply text, input tokens, output tokens).
    Sampling parameters are not sent (current Claude models reject non-default ones)."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    chat = [m for m in messages if m["role"] != "system"]
    kwargs = {"output_config": {"effort": ANTHROPIC_EFFORT,
                                "format": {"type": "json_schema", "schema": _RESULT_SCHEMA}}}
    if not ANTHROPIC_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    resp = _get_anthropic_client().messages.create(
        model=model, max_tokens=ANTHROPIC_MAX_TOKENS, system=system, messages=chat, **kwargs)
    if resp.stop_reason == "refusal":
        raise RuntimeError("the model declined to answer (stop_reason=refusal)")
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> "float | None":
    price = API_PRICES.get(model)
    return None if price is None else (tokens_in * price[0] + tokens_out * price[1]) / 1e6


def check_model(model: str) -> bool:
    """True if `model` can be used. Ollama: server reachable and model pulled.
    Claude API: one tiny real request (~$0.0002) that proves the key, the credit balance,
    the model name and the request format before a long run starts."""
    if not is_anthropic_model(model):
        if check_ollama(model):
            return True
        print(f"[!] Ollama not reachable or model '{model}' not found. "
              f"Start Ollama and run:  ollama pull {model}")
        return False
    try:
        text, _, _ = _call_anthropic(build_messages("Preflight test.", "This is a short test abstract."), model)
        if _parse_response(text) is None:
            print(f"[!] Claude API reachable but the reply is not JSON: {text[:200]!r}")
            return False
        return True
    except Exception as exc:
        print(f"[!] Claude API check failed for '{model}': {type(exc).__name__}: {exc}\n"
              "    Needs ANTHROPIC_API_KEY (new terminal after `setx`) and credit in the "
              "Anthropic Console (Billing).")
        return False


def _call_llm(messages: list[dict], model: str) -> tuple[str, int, int]:
    if is_anthropic_model(model):
        return _call_anthropic(messages, model)
    return _call_ollama(messages, model), 0, 0


def _parse_response(raw: str) -> dict | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def classify_paper(paper: dict, model: str) -> dict:
    """Classify one paper; always returns a record (never raises)."""
    title, abstract = paper.get("title", ""), paper.get("abstract", "")
    rec = {"pmid": str(paper.get("pmid", "")), "model": model, "prompt_version": PROMPT_VERSION,
           "label": None, "evidence": "", "status": ERROR,
           "reason": "", "title_only": not (abstract or "").strip(),
           "classified_at": datetime.now().isoformat(timespec="seconds"), "seconds": 0.0}
    t0 = time.time()
    try:
        raw, tok_in, tok_out = _call_llm(build_messages(title, abstract), model)
    except Exception as exc:
        rec.update(status=ERROR, reason=f"{type(exc).__name__}: {exc}"[:300])
        rec["seconds"] = round(time.time() - t0, 2)
        return rec
    rec["seconds"] = round(time.time() - t0, 2)
    if tok_in or tok_out:
        rec.update(tokens_in=tok_in, tokens_out=tok_out)       # API backend: for cost tracking

    obj = _parse_response(raw)
    if obj is None:
        rec.update(status=INVALID_JSON, reason="response is not a JSON object", raw=raw[:400])
        return rec
    label = _LABEL_ALIASES.get(str(obj.get("condition", "")).strip().lower())
    rec["evidence"] = strip_prompt_label(str(obj.get("evidence", "") or ""))
    if label is None:
        rec.update(status=INVALID_LABEL, reason=f"unknown condition {obj.get('condition')!r}",
                   raw=raw[:400])
        return rec
    rec["label"] = label
    ok, reason = verify_evidence(rec["evidence"], title, abstract)
    rec["reason"] = reason
    if ok:
        rec["status"] = OK
    else:
        rec["status"] = TOO_SHORT if reason.startswith(("empty", "evidence has")) else HALLUCINATED
    return rec


# ── Cache (JSONL, append-only) ───────────────────────────────────────────────

def load_condition_labels(path: "str | Path", model: str | None = None) -> dict[str, dict]:
    """pmid -> record. Later lines win. With `model`, records made with another
    model or prompt version are ignored (treated as not classified)."""
    path = Path(path)
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue            # a half-written last line after a crash
            if model is not None and (rec.get("model") != model
                                      or rec.get("prompt_version") != PROMPT_VERSION):
                continue
            out[rec["pmid"]] = rec
    return out


def collect_unique_papers(results: dict, priority_pairs: "set[str] | None" = None
                          ) -> dict[str, dict]:
    """pmid -> paper (title/abstract) over all pairs of a literature results dict,
    in processing order: papers of `priority_pairs` first, then the other pairs
    from the smallest to the largest (so most pairs are complete early and the
    giant pairs come last). A paper keeps the position of its earliest pair."""
    priority_pairs = priority_pairs or set()
    keys = sorted(results, key=lambda k: (k not in priority_pairs, len(results[k].get("papers", [])), k))
    papers: dict[str, dict] = {}
    for key in keys:
        for p in results[key].get("papers", []):
            pid = str(p.get("pmid", "")).strip()
            if pid and pid not in papers:
                papers[pid] = p
    return papers


def classify_condition_labels(results: dict, labels_path: "str | Path", model: str,
                              limit: int | None = None, seed: int = 0, workers: int | None = None,
                              retry_errors: bool = False,
                              priority_pairs: "set[str] | None" = None,
                              max_cost: float | None = None) -> dict:
    """Classify every not-yet-classified paper of `results` and append to the cache.
    workers: parallel requests (default 1 for Ollama, ANTHROPIC_WORKERS for the Claude API).
    max_cost: Claude API only -- stop once the estimated spend of this run exceeds it (USD)."""
    labels_path = Path(labels_path)
    if not check_model(model):
        if is_anthropic_model(model):
            sys.exit(2)             # stop the whole run (run_haiku_all.ps1 halts on a non-zero exit code)
        return load_condition_labels(labels_path, model)
    if workers is None:
        workers = ANTHROPIC_WORKERS if is_anthropic_model(model) else 1

    papers = collect_unique_papers(results, priority_pairs)
    done = load_condition_labels(labels_path, model)
    todo = [pid for pid in papers
            if pid not in done or (retry_errors and done[pid]["status"] == ERROR)]
    total_unique = len(papers)
    n_already = sum(1 for pid in papers if pid in done)
    if limit is not None:
        random.Random(seed).shuffle(todo)
        todo = todo[:limit]
    print(f"[i] {total_unique} unique papers | {n_already} already classified "
          f"(model {model}, prompt {PROMPT_VERSION}) | {len(todo)} to do | workers: {workers}")
    if not todo:
        return done

    counts: Counter = Counter()
    t_start = time.time()
    n_done = 0
    tok_in = tok_out = 0
    error_streak = 0
    stop_reason = ""
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    def _work(pid: str) -> dict:
        return classify_paper(papers[pid], model)

    def _cost() -> "float | None":
        return estimate_cost(model, tok_in, tok_out)

    # Papers are submitted in small chunks so a stop (budget, repeated errors, Ctrl+C)
    # takes effect within a few requests instead of after the whole queue.
    workers = max(1, workers)
    chunk = workers * 4
    try:
        with open(labels_path, "a", encoding="utf-8") as out, \
                ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(todo), chunk):
                if stop_reason:
                    break
                for rec in pool.map(_work, todo[start:start + chunk]):
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    done[rec["pmid"]] = rec
                    counts[rec["status"]] += 1
                    tok_in += rec.get("tokens_in", 0)
                    tok_out += rec.get("tokens_out", 0)
                    error_streak = error_streak + 1 if rec["status"] == ERROR else 0
                    n_done += 1
                    if n_done % 20 == 0 or n_done == len(todo):
                        el = time.time() - t_start
                        eta = el / n_done * (len(todo) - n_done)
                        cost = _cost()
                        print(f"  [{n_done}/{len(todo)}] {el / n_done:.2f} s/paper | ETA {eta / 3600:.1f} h | "
                              + " ".join(f"{s}={counts[s]}" for s in STATUSES if counts[s])
                              + (f" | ~${cost:.3f}" if cost is not None else ""), flush=True)
                if error_streak >= MAX_CONSECUTIVE_ERRORS:
                    stop_reason = (f"{error_streak} requests failed in a row (last: "
                                   f"{done[rec['pmid']].get('reason', '')[:160]}). Fix the cause, then re-run "
                                   "with --retry-errors.")
                elif max_cost is not None and (_cost() or 0) >= max_cost:
                    stop_reason = f"estimated spend ${_cost():.3f} reached --max-cost ${max_cost:.2f}."
    except KeyboardInterrupt:
        print("\n[!] Interrupted -- progress is saved; re-run the same command to resume.")
    if stop_reason:
        print(f"[!] Stopped: {stop_reason}")
    cost = _cost()
    print(f"[ok] classified {n_done} papers in {(time.time() - t_start) / 60:.1f} min -> {labels_path}"
          + (f" | {tok_in:,} input + {tok_out:,} output tokens ~ ${cost:.3f}" if cost is not None else ""))
    return done


# ── Filter object used by the pipeline ───────────────────────────────────────

class LLMConditionFilter:
    """Accepts a paper if the LLM labelled it with this case's label AND the
    evidence sentence passed the regex test (status 'ok')."""

    def __init__(self, condition: str, labels: dict[str, dict]):
        self.condition = condition
        self.target = CONDITION_TO_LABEL[condition]
        self.labels = labels

    def accepts(self, paper: dict) -> bool:
        rec = self.labels.get(str(paper.get("pmid", "")))
        return bool(rec and rec.get("status") == OK and rec.get("label") == self.target)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    here = Path(__file__).resolve().parent.parent
    out_dir = here / "reference_network_only_metabolic"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=out_dir / "metabolic_literature_results.json")
    ap.add_argument("--labels", type=Path, default=out_dir / "metabolic_condition_labels.jsonl")
    ap.add_argument("--model", default=None,
                    help="Ollama model, or a Claude API model such as claude-haiku-5-5 "
                         "(default: CONDITION_LLM_MODEL from config.py)")
    ap.add_argument("--limit", type=int, default=None, help="classify only N random unclassified papers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None,
                    help=f"parallel requests (default 1 for Ollama, {ANTHROPIC_WORKERS} for the Claude API)")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="Claude API: stop when the estimated spend of this run reaches this many USD")
    ap.add_argument("--retry-errors", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(out_dir))
    import config as _cfg
    model = args.model or getattr(_cfg, "CONDITION_LLM_MODEL", _cfg.LLM_MODEL)
    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)
    priority = set()
    for name in ("healthy_cohort_connections.csv", "obese_cohort_connections.csv"):
        csv_path = out_dir / name
        if csv_path.exists():
            import csv as _csv
            with open(csv_path, encoding="utf-8") as f:
                rows = list(_csv.reader(f))
            for row in rows[1:]:
                for col, val in enumerate(row[1:]):
                    if val.strip() == "1" and row[0].strip() != rows[0][col + 1].strip():
                        a, b = row[0].strip(), rows[0][col + 1].strip()
                        priority.add(f"{min(a, b)}|{max(a, b)}")
    classify_condition_labels(results, args.labels, model, limit=args.limit, seed=args.seed,
                              workers=args.workers, retry_errors=args.retry_errors,
                              priority_pairs=priority, max_cost=args.max_cost)


if __name__ == "__main__":
    main()
