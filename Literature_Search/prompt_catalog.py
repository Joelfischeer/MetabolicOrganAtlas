"""
The LLM prompts of the pipeline, for the dashboard tab "LLM Prompts".

Nothing is copied by hand: every prompt below is produced by calling the same functions and
constants that the pipeline uses (llm_condition_filter, llm_connection_type, llm_descriptions,
claude_descriptions, organ_descriptions) with placeholder data, so the tab cannot drift away from
the prompts that are actually sent. Which variant is shown (Claude API or local Ollama) follows
the model configured for each step in reference_network_only_metabolic/config.py.
"""

import html
import json

from Literature_Search import claude_descriptions as cd
from Literature_Search import llm_condition_filter as cf
from Literature_Search import llm_connection_type as ct
from Literature_Search import llm_descriptions as ld
from Literature_Search import organ_descriptions as od

_A, _B, _ORGAN = "[Organ A]", "[Organ B]", "[Organ]"


def _papers(n: int, abstract: str = "<abstract of the paper>") -> list[dict]:
    return [{"pmid": f"<PMID {i}>", "year": "<year>", "title": f"<title of paper {i}>", "abstract": abstract}
            for i in range(1, n + 1)]


def _backend(model: str) -> str:
    return "Claude API" if str(model).startswith("claude-") else "local Ollama model"


# ── one entry per LLM step ───────────────────────────────────────────────────

def _condition_entry(model: str) -> dict:
    claude = cf.is_anthropic_model(model)
    msgs = cf.build_messages("<title of the paper>", "<abstract of the paper>")
    blocks: list = [("System prompt", msgs[0]["content"])]
    shots = [f"Example {i // 2 + 1} - user message\n{msgs[i]['content']}\n\nExample {i // 2 + 1} - assistant reply\n"
             f"{msgs[i + 1]['content']}" for i in range(1, len(msgs) - 1, 2)]
    blocks.append(("__details__", (f"{len(shots)} worked examples (one per label), sent as earlier chat turns",
                                   "\n\n".join(shots))))
    blocks.append(("User message for every paper (the paper's real title and abstract replace the placeholders)",
                   msgs[-1]["content"]))
    if claude:
        blocks.append(("Reply format (JSON schema the answer must follow)", json.dumps(cf._RESULT_SCHEMA, indent=2)))
        settings = (f"reasoning {'on' if cf.ANTHROPIC_THINKING else 'off'}, effort '{cf.ANTHROPIC_EFFORT}', at most "
                    f"{cf.ANTHROPIC_MAX_TOKENS} output tokens; the abstract is cut at {cf.MAX_ABSTRACT_CHARS:,} characters; "
                    "Claude models do not accept a temperature setting")
    else:
        settings = (f"temperature 0, seed 0, JSON mode, at most {cf.NUM_PREDICT} output tokens, context {cf.NUM_CTX}; "
                    f"the abstract is cut at {cf.MAX_ABSTRACT_CHARS:,} characters")
    return {"title": "1. Condition classification of every paper",
            "purpose": "Assigns each paper one primary condition of the studied subjects and quotes the sentence that led to "
                       "the decision. The quoted sentence is then checked by a regex test (not an LLM step) against the "
                       "title and abstract; if it does not occur there, the answer is treated as a hallucination.",
            "model": model, "backend": _backend(model), "settings": settings, "blocks": blocks}


def _types_entry(model: str, connection_types: dict, max_papers: int) -> dict:
    papers = _papers(2)
    if ct._is_claude(model):
        data, task = ct._claude_prompt_blocks(_A, _B, papers, connection_types, max_papers)
        blocks = [("System prompt", ct._CLAUDE_SYSTEM),
                  ("User message, part 1 - papers and the list of connection types (prompt-cached; shown with 2 "
                   f"placeholder papers, the real prompt lists the first {max_papers} papers of the pair)", data["text"]),
                  ("User message, part 2 - task", task["text"]),
                  ("Reply format (JSON schema the answer must follow)",
                   json.dumps(ct._claude_schema(list(connection_types)), indent=2))]
        settings = (f"{ct.VOTE_RUNS} independent runs per pair, a type is kept if it appears in a majority of them; "
                    f"reasoning off, effort '{ct.CLAUDE_EFFORT}', at most {ct.CLAUDE_MAX_TOKENS} output tokens; abstracts "
                    f"cut at {ct.MAX_ABSTRACT_CHARS} characters")
    else:
        blocks = [("Prompt (shown with 2 placeholder papers; the real prompt lists up to "
                   f"{max_papers} papers of the pair)", ct._build_prompt(_A, _B, papers, connection_types, max_papers))]
        settings = (f"{ct.VOTE_RUNS} independent runs per pair at temperature {ct.VOTE_TEMPERATURE}, majority vote; "
                    f"abstracts cut at {ct.MAX_ABSTRACT_CHARS} characters")
    return {"title": "2. Connection types of an organ pair",
            "purpose": "Chooses 1-3 connection types from the configured list that best fit the papers of one organ pair.",
            "model": model, "backend": _backend(model), "settings": settings, "blocks": blocks}


def _pair_description_entry(model: str) -> dict:
    if cd.is_claude_model(model):
        sel = cd._select_prompt("pair", (_A, _B), _papers(2), cd.CLAUDE_TOP_N).replace("Below are 2 papers", "Below are [N] papers")
        blocks = [("Step 1, choosing the sources - system prompt", cd.SYSTEM_PROMPT),
                  (f"Step 1, choosing the sources - user message (a random pool of up to {cd.CLAUDE_POOL} papers is "
                   f"shown, the {cd.CLAUDE_TOP_N} chosen papers become the sources)", sel),
                  ("Step 1, choosing the sources - reply format (JSON schema the answer must follow)",
                   json.dumps(cd._SELECT_SCHEMA, indent=2)),
                  ("Step 2, writing the summary - system prompt", cd.SYSTEM_PROMPT),
                  (f"Step 2, writing the summary - the {cd.CLAUDE_TOP_N} sources are attached as documents (title + "
                   "abstract, citations enabled), followed by this text", cd._summary_instruction("pair", (_A, _B)))]
        settings = (f"pool of {cd.CLAUDE_POOL} papers, {cd.CLAUDE_TOP_N} sources; reasoning off; effort 'low' for the "
                    f"selection and '{cd.EFFORT}' for the summary; at most {cd.SUMMARY_MAX_TOKENS:,} output tokens. The Claude "
                    "citations feature attaches the supporting passage to every sentence; the reference numbers in the text "
                    "are generated from these citations, not written by the model")
    else:
        n = ld.TOP_N_PAPERS + 1
        blocks = [("Step 1, choosing the sources (shown with placeholder papers)",
                   ld._build_selection_prompt(_A, _B, _papers(n)).replace(f"Below are {n} papers", "Below are [N] papers")),
                  ("Step 2, writing the summary (shown with placeholder papers)",
                   ld._build_summary_prompt(_A, _B, _papers(ld.TOP_N_PAPERS)))]
        settings = (f"pool of {ld.SAMPLE_POOL} papers, {ld.TOP_N_PAPERS} sources; temperature 0.3, at most 512 output "
                    f"tokens; abstracts cut at {ld.MAX_ABSTRACT_CHARS} characters")
    return {"title": "3. Description of an organ-organ connection",
            "purpose": "Two steps: the model picks the most relevant papers of the pair, then writes a cited summary "
                       "based only on those papers.",
            "model": model, "backend": _backend(model), "settings": settings, "blocks": blocks}


def _organ_description_entry(model: str) -> dict:
    if cd.is_claude_model(model):
        sel = cd._select_prompt("organ", (_ORGAN,), _papers(2), cd.CLAUDE_TOP_N).replace("Below are 2 papers", "Below are [N] papers")
        blocks = [("Step 1, choosing the sources - system prompt", cd.SYSTEM_PROMPT),
                  ("Step 1, choosing the sources - user message", sel),
                  ("Step 1, choosing the sources - reply format (JSON schema the answer must follow)",
                   json.dumps(cd._SELECT_SCHEMA, indent=2)),
                  (f"Step 2, writing the summary - the {cd.CLAUDE_TOP_N} sources are attached as documents, followed by "
                   "this text", cd._summary_instruction("organ", (_ORGAN,)))]
        settings = f"same procedure and checks as the connection descriptions (pool of {cd.CLAUDE_POOL}, {cd.CLAUDE_TOP_N} sources)"
    else:
        blocks = [("Prompt (shown with placeholder papers)", od._build_organ_prompt(_ORGAN, _papers(3))),
                  ("Fallback prompt, used only if the model refuses", od._fallback_prompt(_ORGAN))]
        settings = f"up to {od.MAX_PAPERS_PROMPT} papers, abstracts cut at {od.MAX_ABSTRACT_CHARS} characters; temperature 0.3"
    return {"title": "4. Description of a single organ",
            "purpose": "Writes the metabolic overview that is shown when an organ node is clicked.",
            "model": model, "backend": _backend(model), "settings": settings, "blocks": blocks}


def build_prompt_catalog(condition_model: str, type_model: str, description_model: str,
                         connection_types: dict, max_papers: int) -> list[dict]:
    return [_condition_entry(condition_model),
            _types_entry(type_model, connection_types, max_papers),
            _pair_description_entry(description_model),
            _organ_description_entry(description_model)]


# ── HTML ─────────────────────────────────────────────────────────────────────

_LABEL = 'style="font-size:11px;color:#94a3b8;margin:10px 0 2px"'


def prompts_html(catalog: list[dict]) -> str:
    e = html.escape
    out = []
    for entry in catalog:
        out.append(f'<div class="info-h2">{e(entry["title"])}</div>')
        out.append(f'<p class="info-p">{e(entry["purpose"])}</p>')
        out.append(f'<p class="info-p" style="color:#94a3b8;font-size:12px">Model: <code>{e(entry["model"])}</code> '
                   f'({e(entry["backend"])}) &middot; {e(entry["settings"])}</p>')
        for heading, text in entry["blocks"]:
            if heading == "__details__":
                summary, body = text
                out.append(f'<details style="margin:8px 0"><summary style="cursor:pointer;color:#a5b4fc;font-size:12px">'
                           f'{e(summary)}</summary><div class="info-code">{e(body)}</div></details>')
            else:
                out.append(f'<div {_LABEL}>{e(heading)}</div><div class="info-code">{e(text)}</div>')
    return "\n".join(out)
