"""
Loader for the editable texts of the dashboard's "how it works" info panel.

The texts and their HTML layout live in Visualisation/info_panel_texts.html (see the header
of that file for the format). render_section() returns one section with its
[[placeholders]] filled in.
"""
import re
from pathlib import Path

TEXT_FILE = Path(__file__).with_name("info_panel_texts.html")
_SECTION_RE = re.compile(r"^@@ section:\s*(\w+)\s*$")
_PLACEHOLDER_RE = re.compile(r"\[\[(\w+)\]\]")


def load_sections(path: Path = TEXT_FILE) -> dict[str, str]:
    """{section name: raw HTML with [[placeholders]]}; "## " comment lines are dropped."""
    sections: dict[str, str] = {}
    name, lines = None, []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        m = _SECTION_RE.match(line.rstrip())
        if m:
            if name is not None:
                sections[name] = "\n".join(lines).strip()
            name, lines = m.group(1), []
            continue
        if line.startswith("## ") or line.rstrip() == "##":
            continue
        if name is not None:
            lines.append(line)
    if name is not None:
        sections[name] = "\n".join(lines).strip()
    return sections


def render_section(name: str, **values) -> str:
    """Return section `name` with every [[key]] replaced by str(values[key])."""
    sections = load_sections()
    if name not in sections:
        raise KeyError(f"section '{name}' not found in {TEXT_FILE.name} "
                       f"(available: {', '.join(sections)})")

    def _sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in values:
            print(f"  [!] {TEXT_FILE.name}: unknown placeholder [[{key}]] in section '{name}'")
            return m.group(0)
        return str(values[key])

    return _PLACEHOLDER_RE.sub(_sub, sections[name])
