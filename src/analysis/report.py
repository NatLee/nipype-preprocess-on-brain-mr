"""HTML pages: one QC report per subject and an index page for the whole run.

The pages are plain HTML + CSS rendered from Jinja2 templates in
`analysis/templates/`, with figures stored next to them as PNG / GIF files,
so a whole output folder can be zipped and opened anywhere without a server.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from analysis.tissue import METHOD_NAMES, TISSUE_NAMES, TISSUES

TEMPLATES = Path(__file__).with_name("templates")


def _environment() -> Environment:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html", "j2"]))
    env.filters["ml"] = lambda value: f"{value:,.0f}"
    env.filters["pct"] = lambda value: f"{value * 100:.1f}%"
    env.filters["num"] = lambda value, digits=3: f"{value:.{digits}f}"
    env.globals.update(TISSUES=TISSUES, TISSUE_NAMES=TISSUE_NAMES, METHOD_NAMES=METHOD_NAMES)
    return env


def relative(path: str | Path | None, start: Path) -> str | None:
    """Path of `path` as seen from the page in folder `start` (forward slashes, for href/src)."""
    if not path:
        return None
    return Path(os.path.relpath(Path(path), start)).as_posix()


def render_page(template: str, out_file: str | Path, context: Mapping) -> Path:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = _environment().get_template(template).render(generated=generated, **context)
    out_file.write_text(html, encoding="utf-8")
    return out_file
