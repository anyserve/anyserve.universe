"""Render the READMEs into a static site next to the setup page.

The site is the repository's own documentation: the top-level README is the
overview, each example's README is its page, and setup/ is copied as it is.
Output goes to site/out.
"""
from __future__ import annotations
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "out"
PAGES = [
    ("", "Overview", ROOT / "README.md"),
    ("triton", "Triton", ROOT / "triton" / "README.md"),
    ("vllm", "vLLM", ROOT / "vllm" / "README.md"),
    ("comfyui", "ComfyUI", ROOT / "comfyui" / "README.md"),
    ("video", "Video", ROOT / "video" / "README.md"),
    ("vla", "VLA", ROOT / "vla" / "README.md"),
]


def render(template: str, slug: str, title: str, source: Path) -> str:
    prefix = "" if slug == "" else "../"
    links = [(page_slug, page_title) for page_slug, page_title, _ in PAGES] + [("setup", "Setup")]
    nav = "".join(
        f'<a href="{prefix}{link_slug + "/" if link_slug else ""}"{" class=\"is-active\"" if link_slug == slug else ""}>{link_title}</a>'
        for link_slug, link_title in links
    )
    body = markdown.markdown(source.read_text(), extensions=["fenced_code", "tables"])
    return (
        template.replace("{{title}}", title)
        .replace("{{prefix}}", prefix or "./")
        .replace("{{nav}}", nav)
        .replace("{{content}}", body)
    )


def main() -> None:
    template = (ROOT / "site" / "template.html").read_text()
    shutil.rmtree(OUT, ignore_errors=True)
    for slug, title, source in PAGES:
        target = OUT / slug / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(template, slug, title, source))
    shutil.copytree(ROOT / "setup", OUT / "setup", ignore=shutil.ignore_patterns("*.test.js", "README.md"))
    print(f"wrote {sum(1 for _ in OUT.rglob('*') if _.is_file())} files to {OUT}")


if __name__ == "__main__":
    main()
