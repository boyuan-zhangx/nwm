from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
CJK_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _documents() -> list[Path]:
    return sorted(DOCS_ROOT.rglob("*.md"))


def test_handoff_docs_are_english() -> None:
    violations: list[str] = []
    for path in _documents():
        text = path.read_text(encoding="utf-8")
        if CJK_CHARACTER.search(text):
            violations.append(str(path.relative_to(REPO_ROOT)))
    assert not violations, f"CJK text remains in handoff docs: {violations}"


def test_handoff_docs_do_not_contain_maintainer_paths() -> None:
    forbidden = (
        re.compile(r"[A-Za-z]:\\Navware_workspace", re.IGNORECASE),
        re.compile(r"/mnt/[A-Za-z]/Navware_workspace", re.IGNORECASE),
        re.compile(r"/home/[^/\s`]+/\.venvs/navware-nwm"),
        re.compile(r"Legion-[A-Za-z0-9_-]+", re.IGNORECASE),
    )
    violations: list[str] = []
    for path in _documents():
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern.search(text):
                violations.append(f"{path.relative_to(REPO_ROOT)}: {pattern.pattern}")
    assert not violations, f"Maintainer-specific paths found: {violations}"


def test_handoff_shell_scripts_do_not_contain_site_hardcoding() -> None:
    scripts = (
        REPO_ROOT / "nwm.sh",
        REPO_ROOT / "scripts" / "train.sh",
        REPO_ROOT / "scripts" / "infer.sh",
    )
    forbidden = (
        re.compile(r"^#SBATCH\s+--(?:partition|nodelist|account)=", re.MULTILINE),
        re.compile(r"(?:source\s+.*miniconda|conda\s+activate)"),
        re.compile(r"cd\s+[^\n]*\$\{?HOME\}?/"),
        re.compile(r"config/[^\s]*L40S\.yaml", re.IGNORECASE),
        re.compile(r"[A-Za-z]:\\Navware_workspace", re.IGNORECASE),
        re.compile(r"/mnt/[A-Za-z]/Navware_workspace", re.IGNORECASE),
    )
    violations: list[str] = []
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern.search(text):
                violations.append(f"{path.relative_to(REPO_ROOT)}: {pattern.pattern}")
    assert not violations, f"Site hard-coding found: {violations}"


def test_relative_markdown_links_resolve() -> None:
    broken: list[str] = []
    for path in _documents():
        text = path.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target_path = target.split("#", 1)[0]
            if target_path and not (path.parent / target_path).resolve().exists():
                broken.append(f"{path.relative_to(REPO_ROOT)} -> {target}")
    assert not broken, f"Broken local Markdown links: {broken}"
