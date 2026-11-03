"""Fail when docs drift from the canonical reports or resurrect retired claims.

Two layers:

1. Regenerate every generated block (``render_docs.render_all``) and compare.
2. Scan the prose for retired numbers/phrases and require the headline numbers
   that a reviewer should be able to find.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_docs import render_all  # noqa: E402

DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "RESULTS.md",
    ROOT / "docs" / "MODEL_CARD.md",
    ROOT / "docs" / "DECISION_MEMO.md",
    ROOT / "docs" / "INTERVIEW_DEFENSE.md",
    ROOT / "docs" / "TABLEAU_DASHBOARD_GUIDE.md",
    ROOT / "docs" / "resume_bullets.tex",
    ROOT / "dashboards" / "data" / "README.md",
    ROOT / "docs" / "index.html",
]

FORBIDDEN = [
    ("YOUR_USERNAME", r"YOUR_USERNAME"),
    ("0.0500", r"(?<![\d.])0\.0500(?![\d])"),
    ("4.7%", r"(?<![\d.])4\.7%"),
    ("31.8%", r"(?<![\d.])31\.8%"),
    ("10,000 accounts", r"10,000 accounts"),
    ("10,000 per arm", r"10,000 per arm"),
    ("1% sample", r"1% sample"),
    ("1% stratified", r"1% stratified"),
    ("~0.9999", r"~0\.9999"),
    ("invalid p-value", r"invalid p-value"),
    ("DeLong Weighting Flaw", r"DeLong Weighting Flaw"),
    ("DeLong is invalid", r"DeLong is invalid"),
    ("400+", r"(?<!\d)400\+"),
    ("$348 Million", r"\$348 Million"),
    ("$718 Million", r"\$718 Million"),
    ("20 core numeric features", r"20 core numeric features"),
    ("Tableau dashboard published", r"Tableau dashboard published"),
]

# Negated design notes are allowed: the interview doc explicitly says the final
# run did not use these. Any other mention is a stale claim.
ALLOWED_CONTEXTS = {
    "Optuna": ("no Optuna",),
    "GroupKFold": ("no GroupKFold",),
}

REQUIRED = [
    "0.9584",
    "0.7814",
    "0.012055",
    "0.0075%",
    "0.000827",
    "0.040517",
    "0.053124",
    "1.83%",
    "4.26%",
    "60.0%",
    "31.1%",
    "9.0%",
    "$102.0M",
    "$100.6M",
    "$103.5M",
    "17,610",
    "0.0003",
    "199 green",
    "1.717%",
    "25.89%",
    "282",
    "200 features",
    "+5.1%",
    "+9.7%",
    "-17.8%",
    "+30.7%",
    "2.34%",
]


def main() -> None:
    failures: list[str] = []

    for path, expected in render_all().items():
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != expected:
            failures.append(f"stale generated block: {path.relative_to(ROOT)}")

    combined = ""
    for path in DOCS:
        if not path.exists():
            failures.append(f"missing doc: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        combined += "\n" + text

        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in FORBIDDEN:
                if re.search(pattern, line):
                    failures.append(
                        f"{path.relative_to(ROOT)}:{line_number}: retired phrase {label!r}"
                    )
            for phrase in ("Optuna", "GroupKFold"):
                if phrase not in line:
                    continue
                if any(context in line for context in ALLOWED_CONTEXTS[phrase]):
                    continue
                failures.append(
                    f"{path.relative_to(ROOT)}:{line_number}: stale claim {phrase!r}"
                )

    for token in REQUIRED:
        if token not in combined:
            failures.append(f"required headline token missing: {token!r}")

    if failures:
        print("Documentation verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        print("\nRun: python scripts/render_docs.py --write")
        sys.exit(1)

    print("Documentation verification passed.")


if __name__ == "__main__":
    main()
