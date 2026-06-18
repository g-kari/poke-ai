"""Auto-update docs/index.html sections marked with <!-- AUTO:NAME_START --> ... END.

Currently updates:
  - AUTO:SUBMISSIONS: table of Kaggle submissions (live from kaggle CLI)
  - AUTO:UPDATED: timestamp footer

Usage:
    scripts/run.sh python3 scripts/update_pages.py
    scripts/run.sh python3 scripts/update_pages.py --no-fetch    # use cached data

Skips kaggle fetch on --no-fetch (useful in CI / offline).

The script is idempotent: re-running with the same upstream state produces
no diff. CRON / loop can run it after every successful submit.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_HTML = ROOT / "docs" / "index.html"
MATCHUPS_JSON = ROOT / "data" / "matchups.json"
KAGGLE_BIN = ROOT / ".venv" / "bin" / "kaggle"
COMPETITION = "pokemon-tcg-ai-battle"


def fetch_submissions(limit: int = 6) -> list[dict] | None:
    """Use kaggle CLI to fetch recent submissions. Returns None on error."""
    if not KAGGLE_BIN.exists():
        return None
    try:
        result = subprocess.run(
            [str(KAGGLE_BIN), "competitions", "submissions", "-c", COMPETITION],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"kaggle CLI failed: {result.stderr}", file=sys.stderr)
            return None
    except Exception as exc:  # noqa: BLE001
        print(f"kaggle CLI exception: {exc}", file=sys.stderr)
        return None

    # Parse the table — kaggle prints a header + dashes + rows. Columns are
    # whitespace-separated but file names + descriptions have spaces, so
    # we split conservatively from the right.
    lines = result.stdout.strip().split("\n")
    rows: list[dict] = []
    for line in lines[2:]:
        # Skip empty lines.
        if not line.strip():
            continue
        # Status + score are typically at the end.
        # Example tail: "... SubmissionStatus.COMPLETE  897.6"
        m = re.match(
            r"^\s*(\d+)\s+(\S+)\s+(\S+\s+\S+)\s+(.+?)\s+(SubmissionStatus\.\S+)\s+(\S+)?\s*(\S+)?$",
            line,
        )
        if not m:
            continue
        ref, fname, _date, desc, status, public_score, _private = m.groups()
        rows.append(
            {
                "ref": ref,
                "file": fname.replace("submission_", "").replace(".tar.gz", ""),
                "status": status.replace("SubmissionStatus.", ""),
                "public_score": public_score if public_score and public_score != "—" else None,
                "description": desc.strip(),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def render_submissions(rows: list[dict]) -> str:
    """Generate the inner HTML for the submission-history table."""

    # Find the leader (highest public_score among COMPLETE rows).
    def _score(r):
        try:
            return float(r.get("public_score") or 0)
        except (TypeError, ValueError):
            return 0.0

    completed = [r for r in rows if r["status"] == "COMPLETE" and r["public_score"]]
    leader_ref = max(completed, key=_score)["ref"] if completed else None
    latest_ref = rows[0]["ref"] if rows else None

    lines = [
        '  <table class="submission-history">',
        "    <thead>",
        "      <tr><th>ref</th><th>file</th><th>status</th><th>public score</th></tr>",
        "    </thead>",
        "    <tbody>",
    ]
    for r in rows:
        cls = ""
        if r["ref"] == leader_ref:
            cls = ' class="leader"'
        elif r["ref"] == latest_ref:
            cls = ' class="latest"'
        score_cell = (
            f"<strong>{r['public_score']}</strong>"
            if r["ref"] == leader_ref
            else (r["public_score"] or "—")
        )
        lines.append(
            f"      <tr{cls}><td>{r['ref']}</td><td>{r['file']}</td>"
            f"<td>{r['status']}</td><td>{score_cell}</td></tr>"
        )
    lines += [
        "    </tbody>",
        "  </table>",
    ]
    return "\n".join(lines)


def render_ranking() -> str | None:
    """Generate the subject ranking <ol> from data/matchups.json."""
    if not MATCHUPS_JSON.exists():
        return None
    try:
        data = json.loads(MATCHUPS_JSON.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"matchups.json parse failed: {exc}", file=sys.stderr)
        return None
    subjects = data.get("subjects") or []
    # Sort by overall_winrate descending.
    ranked = sorted(
        [s for s in subjects if isinstance(s.get("overall_winrate"), (int, float))],
        key=lambda s: -float(s["overall_winrate"]),
    )
    if not ranked:
        return None
    lines = ['  <ol class="ranking">']
    for s in ranked:
        label = s.get("label") or s.get("id") or "?"
        wr = float(s["overall_winrate"]) * 100
        crit = s.get("critical_weakness")
        crit_note = ""
        if crit:
            opp = crit.get("opp") or "?"
            opp_wr = crit.get("winrate")
            if isinstance(opp_wr, (int, float)):
                crit_note = f" (致命弱点: {opp} {opp_wr * 100:.1f}%)"
        else:
            note = s.get("note")
            if note:
                crit_note = f" ({note})"
        cls = ' class="ours"' if s.get("id", "").startswith("ours_") else ""
        bold_open = "<strong>" if not cls else ""
        bold_close = "</strong>" if not cls else ""
        lines.append(f"    <li{cls}>{bold_open}{label}{bold_close} — {wr:.1f}%{crit_note}</li>")
    lines.append("  </ol>")
    return "\n".join(lines)


def replace_section(html: str, marker: str, new_inner: str) -> str:
    """Replace content between <!-- AUTO:MARKER_START --> and <!-- AUTO:MARKER_END -->."""
    pattern = re.compile(
        rf"(<!-- AUTO:{marker}_START.*?-->)(.*?)(<!-- AUTO:{marker}_END -->)",
        re.DOTALL,
    )
    if not pattern.search(html):
        print(f"  marker AUTO:{marker} not found in HTML, skipping", file=sys.stderr)
        return html
    return pattern.sub(rf"\1\n{new_inner}\n  \3", html)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip kaggle CLI fetch (useful for offline / CI runs).",
    )
    args = p.parse_args()

    if not DOCS_HTML.exists():
        print(f"docs/index.html not found at {DOCS_HTML}", file=sys.stderr)
        return 1

    html = DOCS_HTML.read_text(encoding="utf-8")
    original = html

    # ---- AUTO:SUBMISSIONS ----
    if not args.no_fetch:
        rows = fetch_submissions(limit=6)
        if rows:
            print(f"  fetched {len(rows)} submissions from kaggle")
            html = replace_section(html, "SUBMISSIONS", render_submissions(rows))
        else:
            print("  kaggle fetch failed/skipped; AUTO:SUBMISSIONS unchanged")

    # ---- AUTO:RANKING ----
    ranking_html = render_ranking()
    if ranking_html:
        print("  rendered ranking from matchups.json")
        html = replace_section(html, "RANKING", ranking_html)
    else:
        print("  ranking skipped (no matchups.json or parse error)")

    # ---- AUTO:UPDATED ----
    # Use a date passed via env to keep results deterministic in tests.
    import os

    today = os.environ.get("POKE_AI_TODAY", str(datetime.date.today()))
    html = replace_section(
        html,
        "UPDATED",
        f"    Auto-generated section last updated: {today}",
    )

    if html == original:
        print("  no changes")
        return 0
    DOCS_HTML.write_text(html, encoding="utf-8")
    print(f"  wrote {DOCS_HTML}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
