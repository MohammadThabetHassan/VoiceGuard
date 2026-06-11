#!/usr/bin/env python3
"""
Watch HuggingFace Hub for newly released voice synthesis / cloning models.

Usage:
    python scripts/watch_synthesis_models.py           # dry run, exits 1 if new
    python scripts/watch_synthesis_models.py --update  # append new to JSON
    python scripts/watch_synthesis_models.py --limit 200

Exit codes:
    0 — nothing new
    1 — new models found (useful in CI to trigger re-evaluation)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

CATALOG_PATH = (
    Path(__file__).parent.parent / "src/voiceguard/models/known_synthesis_models.json"
)
HF_API = "https://huggingface.co/api/models"
FILTERS = ["text-to-speech", "voice-conversion", "audio-to-audio"]
SORT = "lastModified"


def _fetch(filter_tag: str, limit: int) -> list[dict]:
    url = f"{HF_API}?filter={filter_tag}&sort={SORT}&limit={limit}&direction=-1"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310
            return json.loads(resp.read())
    except Exception as e:
        print(f"  WARNING: failed to fetch {filter_tag}: {e}", file=sys.stderr)
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch HuggingFace for new voice synthesis models")
    parser.add_argument("--update", action="store_true", help="Append new models to the catalog")
    parser.add_argument("--limit", type=int, default=100, help="Models to fetch per filter tag")
    args = parser.parse_args()

    catalog: list[dict] = json.loads(CATALOG_PATH.read_text())
    known_ids = {m["id"] for m in catalog}

    print(f"Catalog: {len(catalog)} known models")
    print(f"Fetching top {args.limit} models per tag from HuggingFace Hub...")

    new_models: list[dict] = []
    seen_ids: set[str] = set()

    for tag in FILTERS:
        results = _fetch(tag, args.limit)
        for m in results:
            mid = m.get("modelId") or m.get("id", "")
            if not mid or mid in known_ids or mid in seen_ids:
                continue
            seen_ids.add(mid)
            new_models.append({
                "id": mid,
                "type": tag,
                "added": datetime.now(UTC).strftime("%Y-%m-%d"),
                "lastModified": m.get("lastModified", ""),
                "downloads": m.get("downloads", 0),
            })

    if not new_models:
        print("No new models found.")
        return 0

    new_models.sort(key=lambda m: m.get("downloads", 0), reverse=True)
    print(f"\n{len(new_models)} NEW models found:\n")
    for m in new_models:
        dl = m.get("downloads", 0)
        print(f"  [{m['type']:20s}] {m['id']}  ({dl:,} downloads)")

    if args.update:
        for m in new_models:
            catalog.append({"id": m["id"], "type": m["type"], "added": m["added"]})
        CATALOG_PATH.write_text(json.dumps(catalog, indent=2))
        print(f"\nCatalog updated: {CATALOG_PATH}")
    else:
        print("\nRun with --update to add these to the catalog.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
