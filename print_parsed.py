"""Parse a document and save its EXACT parsed output in data/parsed output.

The .parse.json file is a verbatim dump of the raw LlamaParse pages - no
formatting, no additions. A human-readable .txt rendering is written too.

Usage:
    python print_parsed.py <path-to-pdf> [more.pdf ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from app.offline.dump_outputs import save_parse_dump
from app.offline.parser import LlamaParser

OUTPUT_DIR = Path(__file__).parent / "data" / "parsed output"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python print_parsed.py <file.pdf> [file2.pdf ...]")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = LlamaParser()

    for arg in sys.argv[1:]:
        path = Path(arg)
        pages = parser.parse(path)

        # Exact raw output, byte-for-byte what LlamaParse returned.
        json_path = save_parse_dump(path.stem, pages)

        # Readable text rendering of the same data.
        out_path = OUTPUT_DIR / f"{path.stem}.txt"
        lines: list[str] = []
        for page in pages:
            lines.append(f"===== PAGE {page.get('page', '?')} =====")
            for item in page.get("items", []):
                lines.append(f"[{item.get('type')}] {item.get('value') or ''}")
            lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"Saved EXACT raw output ({len(pages)} pages) -> {json_path}")
        print(f"Saved readable rendering                        -> {out_path}")


if __name__ == "__main__":
    main()
