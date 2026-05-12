"""
Rebuild Task A submission_*.tsv files from outputs/predictions_*.tsv.

Reads compound, sentence, expected_order from each predictions file, formats
expected_order like main.py exports, writes submission_{lang}.tsv into outputs/
(lowercase language tags, e.g. submission_zh.tsv).

Usage (from admire_text_bottleneck):
    python3 rebuild_submissions_from_predictions.py
"""

from __future__ import annotations

import csv
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_OUTPUTS = _ROOT / "outputs"

_STEM_TO_CODE: dict[str, str] = {
    "chinese": "zh",
    "georgian": "ka",
    "greek": "el",
    "igbo": "ig",
    "kazakh": "kk",
    "norwegian": "no",
    "portuguese-brazil": "pt-br",
    "portuguese-portugal": "pt-pt",
    "russian": "ru",
    "serbian": "sr",
    "slovak": "sk",
    "slovenian": "sl",
    "spanish-ecuador": "es-ec",
    "turkish": "tr",
    "uzbek": "uz",
}

def _to_expected_order_list_literal(order_csv: str) -> str:
    names = [x.strip() for x in str(order_csv).split(",") if x.strip()]
    return "[" + ", ".join(f"'{n}'" for n in names) + "]"

def main() -> None:
    if not _OUTPUTS.is_dir():
        raise SystemExit(f"Missing outputs directory: {_OUTPUTS}")

    required = ("compound", "expected_order", "sentence")
    n_ok = 0
    for pred in sorted(_OUTPUTS.glob("predictions_*.tsv")):
        stem_title = pred.stem.removeprefix("predictions_")
        key = stem_title.lower().replace(" ", "-")
        code = _STEM_TO_CODE.get(key)
        if not code:
            print(f"skip (unknown language stem): {pred.name!r} -> key={key!r}")
            continue

        with pred.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = reader.fieldnames or []
            missing = [c for c in required if c not in fieldnames]
            if missing:
                print(f"skip {pred.name}: missing columns {missing}")
                continue
            rows = list(reader)

        out = _OUTPUTS / f"submission_{code}.tsv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(required),
                delimiter="\t",
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "compound": row["compound"],
                        "sentence": row["sentence"],
                        "expected_order": _to_expected_order_list_literal(
                            row["expected_order"]
                        ),
                    }
                )

        print(f"wrote {out.name}  ({len(rows)} rows)  <-  {pred.name}")
        n_ok += 1

    print(f"\nDone. {n_ok} submission file(s) under {_OUTPUTS}")

if __name__ == "__main__":
    main()
