"""Evaluate every saved alpha-calibration result and print a compact summary."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evaluation.scored_labels import evaluate


ALPHA_FILE = re.compile(r"^calibrated_alpha_(\d{2})\.jsonl$")
EXPECTED_ALPHA_STEPS = set(range(1, 10))


def find_alpha_files(directory: Path) -> List[Tuple[int, Path]]:
    if not directory.is_dir():
        raise ValueError(f"Alpha sweep directory does not exist: {directory}")

    found: Dict[int, Path] = {}
    for path in directory.glob("calibrated_alpha_*.jsonl"):
        match = ALPHA_FILE.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected alpha result filename: {path.name}")
        step = int(match.group(1))
        if step in found:
            raise ValueError(f"Duplicate result for alpha={step / 10:.1f}")
        found[step] = path

    missing = sorted(EXPECTED_ALPHA_STEPS - found.keys())
    unexpected = sorted(found.keys() - EXPECTED_ALPHA_STEPS)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(f"{step / 10:.1f}" for step in missing))
        if unexpected:
            details.append("unexpected " + ", ".join(f"{step / 10:.1f}" for step in unexpected))
        raise ValueError("Expected alpha results from 0.1 through 0.9: " + "; ".join(details))

    return sorted(found.items())


def evaluate_alpha_sweep(
    directory: Path,
    gold_file: Path,
    percentile: float,
    *,
    details: bool = False,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for step, path in find_alpha_files(directory):
        alpha = step / 10
        if details:
            print(f"\n===== alpha={alpha:.1f} ({path.name}) =====")
        metrics = evaluate(
            gold_file=str(gold_file),
            pred_file=str(path),
            percentile=percentile,
            use_adjusted=True,
            verbose=details,
        )
        counts = metrics["counts"]
        if counts["gold"] != counts["pred"] or counts["gold"] != counts["aligned"]:
            raise ValueError(
                f"Incomplete predictions in {path}: "
                f"gold={counts['gold']}, pred={counts['pred']}, aligned={counts['aligned']}"
            )
        results.append({"alpha": alpha, "file": str(path), **metrics})
    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    print("\n===== Alpha sweep summary (without refinement) =====")
    print("alpha      tau  micro_f1  macro_f1  example_f1  avg_labels")
    for row in results:
        print(
            f"{row['alpha']:>5.1f}  "
            f"{row['tau']:>7.4f}  "
            f"{row['micro']['f1-score']:>8.4f}  "
            f"{row['macro']['f1-score']:>8.4f}  "
            f"{row['samples']['f1-score']:>10.4f}  "
            f"{row['avg_pred_labels']:>10.3f}"
        )


def serializable_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "alpha": row["alpha"],
            "file": row["file"],
            "q": row["percentile"],
            "tau": row["tau"],
            "micro_precision": row["micro"]["precision"],
            "micro_recall": row["micro"]["recall"],
            "micro_f1": row["micro"]["f1-score"],
            "macro_precision": row["macro"]["precision"],
            "macro_recall": row["macro"]["recall"],
            "macro_f1": row["macro"]["f1-score"],
            "example_precision": row["samples"]["precision"],
            "example_recall": row["samples"]["recall"],
            "example_f1": row["samples"]["f1-score"],
            "avg_pred_labels": row["avg_pred_labels"],
            "avg_gold_labels": row["avg_gold_labels"],
        }
        for row in results
    ]


def write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate calibrated_alpha_01.jsonl through calibrated_alpha_09.jsonl."
    )
    parser.add_argument("directory", type=Path, help="Directory containing the alpha sweep JSONL files.")
    parser.add_argument("--gold-file", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--percentile", type=float, default=0.99, help="Quantile q (default: %(default)s).")
    parser.add_argument("--json", type=Path, help="Optional path for the summary in JSON format.")
    parser.add_argument("--csv", type=Path, help="Optional path for the summary in CSV format.")
    parser.add_argument("--details", action="store_true", help="Also print the full report for every alpha.")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    try:
        results = evaluate_alpha_sweep(
            args.directory,
            args.gold_file,
            args.percentile,
            details=args.details,
        )
        print_summary(results)
        rows = serializable_results(results)
        if args.json:
            write_json(args.json, rows)
            print(f"\nWrote JSON summary to {args.json}")
        if args.csv:
            write_csv(args.csv, rows)
            print(f"Wrote CSV summary to {args.csv}")
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
