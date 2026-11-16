"""Calibrate label-wise LLM scores with AIMRaD priors learned on the training split."""

import argparse
import json
import math
from collections import defaultdict, Counter
from contextlib import ExitStack, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Tuple

from taxonomy import OPINION_LABELS


ALPHA_SWEEP = tuple(step / 10 for step in range(1, 10))


def parse_label_scores(raw_output: str) -> Dict[str, float]:
    """Read LABEL_SCORES, including when preceded by GROUP_SCORES."""
    if not raw_output:
        return {}

    lines = [ln.strip() for ln in raw_output.strip().splitlines()]
    scores: Dict[str, float] = {}

    try:
        start_idx = next(
            i for i, ln in enumerate(lines)
            if ln.replace(" ", "") == "LABEL_SCORES:"
        )
    except StopIteration:
        raise ValueError("raw_output does not contain a 'LABEL_SCORES:' block")

    for line in lines[start_idx + 1:]:
        if not line:
            continue
        if line.endswith(":") and ":" not in line[:-1]:
            break
        if ":" not in line:
            continue

        label_id, score_str = line.split(":", 1)
        label_id = label_id.strip()
        score_str = score_str.strip()

        try:
            score = float(score_str)
        except ValueError:
            continue

        scores[label_id] = score

    return scores


def format_label_scores(scores: Dict[str, float]) -> str:
    """Write LABEL_SCORES in taxonomy order with four decimal places."""
    lines = ["LABEL_SCORES:"]
    for label_id in OPINION_LABELS:
        v = scores.get(label_id, 0.0)
        lines.append(f"{label_id}: {v:.4f}")
    return "\n".join(lines)


def compute_section_label_priors(
    gold_path: str,
    section_field: str = "section_norm",
    label_field: str = "label",
    smoothing: float = 1.0,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Estimate P(label | section) and P(label) from training annotations."""
    section_label_counts: Dict[str, Counter] = defaultdict(Counter)
    global_label_counts: Counter = Counter()

    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sec = obj.get(section_field, "UNKNOWN") or "UNKNOWN"
            label_str = (obj.get(label_field) or "").strip()
            if not label_str:
                continue

            labels = [x.strip() for x in label_str.split(";") if x.strip()]
            for lab in labels:
                if lab not in OPINION_LABELS:
                    continue
                section_label_counts[sec][lab] += 1
                global_label_counts[lab] += 1

    num_labels = len(OPINION_LABELS)

    total_global = sum(global_label_counts.values()) + smoothing * num_labels
    global_priors: Dict[str, float] = {}
    for lab in OPINION_LABELS:
        global_priors[lab] = (global_label_counts[lab] + smoothing) / total_global

    section_priors: Dict[str, Dict[str, float]] = {}
    for sec, cnt in section_label_counts.items():
        total = sum(cnt.values()) + smoothing * num_labels
        section_priors[sec] = {lab: (cnt[lab] + smoothing) / total for lab in OPINION_LABELS}

    return section_priors, global_priors


def fuse_scores_with_prior(
    llm_scores: Dict[str, float],
    section: str,
    section_priors: Dict[str, Dict[str, float]],
    global_priors: Dict[str, float],
    alpha: float = 0.8,
) -> Dict[str, float]:
    """Normalize P_text^alpha * P_prior^(1-alpha) over the label set."""
    eps = 1e-12
    fused_unnorm: Dict[str, float] = {}

    priors_for_sec = section_priors.get(section)

    for lab in OPINION_LABELS:
        p_text = float(llm_scores.get(lab, 0.0))
        if priors_for_sec is not None:
            p_prior = float(priors_for_sec.get(lab, global_priors.get(lab, 1.0 / len(OPINION_LABELS))))
        else:
            p_prior = float(global_priors.get(lab, 1.0 / len(OPINION_LABELS)))

        p_text = max(eps, p_text)
        p_prior = max(eps, p_prior)

        fused_unnorm[lab] = math.exp(alpha * math.log(p_text) + (1.0 - alpha) * math.log(p_prior))

    Z = sum(fused_unnorm.values())
    if Z <= 0.0 or not math.isfinite(Z):
        return {lab: 1.0 / len(OPINION_LABELS) for lab in OPINION_LABELS}

    return {lab: fused_unnorm[lab] / Z for lab in OPINION_LABELS}


def build_rowindex_to_section(gold_path: str, section_field: str = "section_norm") -> Dict[int, str]:
    """Read section metadata by row index from the split being calibrated."""
    mapping: Dict[int, str] = {}
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            idx = obj.get("row_index")
            if idx is None:
                continue
            sec = obj.get(section_field, "UNKNOWN") or "UNKNOWN"
            mapping[int(idx)] = sec
    return mapping


def parse_alpha(value: str):
    if value == "all":
        return value
    try:
        return float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a number or 'all'") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--train_gold_path", type=str, required=True,
                        help="Training annotations in JSONL for estimating section priors.")

    parser.add_argument("--gold_path", type=str, required=True,
                        help="JSONL for the split being calibrated, used to look up each row's section.")

    parser.add_argument("--pred_path", type=str, required=True,
                        help="Prediction JSONL containing raw_output for the same split.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output JSONL path. With --alpha all, add _alpha_01 through _alpha_09 before the extension.")

    parser.add_argument("--section_field", type=str, default="section_norm",
                        help="Section field in both input datasets (default: %(default)s).")
    parser.add_argument("--label_field", type=str, default="label",
                        help="Label field in the training data (default: %(default)s).")
    parser.add_argument("--smoothing", type=float, default=1.0,
                        help="Laplace smoothing for estimating the priors (default: %(default)s).")
    parser.add_argument("--alpha", type=parse_alpha, default=0.8,
                        help="Text score exponent (default: %(default)s), or 'all' for 0.1 through 0.9.")
    parser.add_argument("--adjusted_key", type=str, default="raw_output_adjusted",
                        help="Output field for calibrated scores (default: %(default)s).")

    parser.add_argument("--prior_json_path", type=str, default=None,
                        help="Optional output path for the section and global priors in JSON.")

    args = parser.parse_args()
    output_path = Path(args.output_path)
    sweep = args.alpha == "all"
    if sweep:
        outputs = {
            alpha: output_path.with_name(f"{output_path.stem}_alpha_{step:02d}{output_path.suffix}")
            for step, alpha in enumerate(ALPHA_SWEEP, start=1)
        }
        destinations = set(outputs.values())
        if args.prior_json_path:
            destinations.add(Path(args.prior_json_path))
        if Path(args.pred_path).resolve() in {path.resolve() for path in destinations}:
            parser.error("Sweep outputs must be separate from the saved label scores.")
    else:
        outputs = {args.alpha: output_path}

    print(">>> Config:")
    print(f"  train_gold_path = {args.train_gold_path}")
    print(f"  gold_path       = {args.gold_path}")
    print(f"  pred_path       = {args.pred_path}")
    print(f"  output_path     = {args.output_path}")
    print(f"  section_field   = {args.section_field}")
    print(f"  label_field     = {args.label_field}")
    print(f"  smoothing       = {args.smoothing}")
    print(f"  alpha           = {args.alpha}")
    print(f"  adjusted_key    = {args.adjusted_key}")
    print(f"  prior_json_path = {args.prior_json_path}")
    print()

    print(">> Step 1: Estimating TRAIN AIMRaD priors...")
    section_priors, global_priors = compute_section_label_priors(
        gold_path=args.train_gold_path,
        section_field=args.section_field,
        label_field=args.label_field,
        smoothing=args.smoothing,
    )

    if args.prior_json_path:
        priors_obj = {
            "opinion_labels": OPINION_LABELS,
            "section_priors": section_priors,
            "global_priors": global_priors,
            "meta": {
                "train_gold_path": args.train_gold_path,
                "section_field": args.section_field,
                "label_field": args.label_field,
                "smoothing": args.smoothing,
            }
        }
        with open(args.prior_json_path, "w", encoding="utf-8") as f:
            json.dump(priors_obj, f, ensure_ascii=False, indent=2)
        print(f">> Wrote priors to: {args.prior_json_path}")

    print(">> Step 2: Building the row_index -> section mapping from the current gold split...")
    idx2sec = build_rowindex_to_section(
        gold_path=args.gold_path,
        section_field=args.section_field,
    )

    print(">> Step 3: Reading predictions and fusing scores...")
    num_lines = 0
    num_no_sec = 0
    num_unknown_section_not_in_train = 0

    staging = TemporaryDirectory(prefix=".calibration-", dir=output_path.parent) if sweep else nullcontext()
    with staging as directory:
        write_paths = outputs
        if directory is not None:
            write_paths = {alpha: Path(directory) / path.name for alpha, path in outputs.items()}
        with open(args.pred_path, "r", encoding="utf-8") as fin, ExitStack() as stack:
            writers = {
                alpha: stack.enter_context(path.open("w", encoding="utf-8"))
                for alpha, path in write_paths.items()
            }
            for line in fin:
                if not line.strip():
                    continue
                obj = json.loads(line)
                num_lines += 1

                row_idx = obj.get("row_index")
                if row_idx is None:
                    continue
                row_idx = int(row_idx)

                sec = idx2sec.get(row_idx, "UNKNOWN")
                if sec == "UNKNOWN":
                    num_no_sec += 1

                if sec != "UNKNOWN" and sec not in section_priors:
                    num_unknown_section_not_in_train += 1

                raw_output = obj.get("raw_output", "")
                llm_scores = parse_label_scores(raw_output)

                for alpha, fout in writers.items():
                    fused_scores = fuse_scores_with_prior(
                        llm_scores=llm_scores,
                        section=sec,
                        section_priors=section_priors,
                        global_priors=global_priors,
                        alpha=alpha,
                    )
                    obj[args.adjusted_key] = format_label_scores(fused_scores)
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

        if directory is not None:
            for alpha, path in outputs.items():
                write_paths[alpha].replace(path)

    print(f">> Done. Processed {num_lines} predictions.")
    print(f">> {num_no_sec} rows had section=UNKNOWN and fell back to the global prior.")
    print(
        f">> {num_unknown_section_not_in_train} rows had a section unseen in TRAIN "
        "and fell back to the global prior."
    )
    for path in outputs.values() if sweep else (args.output_path,):
        print(f">> Output file: {path}")


if __name__ == "__main__":
    main()
