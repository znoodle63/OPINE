"""Evaluate generated labels or the section-prior-only baseline."""

import argparse
import json
import re
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
)

from taxonomy import OPINION_LABELS as VALID_LABELS


VALID_SET = set(VALID_LABELS)

# Extract label tokens from messy strings, e.g., "A.3, A.3-1" -> ["A.3-1"]
LABEL_RE = re.compile(r"\b([AB]\.\d-\d)\b", re.IGNORECASE)

# tokens to ignore if appear
BAD_TOKENS = {
    "none_class", "None_class", "NONE_CLASS",
    "none", "None", "null", "NULL", "na", "N/A"
}

# -----------------------------
# utils
# -----------------------------
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def norm_label(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    return s[0].upper() + s[1:]

def parse_labels(field: Any, keep_only_valid: bool = True) -> List[str]:
    """
    Robust parsing:
    - list or string
    - if messy string contains multiple things, extract AB.d-d tokens by regex
    - filter None_class etc
    - if keep_only_valid=True, keep only the 18 labels
    """
    if field is None:
        return []

    raw_items: List[str] = []
    if isinstance(field, list):
        raw_items = [str(x) for x in field if str(x).strip()]
    else:
        s = str(field).strip()
        if not s:
            return []
        raw_items = [s]

    found: List[str] = []

    for item in raw_items:
        item = item.strip()
        if not item:
            continue

        if item in BAD_TOKENS:
            continue

        # 1) regex extraction (handles "A.3, A.3-1")
        for m in LABEL_RE.findall(item):
            y = norm_label(m)
            if y in BAD_TOKENS:
                continue
            if keep_only_valid:
                if y in VALID_SET:
                    found.append(y)
            else:
                found.append(y)

        # 2) clean direct case
        clean = norm_label(item)
        if (not keep_only_valid) or (clean in VALID_SET):
            if clean in VALID_SET:
                found.append(clean)

    # dedup while preserving order
    out, seen = [], set()
    for y in found:
        if y and y not in seen:
            out.append(y)
            seen.add(y)
    return out

def make_key(obj: Dict[str, Any], strict_pair: bool) -> Tuple[str, str]:
    rid = obj.get("row_index")
    rid = str(int(rid)) if isinstance(rid, float) and rid.is_integer() else str(rid).strip()
    pid = str(obj.get("paper_id", "")).strip()
    return (pid, rid) if strict_pair else ("", rid)


def evaluate_generated_labels(args: argparse.Namespace) -> None:
    print(">>> Eval config:")
    print(f"  gold_file    = {args.test}")
    print(f"  pred_file    = {args.pred}")
    print(f"  strict_pair  = {args.strict_pair}")
    print(f"  gold_key     = {args.gold_key}")
    print(f"  pred_key     = {args.pred_key}")
    print(f"  digits       = {args.digits}")
    print()

    # 1) load gold (ALL test instances kept)
    gold_map: Dict[Tuple[str, str], List[str]] = {}
    n_test = 0
    dropped_gold = 0
    for r in read_jsonl(args.test):
        n_test += 1
        k = make_key(r, args.strict_pair)
        ys = parse_labels(r.get(args.gold_key), keep_only_valid=True)
        if not ys:
            # gold has no valid label -> drop (usually should not happen)
            dropped_gold += 1
            continue
        gold_map[k] = ys

    if not gold_map:
        raise ValueError("No gold labels loaded (after filtering to 18 labels). Check test file / gold_key.")

    # 2) load preds (DO NOT drop missing/invalid pred; treat as empty [])
    pred_map: Dict[Tuple[str, str], List[str]] = {}
    n_pred = 0
    dropped_pred_to_empty = 0
    for p in read_jsonl(args.pred):
        n_pred += 1
        k = make_key(p, args.strict_pair)

        if args.allow_multilabel_pred_key in p and p.get(args.allow_multilabel_pred_key) is not None:
            yp = parse_labels(p.get(args.allow_multilabel_pred_key), keep_only_valid=True)
        else:
            yp = parse_labels(p.get(args.pred_key), keep_only_valid=True)

        if not yp:
            dropped_pred_to_empty += 1
            yp = []  # ✅ keep sample, count as empty prediction

        pred_map[k] = yp

    # 3) align: use ALL gold keys; missing pred => []
    keys = list(gold_map.keys())
    missing_pred = sum(1 for k in keys if k not in pred_map)

    y_true = [gold_map[k] for k in keys]
    y_pred = [pred_map.get(k, []) for k in keys]

    print(f">>> #gold = {len(gold_map)}, #pred = {len(pred_map)}, #aligned = {len(keys)}")
    print(f">>> pred_missing_for_gold = {missing_pred}, pred_empty_after_filter = {dropped_pred_to_empty}")
    print()

    # 4) binarize (force 18 labels in fixed order)
    classes = VALID_LABELS
    mlb = MultiLabelBinarizer(classes=classes)
    Y_true = mlb.fit_transform(y_true)
    Y_pred = mlb.transform(y_pred)

    # sanity check: positives
    print(f">>> total gold positives = {int(np.sum(Y_true))}")
    print(f">>> total pred positives = {int(np.sum(Y_pred))}")
    print()

    # 5) quick summary (micro/macro/samples)
    for avg in ["micro", "macro", "samples"]:
        p, r, f, _ = precision_recall_fscore_support(
            Y_true, Y_pred, average=avg, zero_division=0
        )
        print(f"{avg.upper():7s}  P={p:.4f}  R={r:.4f}  F1={f:.4f}")

    subset_acc = accuracy_score(Y_true, Y_pred)
    print(f"\nSUBSET_ACC (exact set match) = {subset_acc:.4f}")

    # 6) sklearn report (ONLY 18 labels, no garbage)
    print("\n===== Classification report (sklearn style) =====")
    print(classification_report(
        Y_true,
        Y_pred,
        target_names=classes,
        digits=args.digits,
        zero_division=0
    ))

    print("\n--- Alignment summary ---")
    print(f"test rows read: {n_test}")
    print(f"gold instances kept (with valid label): {len(gold_map)}")
    print(f"gold instances dropped (no valid label): {dropped_gold}")
    print(f"pred rows read: {n_pred}")
    print(f"pred instances stored: {len(pred_map)}")
    print(f"aligned instances (== gold kept): {len(keys)}")
    print(f"missing predictions for gold: {missing_pred}")
    print(f"pred empty after filtering to 18 labels: {dropped_pred_to_empty}")


def load_priors(prior_path: str):
    """
    Calibration export:
    {
      "opinion_labels": [...],
      "section_priors": { "Abstract": {label: p, ...}, ... },
      "global_priors":  { label: p, ... }
    }
    """
    with open(prior_path, "r", encoding="utf-8") as f:
        pri = json.load(f)

    opinion_labels: List[str] = pri["opinion_labels"]
    section_priors: Dict[str, Dict[str, float]] = pri["section_priors"]
    global_priors: Dict[str, float] = pri["global_priors"]
    return opinion_labels, section_priors, global_priors


def parse_gold_labels(label_str: str) -> Set[str]:
    """
    gold label: "A.3-3;A.2-1" -> {"A.3-3","A.2-1"}
    """
    label_str = (label_str or "").strip()
    if not label_str:
        return set()
    return {x.strip() for x in label_str.split(";") if x.strip()}


def argmax_label(score_map: Dict[str, float], labels: List[str]) -> str:
    """Find the best label, treating missing scores as zero."""
    best_lab = labels[0]
    best_val = float(score_map.get(best_lab, 0.0))
    for lab in labels[1:]:
        v = float(score_map.get(lab, 0.0))
        if v > best_val:
            best_val = v
            best_lab = lab
    return best_lab


def evaluate_prior_only(gold_path: str, prior_path: str) -> None:
    opinion_labels, section_priors, global_priors = load_priors(prior_path)

    gold_sets: List[Set[str]] = []
    pred_sets: List[Set[str]] = []

    n = 0
    n_unknown_sec = 0

    with open(gold_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            sec = (obj.get("section_norm") or "UNKNOWN").strip() or "UNKNOWN"
            gold = parse_gold_labels(obj.get("label", ""))

            pri_map = section_priors.get(sec)
            if pri_map is None:
                pri_map = global_priors
                n_unknown_sec += 1

            pred_top1 = argmax_label(pri_map, opinion_labels)

            gold_sets.append(gold)
            pred_sets.append({pred_top1})

            n += 1

    mlb = MultiLabelBinarizer(classes=opinion_labels)
    Y_true = mlb.fit_transform(gold_sets)
    Y_pred = mlb.transform(pred_sets)

    print(f">>> #samples = {n}")
    print(f">>> #unknown_section_fallback_to_global = {n_unknown_sec}")
    print("\n===== Prior-only baseline (Top-1 by section) =====")
    print(
        classification_report(
            Y_true,
            Y_pred,
            target_names=opinion_labels,
            zero_division=0,
            digits=4,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="./evaluate.sh", description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    ap = modes.add_parser("label-generation", help="Evaluate saved generated labels.")
    ap.add_argument("--test", required=True, help="test.jsonl (gold)")
    ap.add_argument("--pred", required=True, help="pred.jsonl (pred)")
    ap.add_argument("--strict_pair", action="store_true",
                    help="Align by (paper_id,row_index). Recommended.")
    ap.add_argument("--gold_key", default="label", help="gold label field name in test.jsonl")
    ap.add_argument("--pred_key", default="pred_label", help="predicted label field name in pred.jsonl")
    ap.add_argument("--allow_multilabel_pred_key", default="pred_labels",
                    help="optional field name for multi-label predictions if you store as list/string")
    ap.add_argument("--digits", type=int, default=4, help="digits for sklearn report")
    prior = modes.add_parser("prior-only", help="Evaluate the section-prior baseline.")
    prior.add_argument("--gold_file", default="data/test.jsonl", help="Test labels and sections in JSONL format.")
    prior.add_argument("--prior_file", required=True, help="JSON priors written by the calibration stage.")

    args = parser.parse_args()
    if args.mode == "label-generation":
        evaluate_generated_labels(args)
    else:
        evaluate_prior_only(args.gold_file, args.prior_file)


if __name__ == "__main__":
    main()
