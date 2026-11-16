"""Read, prepare, and decode scored JSONL predictions."""

import json
from typing import Any, Dict, List, Set

import numpy as np

from scoring.parsing import parse_pathwise_scores
from taxonomy import OPINION_LABELS, GROUP2LABELS


def parse_adjusted_scores(raw_text: str) -> Dict[str, float]:
    """Read label-score pairs from the adjusted LABEL_SCORES text."""
    if not raw_text:
        return {}

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    scores: Dict[str, float] = {}

    for i, line in enumerate(lines):
        if i == 0 and line.upper().startswith("LABEL_SCORES"):
            continue
        if ":" not in line:
            continue
        left, right = line.split(":", 1)
        label_id = left.strip()
        val_str = right.strip()
        try:
            val = float(val_str)
        except ValueError:
            continue
        scores[label_id] = val

    return scores


def load_gold_labels(path: str) -> Dict[int, Set[str]]:
    rid2gold: Dict[int, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = int(row["row_index"])
            label_str = (row.get("label") or "").strip()
            if not label_str:
                gold_set = set()
            else:
                gold_set = {lab.strip() for lab in label_str.split(";") if lab.strip()}
            rid2gold[rid] = gold_set
    return rid2gold


def load_pred_records(path: str) -> Dict[int, Dict[str, Any]]:
    rid2pred: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = int(rec["row_index"])
            rid2pred[rid] = rec
    return rid2pred


def quantile_threshold(
    probs_list: List[Dict[str, float]],
    q: float = 0.99,
    exclude_argmax: bool = True,
) -> float:
    """Compute a global quantile, excluding each record's top score by default."""
    bg: List[float] = []
    for probs in probs_list:
        if not probs:
            continue

        if exclude_argmax:
            top1 = max(probs.items(), key=lambda kv: kv[1])[0]
            for lab, p in probs.items():
                if lab == top1:
                    continue
                bg.append(float(p))
        else:
            bg.extend([float(p) for p in probs.values()])

    if not bg:
        return 1.0  # degenerate: never add extra labels

    tau = float(np.quantile(np.array(bg, dtype=float), float(q)))
    return tau


def decode_quantile(
    probs: Dict[str, float],
    tau: float,
) -> Set[str]:
    """Keep the top label and add any other label at or above the threshold."""
    if not probs:
        return set()

    top1 = max(probs.items(), key=lambda kv: kv[1])[0]
    pred = {top1}
    for lab, p in probs.items():
        if lab == top1:
            continue
        if float(p) >= float(tau):
            pred.add(lab)
    return pred


def _clip_01_over_labels(raw_scores: Dict[str, float],
                         all_labels=None) -> Dict[str, float]:
    """Clip each score independently to [0, 1], keeping its scale."""
    if all_labels is None:
        keys = list(raw_scores.keys())
    else:
        keys = list(all_labels)

    clipped: Dict[str, float] = {}
    for lab in keys:
        v = float(raw_scores.get(lab, 0.0))
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        clipped[lab] = v
    return clipped


def _combine_pathwise_scores(
    group_scores: Dict[str, float],
    label_in_group: Dict[str, float],
) -> Dict[str, float]:
    """Multiply clipped group and label scores without renormalizing."""
    P_grp = _clip_01_over_labels(group_scores)
    clipped_label_scores = _clip_01_over_labels(label_in_group, all_labels=OPINION_LABELS)

    tilde: Dict[str, float] = {}
    for g, labels in GROUP2LABELS.items():
        p_g = P_grp.get(g, 0.0)
        for lab in labels:
            s_y = float(clipped_label_scores.get(lab, 0.0))
            tilde[lab] = tilde.get(lab, 0.0) + p_g * s_y

    return tilde


def scores_from_record(rec: Dict[str, Any]) -> Dict[str, float]:
    """Prepare label scores using the record's scoring strategy."""
    strategy = rec.get("strategy")

    if strategy in ("labelwise", "listwise"):
        raw = rec.get("label_scores", {}) or {}
        probs = _clip_01_over_labels(raw, all_labels=OPINION_LABELS)

    elif strategy == "pathwise":
        group_scores = rec.get("group_scores", {}) or {}

        # Use the raw model response when parsed label scores are unavailable.
        label_in_group = rec.get("label_scores_in_group") or {}

        if not label_in_group:
            label_in_group = rec.get("label_scores", {}) or {}

        if not label_in_group:
            raw_output = rec.get("raw_output") or ""

            if raw_output:
                try:
                    g2, parsed_labels = parse_pathwise_scores(raw_output)

                    if not group_scores and g2:
                        group_scores = g2

                    label_in_group = parsed_labels

                except Exception as e:
                    print("[WARN] Could not parse path-wise scores:", e)

        # Record the scores used here for downstream inspection.
        rec["label_scores_in_group"] = label_in_group

        probs = _combine_pathwise_scores(group_scores, label_in_group)

    else:
        raise ValueError(f"Unknown strategy in record: {strategy}")

    return probs
