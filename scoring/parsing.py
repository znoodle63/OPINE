import re
from typing import Dict, Tuple

from scoring.prompts import INSTANCE_BLOCK_TEMPLATE
from taxonomy import OPINION_LABELS


PAIRWISE_PATTERN = re.compile(r"P_A:\s*([0-1](?:\.\d+)?)")


def build_instance_block(row: dict, use_context: bool = False) -> str:
    sentence = (row.get("sentence") or "").strip()

    if use_context:
        context = (row.get("context") or "").strip()
        if not context:
            context = "[none]"
    else:
        context = "[none]"

    return INSTANCE_BLOCK_TEMPLATE.format(sentence=sentence, context=context)


def parse_label_scores(text: str) -> Dict[str, float]:
    """Read label-score rows after the LABEL_SCORES header."""
    if not text or not text.strip():
        raise ValueError("Empty LLM output for label scores")

    raw_lines = text.splitlines()

    start_idx = None
    for i, ln in enumerate(raw_lines):
        if ln.strip().upper().startswith("LABEL_SCORES"):
            start_idx = i
            break

    if start_idx is None:
        print("[WARN] LABEL_SCORES header not found.")
        return {}

    lines = [ln.strip() for ln in raw_lines[start_idx:] if ln.strip()]

    scores: Dict[str, float] = {}

    for line in lines[1:]:
        if ": " not in line:
            continue
        label_id, val_str = line.split(": ", 1)
        label_id = label_id.strip()
        if label_id not in OPINION_LABELS:
            continue

        # Ignore any explanation after the numeric score.
        val_token = val_str.strip().split()[0]
        try:
            score = float(val_token)
        except ValueError:
            continue

        scores[label_id] = score

    missing = [lab for lab in OPINION_LABELS if lab not in scores]
    if missing:
        print(f"[WARN] missing label scores for {len(missing)} labels, e.g.: {missing[:5]}")

    return scores


def parse_pathwise_scores(text: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Read the GROUP_SCORES and flattened LABEL_SCORES blocks."""
    if not text or not text.strip():
        raise ValueError("Empty LLM output for path-wise scores")

    raw_lines = text.splitlines()

    gs_start = None
    for i, ln in enumerate(raw_lines):
        if ln.strip().upper().startswith("GROUP_SCORES"):
            gs_start = i
            break
    if gs_start is None:
        print("[WARN] GROUP_SCORES header not found.")
        return {}, {}

    lines = raw_lines[gs_start:]

    i = 1  # Skip the GROUP_SCORES header.

    group_scores: Dict[str, float] = {}

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line.upper().startswith("LABEL_SCORES"):
            break

        if ": " in line:
            gid, val_str = line.split(": ", 1)
            gid = gid.strip()
            val_token = val_str.strip().split()[0]
            try:
                group_scores[gid] = float(val_token)
            except ValueError:
                pass
        i += 1

    if i == len(lines):
        print("[WARN] LABEL_SCORES block not found in path-wise output.")
        return group_scores, {}

    i += 1  # Skip the LABEL_SCORES header.

    label_scores: Dict[str, float] = {}

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line:
            continue
        if ": " not in line:
            continue

        lid_raw, val_str = line.split(": ", 1)
        lid = lid_raw.strip().split()[0]

        if lid not in OPINION_LABELS:
            continue

        val_token = val_str.strip().split()[0]
        try:
            score = float(val_token)
        except ValueError:
            continue

        label_scores[lid] = score

    missing = [lab for lab in OPINION_LABELS if lab not in label_scores]
    if missing:
        print(f"[WARN] Path-wise: missing scores for {len(missing)} labels, e.g. {missing[:5]}")

    return group_scores, label_scores


def parse_pairwise_prob(text: str) -> float:
    """Read the probability that label A is preferred from the P_A field."""
    if not text or not text.strip():
        raise ValueError("Empty LLM output for pairwise prob")

    m = PAIRWISE_PATTERN.search(text)
    if not m:
        print("[WARN] Cannot find 'P_A:' in pairwise output.")
        raise ValueError("Cannot find P_A in pairwise output.")

    p_val = float(m.group(1))
    if p_val < 0.0:
        p_val = 0.0
    if p_val > 1.0:
        p_val = 1.0
    return p_val
