import argparse
import copy
import json
from typing import Any, Dict

from taxonomy import CONFUSABLE_PAIRS


def fuse_confusable_margin_shrink(rec: Dict[str, Any]) -> Dict[str, float]:
    """Rescale the score margin of a confusable top-two pair by 2 * P(top1 ≻ top2)."""
    base = rec.get("base_scores") or rec.get("label_scores")
    if not base:
        raise ValueError(f"Record missing base_scores/label_scores: {rec}")

    fused = copy.deepcopy(base)

    labels = list(base.keys())
    if len(labels) < 2:
        return fused

    labels_sorted = sorted(labels, key=lambda lab: float(base[lab]), reverse=True)
    top1 = labels_sorted[0]
    top2 = labels_sorted[1]

    edge = frozenset([top1, top2])
    if edge not in CONFUSABLE_PAIRS:
        return fused

    pairwise_p: Dict[str, float] = rec.get("pairwise_p") or {}
    if top2 not in pairwise_p:
        return fused

    P_top1 = float(pairwise_p[top2])  # P(top1 ≻ top2)

    # Skip refinement when the pairwise judgment strongly supports the top label.
    if P_top1 >= 0.9:
        return fused

    # Adjust the margin while preserving the total score of the pair.
    s1 = float(base[top1])
    s2 = float(base[top2])
    S = s1 + s2
    if S <= 0:
        return fused

    m = s1 - s2
    if m <= 0:
        return fused

    # P = 0.5 leaves the margin unchanged.
    m_new = 2.0 * P_top1 * m

    s1_new = (S + m_new) / 2.0
    s2_new = (S - m_new) / 2.0

    fused[top1] = s1_new
    fused[top2] = s2_new

    return fused


def scores_to_raw_text(scores: Dict[str, float]) -> str:
    """Format scores as the LABEL_SCORES block expected by evaluation."""
    lines = ["LABEL_SCORES:"]
    for lab in sorted(scores.keys()):
        lines.append(f"{lab}: {scores[lab]:.4f}")
    return "\n".join(lines)


def process_file(in_path: str, out_path: str) -> None:
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            base_scores = rec.get("base_scores") or rec.get("label_scores")
            if base_scores is None:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            fused_scores = fuse_confusable_margin_shrink(rec)

            out_rec = dict(rec)
            out_rec["label_scores"] = base_scores
            out_rec["label_scores_refined"] = fused_scores
            out_rec["raw_output_adjusted"] = scores_to_raw_text(fused_scores)

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", type=str, required=True,
        help="JSONL file with base scores and pairwise judgments."
    )
    ap.add_argument(
        "--output", type=str, required=True,
        help="JSONL file with refined scores for evaluation."
    )
    args = ap.parse_args()

    process_file(args.input, args.output)


if __name__ == "__main__":
    main()
