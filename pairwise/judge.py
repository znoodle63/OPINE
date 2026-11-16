"""Collect model judgments for pairs of candidate labels."""

import json
from typing import Dict, Any, List, Tuple

from openai import OpenAI
from tqdm import tqdm

from scoring.arguments import get_args
from scoring.client import LLMConfig, build_llm_client, call_llm
from scoring.prompts import PAIRWISE_REFINE_PROMPT
from taxonomy import OPINION_ID2DESC
from scoring.parsing import build_instance_block, parse_label_scores, parse_pairwise_prob


def choose_base_scores(row: Dict[str, Any]) -> Dict[str, float]:
    """Use calibrated scores when present and raw label scores otherwise."""
    if "raw_output_adjusted" in row and row["raw_output_adjusted"]:
        return parse_label_scores(row["raw_output_adjusted"])
    if "label_scores" in row and row["label_scores"]:
        return {k: float(v) for k, v in row["label_scores"].items()}
    raise ValueError(
        "no base scores found: need at least one of `raw_output_adjusted` or `label_scores`"
    )


def select_anchor_and_opponents(
    scores: Dict[str, float],
) -> Tuple[str, List[str]]:
    """Use the highest-scoring label as the anchor and the runner-up as its opponent."""
    # Break score ties by label ID for deterministic candidate selection.
    sorted_labels = sorted(
        scores.items(), key=lambda kv: (-kv[1], kv[0])
    )
    anchor = sorted_labels[0][0]
    opponents = [lab for lab, _ in sorted_labels[1:2]]
    return anchor, opponents


def build_pairwise_prompt(
    row: Dict[str, Any],
    anchor: str,
    other: str,
    use_context: bool,
) -> str:
    instance_block = build_instance_block(row, use_context=use_context)
    labA_def = OPINION_ID2DESC.get(anchor, "")
    labB_def = OPINION_ID2DESC.get(other, "")

    user_prompt = PAIRWISE_REFINE_PROMPT.format(
        instance_block=instance_block,
        labA_id=anchor,
        labA_def=labA_def,
        labB_id=other,
        labB_def=labB_def,
    )
    return user_prompt


def pairwise_score_one_row(
    row: Dict[str, Any],
    client: OpenAI,
    cfg: LLMConfig,
    use_context: bool,
) -> Dict[str, Any]:
    base_scores = choose_base_scores(row)
    anchor, opponents = select_anchor_and_opponents(base_scores)

    pairwise_p: Dict[str, float] = {}
    pairwise_raw: Dict[str, str] = {}

    for y in opponents:
        try:
            user_prompt = build_pairwise_prompt(
                row=row,
                anchor=anchor,
                other=y,
                use_context=use_context,
            )
            raw = call_llm(client, cfg, user_prompt)
            p_val = parse_pairwise_prob(raw)
            pairwise_p[y] = p_val
            pairwise_raw[y] = raw
        except Exception as e:
            print(
                f"[PAIRWISE-ERROR] row_index={row.get('row_index')} "
                f"anchor={anchor} vs {y}: {e}"
            )

    out = {
        "row_index": row.get("row_index"),
        "paper_id": row.get("paper_id"),
        "strategy": row.get("strategy", "labelwise"),
        "base_scores": base_scores,
        "anchor": anchor,
        "topk_set": [anchor] + opponents,
        "pairwise_p": pairwise_p,      # y -> P(anchor ≻ y | x)
        "pairwise_raw": pairwise_raw,
    }
    return out


def main():
    args = get_args(pairwise=True)

    print(">>> Pairwise Top-2 Scoring Config:")
    print(f"  input        = {args.input}")
    print(f"  output       = {args.output}")
    print(f"  model        = {args.model}")
    print(f"  base_url     = {args.base_url}")
    print(f"  use_context  = {args.use_context}")
    print()

    cfg = LLMConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
    )
    client = build_llm_client(cfg)

    with open(args.input, "r", encoding="utf-8") as fin:
        lines = [ln for ln in fin if ln.strip()]

    total = len(lines)
    print(f">>> Total rows to pairwise-score: {total}")

    with open(args.output, "w", encoding="utf-8") as fout:
        for line in tqdm(lines, desc="Pairwise-Topk", total=total):
            row = json.loads(line)
            try:
                result = pairwise_score_one_row(
                    row=row,
                    client=client,
                    cfg=cfg,
                    use_context=args.use_context,
                )
            except Exception as e:
                print(f"[ERROR] row_index={row.get('row_index')}: {e}")
                result = {
                    "row_index": row.get("row_index"),
                    "paper_id": row.get("paper_id"),
                    "strategy": row.get("strategy", "labelwise"),
                    "error": str(e),
                }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
