"""Generate and parse model scores for opinion labels."""

import json
from contextlib import nullcontext
from typing import Dict, Any, Optional

from openai import OpenAI
from tqdm import tqdm

from scoring.arguments import get_args
from scoring.client import LLMConfig, build_llm_client, call_llm
from scoring.prompts import (
    LABELWISE_SCORE_PROMPT,
    LISTWISE_SCORE_PROMPT,
    PATHWISE_SCORE_PROMPT,
)
from scoring.parsing import (
    build_instance_block,
    parse_label_scores,
    parse_pathwise_scores,
)


def score_one_row(
    row: Dict[str, Any],
    strategy: str,
    client: OpenAI,
    cfg: LLMConfig,
    use_context: bool,
) -> Dict[str, Any]:
    """Score one example and retain the model response even if parsing fails."""
    instance_block = build_instance_block(row, use_context=use_context)
    raw_text: Optional[str] = None

    try:
        if strategy in ("labelwise", "listwise"):
            template = LABELWISE_SCORE_PROMPT if strategy == "labelwise" else LISTWISE_SCORE_PROMPT
            user_prompt = template.format(instance_block=instance_block)
            raw_text = call_llm(client, cfg, user_prompt)
            label_scores = parse_label_scores(raw_text)
            result = {
                "row_index": row.get("row_index"),
                "paper_id": row.get("paper_id"),
                "strategy": strategy,
                "label_scores": label_scores,
                "raw_output": raw_text,
            }

        elif strategy == "pathwise":
            user_prompt = PATHWISE_SCORE_PROMPT.format(instance_block=instance_block)
            raw_text = call_llm(client, cfg, user_prompt)
            group_scores, label_in_group = parse_pathwise_scores(raw_text)
            result = {
                "row_index": row.get("row_index"),
                "paper_id": row.get("paper_id"),
                "strategy": strategy,
                "group_scores": group_scores,
                "label_scores_in_group": label_in_group,
                "raw_output": raw_text,
            }

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    except Exception as e:
        print(f"[ERROR] row_index={row.get('row_index')}: {e}")
        result = {
            "row_index": row.get("row_index"),
            "paper_id": row.get("paper_id"),
            "strategy": strategy,
            "error": str(e),
        }
        if raw_text is not None:
            result["raw_output"] = raw_text

    return result


def main():
    args = get_args()

    print(">>> Config:")
    print(f"  input        = {args.input}")
    print(f"  output       = {args.output}")
    print(f"  strategy     = {args.strategy}")
    print(f"  model        = {args.model}")
    print(f"  base_url     = {args.base_url}")
    print(f"  use_context  = {args.use_context}")
    raw_output_path = args.raw_output
    print(f"  raw_output   = {raw_output_path}")
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
    if args.max_rows is not None:
        total = min(total, args.max_rows)

    print(f">>> Total rows to score: {total}")

    raw_ctx = (
        open(raw_output_path, "w", encoding="utf-8")
        if raw_output_path
        else nullcontext()
    )

    with open(args.output, "w", encoding="utf-8") as fout, raw_ctx as fraw:
        for line in tqdm(lines[:total], desc="Scoring", total=total):
            row = json.loads(line)
            result = score_one_row(
                row=row,
                strategy=args.strategy,
                client=client,
                cfg=cfg,
                use_context=args.use_context,
            )

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            if raw_output_path:
                raw_rec = {
                    "row_index": result.get("row_index", row.get("row_index")),
                    "paper_id": result.get("paper_id", row.get("paper_id")),
                    "strategy": args.strategy,
                    "raw_output": result.get("raw_output", ""),
                }
                fraw.write(json.dumps(raw_rec, ensure_ascii=False) + "\n")
                fraw.flush()


if __name__ == "__main__":
    main()
