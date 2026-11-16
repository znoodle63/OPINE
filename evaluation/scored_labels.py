"""Decode and evaluate saved raw or AIMRaD-adjusted label scores."""

import argparse
from typing import Any, Dict, List, Set

from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

from evaluation.decoding import (
    quantile_threshold,
    load_gold_labels,
    load_pred_records,
    parse_adjusted_scores,
    decode_quantile,
    scores_from_record,
)
from taxonomy import OPINION_LABELS


def evaluate(
    gold_file: str,
    pred_file: str,
    percentile: float = 0.99,
    use_adjusted: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:

    rid2gold = load_gold_labels(gold_file)
    rid2pred = load_pred_records(pred_file)

    common_ids = sorted(set(rid2gold.keys()) & set(rid2pred.keys()))
    if verbose:
        print(f">>> #gold = {len(rid2gold)}, #pred = {len(rid2pred)}, #aligned = {len(common_ids)}")
        print(f">>> use_adjusted = {use_adjusted}")
        print()

    gold_sets: List[Set[str]] = []
    probs_list: List[Dict[str, float]] = []

    for rid in common_ids:
        gold_set = rid2gold[rid]
        rec = rid2pred[rid]

        rec_for_norm = rec
        if use_adjusted and rec.get("strategy") == "labelwise":
            raw_adj = rec.get("raw_output_adjusted")
            if raw_adj:
                adj_scores = parse_adjusted_scores(raw_adj)
                rec_for_norm = dict(rec_for_norm)
                rec_for_norm["label_scores"] = adj_scores

        probs = scores_from_record(rec_for_norm)

        gold_sets.append(gold_set)
        probs_list.append(probs)

    # Estimate one threshold over the split using only non-top scores.
    tau = quantile_threshold(
        probs_list,
        q=percentile,
        exclude_argmax=True,
    )
    if verbose:
        print(f">>> [QUANTILE] tau (q={percentile:.2f}, exclude_top1=True) = {tau:.4f}")

    pred_sets = [decode_quantile(probs, tau=tau) for probs in probs_list]

    avg_pred_k = sum(len(s) for s in pred_sets) / len(pred_sets) if pred_sets else 0.0
    avg_gold_k = sum(len(s) for s in gold_sets) / len(gold_sets) if gold_sets else 0.0
    if verbose:
        print(f">>> avg #pred labels = {avg_pred_k:.3f}, avg #gold labels = {avg_gold_k:.3f}")

    mlb = MultiLabelBinarizer(classes=OPINION_LABELS)
    Y_true = mlb.fit_transform(gold_sets)
    Y_pred = mlb.transform(pred_sets)

    report = classification_report(
        Y_true,
        Y_pred,
        target_names=OPINION_LABELS,
        zero_division=0,
        output_dict=True,
    )
    if verbose:
        print("\n===== Classification report (sklearn style) =====")
        print(
            classification_report(
                Y_true,
                Y_pred,
                target_names=OPINION_LABELS,
                zero_division=0,
                digits=4,
            )
        )

    return {
        "percentile": percentile,
        "tau": tau,
        "use_adjusted": use_adjusted,
        "counts": {
            "gold": len(rid2gold),
            "pred": len(rid2pred),
            "aligned": len(common_ids),
        },
        "avg_pred_labels": avg_pred_k,
        "avg_gold_labels": avg_gold_k,
        "micro": report["micro avg"],
        "macro": report["macro avg"],
        "samples": report["samples avg"],
    }


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gold_file", type=str, required=True,
                        help="Path to gold jsonl (with 'row_index' and 'label').")
    parser.add_argument("--pred_file", type=str, required=True,
                        help="Path to prediction jsonl produced by scoring.labels.")

    parser.add_argument("--percentile", type=float, default=0.99,
                        help="Quantile q for additional labels (default: %(default)s).")

    parser.add_argument("--use_adjusted", action="store_true",
                        help="If set, use AIMRaD-adjusted scores from 'raw_output_adjusted' "
                             "for labelwise strategy.")

    return parser.parse_args()


def main():
    args = get_args()

    print(">>> Eval config:")
    print(f"  gold_file    = {args.gold_file}")
    print(f"  pred_file    = {args.pred_file}")
    print(f"  percentile   = {args.percentile}")
    print(f"  use_adjusted = {args.use_adjusted}")
    print()

    evaluate(
        gold_file=args.gold_file,
        pred_file=args.pred_file,
        percentile=args.percentile,
        use_adjusted=args.use_adjusted,
    )


if __name__ == "__main__":
    main()
