"""Command-line arguments shared by label scoring and pairwise judging."""

import argparse
import os


def get_args(*, pairwise=False):
    description = "Judge pairs of candidate labels." if pairwise else "Score opinion labels."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, metavar="FILE", help="Input JSONL file.")
    parser.add_argument("--output", required=True, metavar="FILE", help="Scores to write as JSONL.")
    parser.add_argument("--model", required=True, help="Model ID at the selected endpoint.")
    parser.add_argument("--base_url", required=True, help="OpenAI-compatible API base URL.")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature (default: %(default)s).")
    context = parser.add_mutually_exclusive_group()
    context.add_argument("--use_context", action="store_true",
                         help="Include local sentence context (default).")
    context.add_argument("--no_context", dest="use_context", action="store_false",
                         help="Score the sentence without local context.")
    parser.set_defaults(use_context=True)

    if not pairwise:
        parser.add_argument("--strategy", default="labelwise", choices=("labelwise", "listwise", "pathwise"),
                            help="Label scoring strategy.")
        parser.add_argument("--raw_output", metavar="FILE", help="Optional separate JSONL of model responses.")
        parser.add_argument("--max_rows", type=int, help="Process at most this many input rows.")

    args = parser.parse_args()
    args.api_key = os.environ.get("OPINE_API_KEY")
    if not args.api_key:
        parser.error("Set OPINE_API_KEY in the environment.")
    return args
