#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./evaluate.sh predictions FILE
  ./evaluate.sh alpha-sweep DIRECTORY [--percentile Q] [--details] \
      [--json FILE] [--csv FILE] [--gold-file FILE]
  ./evaluate.sh ablations --full FILE --without-prior FILE \
      --without-refinement FILE --without-both FILE [--gold-file FILE]
  ./evaluate.sh label-generation --test FILE --pred FILE
  ./evaluate.sh prior-only --prior_file FILE [--gold_file FILE]

Evaluate saved predictions without model access. Set PYTHON to choose a Python executable.
Relative paths are resolved from the repository root, except FILE in predictions mode
and DIRECTORY in alpha-sweep mode, which are resolved from the caller's directory.
EOF
}

python_cmd="${PYTHON:-python3}"
caller_dir="$PWD"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

case "${1:-}" in
  predictions)
    if [[ $# -ne 2 ]]; then
      usage >&2
      exit 2
    fi
    prediction="$2"
    if [[ "$prediction" != /* ]]; then
      prediction="$caller_dir/$prediction"
    fi
    if [[ ! -f "$prediction" || ! -r "$prediction" ]]; then
      printf 'Cannot read file: %s\n' "$prediction" >&2
      exit 1
    fi
    exec "$python_cmd" -m evaluation.scored_labels \
      --gold_file data/test.jsonl --pred_file "$prediction" \
      --percentile 0.99 --use_adjusted
    ;;
  alpha-sweep)
    if [[ $# -lt 2 ]]; then
      usage >&2
      exit 2
    fi
    sweep_directory="$2"
    shift 2
    if [[ "$sweep_directory" != /* ]]; then
      sweep_directory="$caller_dir/$sweep_directory"
    fi
    exec "$python_cmd" -m evaluation.alpha_sweep "$sweep_directory" "$@"
    ;;
  label-generation|prior-only)
    exec "$python_cmd" -m evaluation.baselines "$@"
    ;;
  ablations)
    shift
    ;;
  --help|-h)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

gold_file=data/test.jsonl
full=''
without_prior=''
without_refinement=''
without_both=''

while [[ $# -gt 0 ]]; do
  if [[ "$1" == --help || "$1" == -h ]]; then
    usage
    exit 0
  fi
  if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
    printf 'Expected a file after %s\n' "$1" >&2
    exit 2
  fi
  case "$1" in
    --gold-file) gold_file="$2" ;;
    --full) full="$2" ;;
    --without-prior) without_prior="$2" ;;
    --without-refinement) without_refinement="$2" ;;
    --without-both) without_both="$2" ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift 2
done

if [[ -z "$full" || -z "$without_prior" || -z "$without_refinement" || -z "$without_both" ]]; then
  usage >&2
  exit 2
fi
for file in "$gold_file" "$full" "$without_prior" "$without_refinement" "$without_both"; do
  if [[ ! -f "$file" || ! -r "$file" ]]; then
    printf 'Cannot read file: %s\n' "$file" >&2
    exit 1
  fi
done

evaluate_variant() {
  printf '\n===== %s =====\n' "$1"
  "$python_cmd" -m evaluation.scored_labels \
    --gold_file "$gold_file" --pred_file "$2" \
    --percentile 0.99 --use_adjusted
}

evaluate_variant OPINE "$full"
evaluate_variant 'w/o Prior' "$without_prior"
evaluate_variant 'w/o Refine' "$without_refinement"
evaluate_variant 'w/o All' "$without_both"
