"""Run OPINE, ablations, and scoring baselines with shared label scores."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from calibration import ALPHA_SWEEP
from taxonomy import OPINION_LABELS


TRAIN = Path("data/train.jsonl")
TEST = Path("data/test.jsonl")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_directory(model: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip(".-")[:80]
    return f"{name or 'model'}-{hashlib.sha256(model.encode()).hexdigest()[:10]}"


def prediction_key(row):
    return str(row["paper_id"]), int(row["row_index"])


def prediction_rows(path: Path):
    keys = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = prediction_key(row)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid prediction row {line_number} in {path}.") from error
            if key in keys:
                raise ValueError(f"Duplicate prediction for row {key[1]} in {path}")
            keys.add(key)
            yield key, row


def label_score_error(row, *, strategy="labelwise", require_all_labels=False):
    scores = row.get("label_scores_in_group" if strategy == "pathwise" else "label_scores")
    if row.get("error") or not scores:
        return "Incomplete prediction"
    if (
        row.get("strategy") != strategy
        or not isinstance(scores, dict)
        or not isinstance(row.get("raw_output"), str)
        or not row["raw_output"].strip()
        or row.get("raw_output_adjusted")
    ):
        return f"Expected uncalibrated {strategy} scores"
    if strategy == "pathwise" and (not isinstance(row.get("group_scores"), dict) or not row["group_scores"]):
        return "Missing pathwise group scores"
    if require_all_labels and (
        scores.keys() != set(OPINION_LABELS)
        or any(not 0 <= score <= 1 for score in scores.values())
    ):
        return f"Expected a score in [0, 1] for each of the {len(OPINION_LABELS)} labels"
    return None


def inspect_label_scores(path: Path, *, require_all_labels=False):
    keys = set()
    error = None
    for key, row in prediction_rows(path):
        keys.add(key)
        if error is None:
            detail = label_score_error(row, require_all_labels=require_all_labels)
            if detail is not None:
                error = f"{detail} for row {key[1]} in {path}."
    return keys, error


def valid_pairwise_row(row, input_row):
    from pairwise.judge import choose_base_scores, select_anchor_and_opponents

    if row.get("error"):
        return False
    try:
        base = choose_base_scores(input_row)
        anchor, opponents = select_anchor_and_opponents(base)
    except (ValueError, TypeError, IndexError, AttributeError):
        return False

    recorded_base = row.get("base_scores")
    probabilities = row.get("pairwise_p")
    return (
        isinstance(recorded_base, dict)
        and list(recorded_base.items()) == list(base.items())
        and row.get("anchor") == anchor
        and row.get("topk_set") == [anchor, *opponents]
        and isinstance(probabilities, dict)
        and probabilities.keys() == set(opponents)
        and all(type(p) in (int, float) and 0 <= p <= 1 for p in probabilities.values())
    )


def read_model_attempt(path, inputs, completed, valid_row):
    with path.open("rb") as source:
        for line in source:
            # A process interrupted while writing can leave an unfinished final row.
            if not line.endswith(b"\n"):
                break
            try:
                row = json.loads(line)
                key = prediction_key(row)
            except (ValueError, TypeError, KeyError):
                continue
            if key in inputs and key not in completed and valid_row(row, inputs[key]):
                completed[key] = row


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


class Workflow:
    def __init__(
        self,
        directory: Path,
        state_directory: Path,
        model: str,
        endpoint: str,
        *,
        use_context: bool = True,
    ):
        self.directory = directory
        self.state_directory = state_directory
        self.cache_directory = state_directory / "cache"
        self.checkpoint_directory = state_directory / "checkpoints"
        self.staging_directory = state_directory / "staging"
        self.model = model
        self.endpoint = endpoint
        context_flag = "--use_context" if use_context else "--no_context"
        self.scoring_args = (context_flag, "--temperature", "0.2")
        root = Path(__file__).parent
        paths = [root / name for name in ("calibration.py", "taxonomy.py", "workflow.py")]
        for package in ("scoring", "pairwise", "evaluation"):
            paths.extend((root / package).rglob("*.py"))
        self.source_digests = {path.relative_to(root).as_posix(): file_digest(path) for path in paths}
        self.test_rows = dict(prediction_rows(TEST))

    def output_path(self, *parts: str) -> Path:
        return self.directory.joinpath(*parts)

    def cache_path(self, primary: Path) -> Path:
        """Mirror an output's relative path under the private cache directory."""
        relative = primary.relative_to(self.directory)
        return (self.cache_directory / relative).with_suffix(".cache.json")

    def label_scoring_args(self, input_path=TEST, *, strategy="labelwise"):
        return ("--strategy", strategy, *self.scoring_args, "--input", input_path)

    def model_options(self):
        if not os.environ.get("OPINE_API_KEY") or not self.endpoint:
            raise ValueError("Set base_url in the configuration and OPINE_API_KEY in the environment to call the model.")
        return ["--model", self.model, "--base_url", self.endpoint]

    def check_api(self):
        options = self.model_options()
        with TemporaryDirectory(prefix="opine-api-check-") as temporary:
            output = Path(temporary) / "label_scores.jsonl"
            command = [sys.executable, "-m", "scoring.labels", *map(str, self.label_scoring_args()),
                       *options, "--max_rows", "1", "--output", str(output)]
            subprocess.run(command, check=True)
            keys, error = inspect_label_scores(output, require_all_labels=True)
            if error is not None:
                raise ValueError(f"API check failed. {error}")
            if len(keys) != 1:
                raise ValueError("API check did not return usable label scores. See the scoring output above.")
        print("API check passed for one test example.")

    def finish_model_stage(self, module, args, pairwise_input, checkpoint, staging, staged, *, scoring_strategy):
        if scoring_strategy is not None:
            original_input, inputs = TEST, self.test_rows
            stage_name = "Label scoring"

            def valid_row(row, _input_row):
                return label_score_error(row, strategy=scoring_strategy) is None
        else:
            original_input = pairwise_input
            inputs = dict(prediction_rows(original_input))
            if inputs.keys() != self.test_rows.keys():
                raise ValueError(f"{original_input} does not cover the complete test split.")
            stage_name = "Pairwise judging"
            valid_row = valid_pairwise_row

        attempts = sorted(checkpoint.glob("attempt-*.jsonl"))
        completed = {}
        for path in attempts:
            read_model_attempt(path, inputs, completed, valid_row)

        pending = inputs.keys() - completed.keys()
        if pending:
            options = self.model_options()
            checkpoint.mkdir(parents=True, exist_ok=True)
            number = int(attempts[-1].stem.rsplit("-", 1)[-1]) + 1 if attempts else 1
            attempt = checkpoint / f"attempt-{number:08d}.jsonl"
            input_path = original_input
            if completed:
                input_path = staging / "pending_rows.jsonl"
                write_jsonl(input_path, (row for key, row in inputs.items() if key in pending))
            if scoring_strategy is not None:
                stage_args = self.label_scoring_args(input_path, strategy=scoring_strategy)
            else:
                stage_args = (*args, "--input", input_path)
            print(f"{stage_name}: {len(completed)} complete, {len(pending)} pending.", file=sys.stderr)
            command = [sys.executable, "-m", module, *map(str, stage_args), *options, "--output", str(attempt)]
            attempt.touch(exist_ok=False)
            subprocess.run(command, check=True)
            read_model_attempt(attempt, inputs, completed, valid_row)

        if completed.keys() != inputs.keys():
            remaining = len(inputs) - len(completed)
            raise ValueError(
                f"{stage_name} has {remaining} failed or unfinished rows. Progress saved in {checkpoint}. "
                "Run the same command again to resume."
            )

        write_jsonl(staged["--output"], (completed[key] for key in inputs))
        if "--raw_output" in staged:
            fields = ("row_index", "paper_id", "strategy", "raw_output")
            write_jsonl(staged["--raw_output"], ({field: completed[key][field] for field in fields} for key in inputs))

    def run_stage(
        self, module, args, inputs, outputs, *, pairwise_input=None, saved_only=False, scoring_strategy=None,
    ):
        """Keep completed model calls reusable across changes to the surrounding workflow."""
        primary = next(iter(outputs.values()))
        for path in outputs.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        receipt = self.cache_path(primary)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        self.staging_directory.mkdir(parents=True, exist_ok=True)
        label_scoring = scoring_strategy is not None
        source_identity = self.source_digests
        if label_scoring:
            source_identity = "label-scores-v1"
        elif pairwise_input is not None:
            source_identity = "pairwise-judgments-v1"
        settings = [module, [str(arg) for arg in args], source_identity]
        if pairwise_input is not None:
            settings += [file_digest(path) for path in inputs]
        else:
            settings += [[str(path), file_digest(path)] for path in inputs]
        if label_scoring or pairwise_input is not None:
            settings += [self.model, self.endpoint]
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

        try:
            cached = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = {}
        if not isinstance(cached, dict):
            cached = {}
        artifact_digests = cached.get("artifacts")
        if not isinstance(artifact_digests, dict):
            artifact_digests = {}
        if (
            cached.get("key") == key
            and all(path.is_file() and file_digest(path) == artifact_digests.get(path.name)
                    for path in outputs.values())
        ):
            print(f"Reusing {primary}", file=sys.stderr)
            return primary

        if saved_only:
            raise ValueError("No reusable label scores found. Run ./run_opine.sh with the same configuration first.")

        with TemporaryDirectory(prefix="stage-", dir=self.staging_directory) as temporary:
            staging = Path(temporary)
            staged = {flag: staging / path.name for flag, path in outputs.items()}
            print(f"Generating {primary}", file=sys.stderr)
            checkpoint = None
            if label_scoring:
                checkpoint = self.checkpoint_directory / "label" / key
            elif pairwise_input is not None:
                relative_stem = primary.relative_to(self.directory).with_suffix("")
                checkpoint = self.checkpoint_directory / "pairwise" / relative_stem / key
            if checkpoint is not None:
                self.finish_model_stage(module, args, pairwise_input, checkpoint, staging, staged,
                                        scoring_strategy=scoring_strategy)
            else:
                command = [sys.executable, "-m", module, *map(str, args)]
                for flag, path in staged.items():
                    command.extend((flag, str(path)))
                subprocess.run(command, check=True)

            for path in staged.values():
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"{module} did not produce {path.name}")

            receipt.unlink(missing_ok=True)
            for flag, path in staged.items():
                path.replace(outputs[flag])
            artifacts = {outputs[flag].name: file_digest(outputs[flag]) for flag in staged}
            temporary_receipt = staging / receipt.name
            temporary_receipt.write_text(json.dumps({"key": key, "artifacts": artifacts}) + "\n", encoding="utf-8")
            temporary_receipt.replace(receipt)
            if checkpoint is not None:
                shutil.rmtree(checkpoint, ignore_errors=True)
        return primary

    def score_labels(self, *, strategy="labelwise", saved_only=False):
        if strategy == "labelwise":
            base = self.output_path("shared", "labelwise")
        else:
            base = self.output_path("baselines", strategy)
        return self.run_stage(
            "scoring.labels", self.label_scoring_args(strategy=strategy), (TEST,),
            {"--output": base / "label_scores.jsonl",
             "--raw_output": base / "raw_responses.jsonl"},
            saved_only=saved_only, scoring_strategy=strategy,
        )

    def calibrate(self, scores, alpha, *, sweep=False):
        filename = f"calibrated_alpha_{alpha.replace('.', '')}.jsonl"
        if sweep:
            output = self.output_path("sweeps", "alpha", filename)
        else:
            output = self.output_path("shared", "calibration", filename)
        files = {"--output_path": output}
        if alpha == "0.8":
            files["--prior_json_path"] = self.output_path("shared", "priors", "aimrad_priors.json")
        if scores.resolve() in {path.resolve() for path in files.values()}:
            raise ValueError("Calibration outputs must be separate from the saved label scores.")
        return self.run_stage(
            "calibration",
            ("--train_gold_path", TRAIN, "--gold_path", TEST, "--pred_path", scores, "--alpha", alpha),
            (TRAIN, TEST, scores), files,
        )

    def judge(self, scores, *relative_parts):
        return self.run_stage(
            "pairwise.judge", self.scoring_args,
            (TEST, scores), {"--output": self.output_path(*relative_parts)}, pairwise_input=scores,
        )

    def refine(self, judgments, *relative_parts):
        return self.run_stage(
            "pairwise.refine", ("--input", judgments),
            (judgments,), {"--output": self.output_path(*relative_parts)},
        )


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("opine", "ablations", "baselines"):
        raise SystemExit("Use ./run_opine.sh, ./run_ablations.sh, or ./run_baselines.sh.")
    mode = sys.argv[1]
    parser = argparse.ArgumentParser(
        prog=f"./run_{mode}.sh",
        description=__doc__,
    )
    parser.add_argument("--config", type=Path, default=Path("config.json"), help="Shared run configuration (default: config.json).")
    parser.add_argument("--no-context", action="store_true", help="Score the target sentence without local context.")
    parser.set_defaults(alpha=None, check_api=False, strategy="labelwise")
    if mode == "opine":
        mode_options = parser.add_mutually_exclusive_group()
        mode_options.add_argument("--check-api", action="store_true", help="Check the configured API on one test example.")
        mode_options.add_argument("--alpha", choices=("all",),
                                  help="Calibrate this run's label scores for alpha=0.1 through 0.9 without model calls.")
    elif mode == "baselines":
        parser.add_argument("--strategy", choices=("labelwise", "listwise", "pathwise"),
                            help="Label scoring strategy (default: %(default)s).")
    args = parser.parse_args(sys.argv[2:])
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        keys = {"model", "base_url", "output_root"}
        if not isinstance(config, dict) or config.keys() - keys or not all(isinstance(v, str) for v in config.values()):
            raise ValueError(f"Supported configuration fields (string values): {', '.join(sorted(keys))}.")
        model = config.get("model")
        if not model or not model.strip():
            raise ValueError(f"Set model to your endpoint's model ID in {args.config}.")
        model_root = Path(config.get("output_root", "outputs")) / model_directory(model)
        context_name = "no_context" if args.no_context else "context"
        directory = model_root / context_name
        state_directory = model_root / ".state" / context_name
        workflow = Workflow(
            directory,
            state_directory,
            model,
            config.get("base_url", ""),
            use_context=not args.no_context,
        )
        if args.check_api:
            workflow.check_api()
            return
        directory.mkdir(parents=True, exist_ok=True)
        state_directory.mkdir(parents=True, exist_ok=True)
        with (state_directory / "workflow.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            label_scores = workflow.score_labels(strategy=args.strategy, saved_only=args.alpha == "all")

            if mode == "baselines":
                subprocess.run(["bash", "evaluate.sh", "predictions", str(label_scores)], check=True,
                               env={**os.environ, "PYTHON": sys.executable})
                return

            if args.alpha == "all":
                for alpha in ALPHA_SWEEP:
                    workflow.calibrate(label_scores, f"{alpha:.1f}", sweep=True)
                return

            calibrated = workflow.calibrate(label_scores, "0.8")
            full_pairs = workflow.judge(calibrated, "opine", "with_prior_pairs.jsonl")
            full = workflow.refine(full_pairs, "opine", "full.jsonl")

            if mode == "opine":
                subprocess.run([sys.executable, "-m", "evaluation.scored_labels",
                                "--gold_file", str(TEST), "--pred_file", str(full),
                                "--percentile", "0.99", "--use_adjusted"], check=True)
                return

            text_only = workflow.calibrate(label_scores, "1.0")
            no_prior_pairs = workflow.judge(
                label_scores, "ablations", "without_prior", "without_prior_raw_pairs.jsonl"
            )
            no_prior = workflow.refine(
                no_prior_pairs, "ablations", "without_prior", "without_prior_raw.jsonl"
            )
            subprocess.run(["bash", "evaluate.sh", "ablations",
                            "--full", str(full), "--without-prior", str(no_prior),
                            "--without-refinement", str(calibrated),
                            "--without-both", str(text_only)], check=True,
                           env={**os.environ, "PYTHON": sys.executable})
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode, "OPINE stage failed. See its output above.\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
