"""Standard-library command line and local-path checks for the TCM experiments.

This module deliberately does not import any training dependency or alter algorithms.
"""

import argparse
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "tcm"
MODEL_ROOT = REPO_ROOT / "models" / "tcm"
RUN_ROOT = REPO_ROOT / "runs" / "tcm"
RESOURCE_ROOT = Path(__file__).resolve().parent / "data_preprocess"


def resolve_path(value, base=REPO_ROOT):
    """Resolve relative project paths independently of the working directory."""
    value = Path(value).expanduser()
    return str((base / value).resolve() if not value.is_absolute() else value.resolve())


def validate_jsonl(path, require_labels=True):
    """Check the modern input contract without loading ML libraries or printing data."""
    count = 0
    try:
        with open(path, encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                # Original loaders do not skip blank records.
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    return f"{path}:{line_no}: invalid JSONL ({exc.msg})"
                if not isinstance(row, dict) or not isinstance(row.get("TEXT"), str):
                    return (f"{path}:{line_no}: expected a TEXT string; raw TCM-SD "
                            "samples require an explicit preprocessing/label mapping step")
                if require_labels:
                    try:
                        int(row["LABEL"])
                    except (KeyError, TypeError, ValueError):
                        return f"{path}:{line_no}: expected an integer-convertible LABEL"
                count += 1
    except (OSError, UnicodeError) as exc:
        return f"{path}: {exc}"
    return None if count else f"{path}: empty dataset"


def parse_experiment(entry, defaults, argv=None, *, require_labels=True):
    """Parse configuration and fail before loading torch when inputs are missing.

    Defaults retain each source experiment's hyperparameters and input basenames.
    --check-paths validates inputs and reports paths; it creates no output directory.
    """
    parser = argparse.ArgumentParser(
        description=f"TCM experiment: {entry}. Original algorithm; configurable local paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Relative paths resolve from the repository root. See tasks/tcm/README.md.")
    parser.add_argument("--data-dir", default=str(DATA_ROOT), help="Directory containing prepared experiment inputs")
    parser.add_argument("--model-dir", "--model-name-or-path", dest="model_dir",
                        default=str(MODEL_ROOT / defaults["model_name"]),
                        help="Local pretrained model/tokenizer directory (not bundled)")
    parser.add_argument("--output-dir", default=str(RUN_ROOT / entry), help="Experiment outputs")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--cuda-visible-devices", default=None,
                        help="Optional CUDA_VISIBLE_DEVICES value, applied before importing torch")
    parser.add_argument("--check-paths", action="store_true",
                        help="Check local model/data paths and JSONL schema without training")
    path_keys = {"train_file", "dev_file", "test_file", "input_path", "output_file"}
    for key, value in defaults.items():
        if key == "model_name":
            continue
        if key in path_keys:
            parser.add_argument("--" + key.replace("_", "-"), default=None,
                                help=f"Override path (default basename: {value})")
        else:
            value_type = float if key == "q" else type(value)
            parser.add_argument("--" + key.replace("_", "-"), type=value_type, default=value,
                                help="Original experiment parameter")
    args = parser.parse_args(argv)
    if args.device not in {"auto", "cpu", "cuda"}:
        prefix, _, index = args.device.partition(":")
        if prefix != "cuda" or not index.isdigit():
            parser.error("--device must be auto, cpu, cuda, or cuda:N")
    args.data_dir = resolve_path(args.data_dir)
    args.model_dir = resolve_path(args.model_dir)
    args.output_dir = resolve_path(args.output_dir)
    for key, value in defaults.items():
        if key in path_keys:
            base = Path(args.output_dir) if key == "output_file" else Path(args.data_dir)
            override = getattr(args, key)
            setattr(args, key, resolve_path(override) if override else resolve_path(value, base))
    problems = []
    for key in ("train_file", "dev_file", "test_file", "input_path"):
        if hasattr(args, key):
            error = validate_jsonl(getattr(args, key), require_labels=require_labels)
            if error:
                problems.append(error)
    if not Path(args.model_dir).is_dir():
        problems.append(f"Missing local pretrained model directory: {args.model_dir}")
    if args.check_paths:
        print(json.dumps({"entry": entry, "configuration": vars(args), "problems": problems},
                         ensure_ascii=False, indent=2))
        raise SystemExit(1 if problems else 0)
    if problems:
        parser.error("\n".join(problems))
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return args


def torch_device(torch, value):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)


def prepare_legacy(args, parser, *, mrc=False):
    """Resolve historical CLI paths without importing legacy transformers APIs."""
    for key in ("data_dir", "model_name_or_path", "output_dir", "label_vocab"):
        value = getattr(args, key, None)
        if value:
            setattr(args, key, resolve_path(value))
    if hasattr(args, "knowledge_file"):
        args.knowledge_file = resolve_path(args.knowledge_file) if args.knowledge_file else str(Path(args.data_dir) / "v2_tcm.xlsx")
    problems = []
    files = [args.label_vocab]
    if mrc:
        if args.do_train:
            files.append(str(Path(args.data_dir) / (args.train_file or "train-v1.1.json")))
        if args.do_eval:
            files.append(str(Path(args.data_dir) / (args.predict_file or "dev-v1.1.json")))
    else:
        if args.do_train:
            files.append(str(Path(args.data_dir) / "train_v3_knowledge_tfidf_3k_shuffle.json"))
        if args.do_eval:
            files.append(str(Path(args.data_dir) / "dev_v3_knowledge_tfidf_3k.json"))
        if args.do_test:
            files.append(str(Path(args.data_dir) / "test_v3_knowledge_tfidf_3k.json"))
        if args.do_train or args.do_eval or args.do_test:
            files.append(args.knowledge_file)
    problems.extend(f"Missing legacy input: {value}" for value in files if not Path(value).is_file())
    if not Path(args.model_name_or_path).is_dir():
        problems.append(f"Missing local pretrained model directory: {args.model_name_or_path}")
    if args.check_paths:
        print(json.dumps({"configuration": vars(args), "problems": problems,
                          "note": "Legacy API compatibility and data schema are not validated by this check."},
                         ensure_ascii=False, indent=2))
        raise SystemExit(1 if problems else 0)
    if problems:
        parser.error("\n".join(problems))
