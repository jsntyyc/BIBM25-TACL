"""Repository-relative ICD paths. Relative overrides are anchored at PROJECT_ROOT.

No YAML, data, model, directory creation, or optional dependency is loaded here.
"""
import os
from pathlib import Path
from tempfile import mkdtemp

TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK_ROOT.parents[1]
CONFIG_ROOT = TASK_ROOT / "configs"
DATASETS = ("mimiciii_full", "mimiciii_50", "mimiciii_clean", "mimiciv_icd9", "mimiciv_icd10")


def repository_path(value):
    """Resolve an absolute or repository-relative local path independently of cwd."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def environment_path(name, default):
    return repository_path(os.environ.get(name, str(default)))


DATA_ROOT = environment_path("BIBM25_ICD_DATA_ROOT", "data/icd/datasets")
SPLIT_ROOT = environment_path("BIBM25_ICD_SPLIT_ROOT", "data/icd/splits")
MODEL_ROOT = environment_path("BIBM25_ICD_MODEL_ROOT", "models/icd")
EXPERIMENT_DIR = environment_path("BIBM25_ICD_RUNS_DIR", "runs/icd")


def dataset_directory(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown ICD dataset: {dataset}")
    return DATA_ROOT / dataset


def split_directory(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown ICD dataset: {dataset}")
    return SPLIT_ROOT / dataset


def data_paths(config):
    """Resolve training data/split paths; old configs without split_dir remain usable."""
    data_dir = repository_path(config.dir)
    split_dir = repository_path(getattr(config, "split_dir", None) or data_dir)
    return data_dir / config.data_filename, split_dir / config.split_filename


def create_run_directory(directory=None):
    directory = repository_path(directory or EXPERIMENT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return Path(mkdtemp(prefix="icd-", dir=directory))


def report_output_path(relative):
    path = EXPERIMENT_DIR / "images" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
ID_COLUMN = "_id"
TEXT_COLUMN = "text"
TARGET_COLUMN = "target"
SUBJECT_ID_COLUMN = "subject_id"
DOWNLOAD_DIRECTORY_MIMICIII = environment_path("BIBM25_MIMICIII_RAW", "data/icd/raw/mimiciii")
DOWNLOAD_DIRECTORY_MIMICIV = environment_path("BIBM25_MIMICIV_RAW", "data/icd/raw/mimiciv")
DOWNLOAD_DIRECTORY_MIMICIV_NOTE = environment_path("BIBM25_MIMICIV_NOTE_RAW", "data/icd/raw/mimiciv-note")
DATA_DIRECTORY_MIMICIII_FULL = dataset_directory("mimiciii_full")
DATA_DIRECTORY_MIMICIII_50 = dataset_directory("mimiciii_50")
DATA_DIRECTORY_MIMICIII_CLEAN = dataset_directory("mimiciii_clean")
DATA_DIRECTORY_MIMICIV_ICD9 = dataset_directory("mimiciv_icd9")
DATA_DIRECTORY_MIMICIV_ICD10 = dataset_directory("mimiciv_icd10")
PROJECT = os.environ.get("BIBM25_WANDB_PROJECT", "<your project name>")
PALETTE = {
    "PLM-ICD": "#E69F00",
    "LAAT": "#009E73",
    "MultiResCNN": "#D55E00",
    "CAML": "#56B4E9",
    "CNN": "#CC79A7",
    "Bi-GRU": "#F5C710",
}
HUE_ORDER = ["PLM-ICD", "LAAT", "MultiResCNN", "CAML", "Bi-GRU", "CNN"]
MODEL_NAMES = {"PLMICD": "PLM-ICD", "VanillaConv": "CNN", "VanillaRNN": "Bi-GRU"}
