"""Explicit task paths. Importing this module never reads patient data."""
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]

def _root(variable, default):
    return Path(os.environ.get(variable, str(default))).expanduser().resolve()

DATA_DIR = _root('TACL_READMISSION_DATA_DIR', REPOSITORY / 'data/readmission')
MODEL_DIR = _root('TACL_READMISSION_MODEL_DIR', REPOSITORY / 'models/readmission')
RAW_DIR = _root('TACL_MIMICIII_RAW_DIR', DATA_DIR / 'raw/mimiciii/1.4')

def output_file(name):
    return str(Path(os.environ['TACL_READMISSION_OUTPUT_DIR']) / name)

def data_file(split, default):
    return os.environ.get('TACL_READMISSION_' + split.upper() + '_FILE', default)

def difficulty_file(default):
    return os.environ.get('TACL_READMISSION_DIFFICULTY_FILE', default)

def device_name(default):
    return os.environ.get('TACL_READMISSION_DEVICE', default)

