"""Lazy CLI for the preserved 0428 experiment. See README for prerequisites."""
import argparse
from tasks.readmission._paths import DATA_DIR, MODEL_DIR, RAW_DIR, REPOSITORY, data_file, output_file, difficulty_file
from tasks.readmission._entry import launch

def build_parser():
    parser = argparse.ArgumentParser(description="Preserved ClinicalBERT experiment; original training settings are unchanged.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, help="PyTorch device override, e.g. cpu or cuda:0; original default retained if omitted")
    parser.add_argument("--check-paths", action="store_true", help="Validate external paths without importing ML libraries")
    return parser

def main(argv=None):
    return launch(__file__, build_parser(), argv)

if __name__ == "__main__":
    raise SystemExit(main())
