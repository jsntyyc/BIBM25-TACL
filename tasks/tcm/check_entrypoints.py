"""Dependency-free interface checks: python -B -m tasks.tcm.check_entrypoints."""

import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ._cli import REPO_ROOT, validate_jsonl


MODERN = (
    "run", "run_cl", "run_cl_v2", "run_tcmbert_cl", "runroberta",
    "runroberta_cl", "runroberta_cl_v2", "model.run_tcmbert_cl",
    "curriculumlearning", "curriculum_learning.apw_kmeans",
    "curriculum_learning.kmeans_distance_apw",
)
LEGACY = ("model.run_tcm", "model.run_tcm_mrc", "model.traiditonal_classifer")


class EntryPointChecks(unittest.TestCase):
    def run_python(self, *args):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, "-B", "-S", *args], cwd=REPO_ROOT,
                              env=env, capture_output=True, text=True, encoding="utf-8", timeout=15)

    def test_help_without_site_packages(self):
        for entry in MODERN + LEGACY:
            with self.subTest(entry=entry):
                result = self.run_python("-m", "tasks.tcm." + entry, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_import_does_not_load_training_libraries(self):
        entries = ["tasks.tcm." + entry for entry in MODERN + LEGACY]
        code = """
import importlib, importlib.abc, os, sys
class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'pandas', 'transformers', 'tqdm', 'sklearn', 'matplotlib', 'bert_serving'}:
            raise AssertionError('Unexpected dependency import: ' + fullname)
before = dict(os.environ)
sys.meta_path.insert(0, BlockTrainingImports())
for entry in ENTRIES:
    importlib.import_module(entry)
assert dict(os.environ) == before
""".replace("ENTRIES", repr(entries))
        result = self.run_python("-c", code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_jsonl_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.jsonl"
            path.write_text('{"TEXT":"example","LABEL":"1"}\n', encoding="utf-8")
            self.assertIsNone(validate_jsonl(path))
            path.write_text('{"chief_complaint":"raw sample"}\n', encoding="utf-8")
            self.assertIn("preprocessing", validate_jsonl(path))
            path.write_text('{"TEXT":"example"}\n', encoding="utf-8")
            self.assertIn("LABEL", validate_jsonl(path))
            self.assertIsNone(validate_jsonl(path, require_labels=False))
            path.write_text('{broken}\n', encoding="utf-8")
            self.assertIn(":1:", validate_jsonl(path))
            path.write_text("", encoding="utf-8")
            self.assertIn("empty", validate_jsonl(path))

    def test_preflight_success_and_missing_input_create_no_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            source = data / "input.jsonl"
            source.write_text('{"TEXT":"example","LABEL":0}\n', encoding="utf-8")
            model = data / "local-model"
            model.mkdir()
            output = data / "must-not-be-created"
            common = ["-m", "tasks.tcm.run", "--check-paths", "--model-dir", str(model),
                      "--output-dir", str(output), "--train-file", str(source),
                      "--dev-file", str(source), "--test-file", str(source)]
            result = self.run_python(*common)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["problems"], [])
            self.assertFalse(output.exists())
            source.unlink()
            result = self.run_python(*common)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue(json.loads(result.stdout)["problems"])
            self.assertFalse(output.exists())

    def test_original_experiment_defaults_and_distinct_outputs(self):
        expected = {"run_cl_v2": 0.8, "runroberta_cl_v2": 0.9, "model.run_tcmbert_cl": 0.5}
        for name, beta in expected.items():
            self.assertEqual(importlib.import_module("tasks.tcm." + name).DEFAULTS["beta"], beta)
        self.assertEqual(importlib.import_module("tasks.tcm.run").DEFAULTS["max_seq_length"], 256)
        self.assertEqual(importlib.import_module("tasks.tcm.curriculum_learning.apw_kmeans").DEFAULTS["q"], 1.0)
        self.assertEqual(importlib.import_module("tasks.tcm.curriculum_learning.kmeans_distance_apw").DEFAULTS["q"], 10)

    def test_all_python_sources_parse(self):
        for path in Path(__file__).resolve().parent.rglob("*.py"):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
