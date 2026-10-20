"""Dependency-free checks for the namespace and path migration."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tasks.icd.src import settings


class PathMigrationTests(unittest.TestCase):
    def test_paths_are_independent_of_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(settings.repository_path("data/icd/splits"), settings.PROJECT_ROOT / "data/icd/splits")
            finally:
                os.chdir(previous)

    def test_separate_split_directory(self):
        config = SimpleNamespace(dir="data/icd/datasets/sample", split_dir="data/icd/splits/sample", data_filename="corpus.feather", split_filename="split.feather")
        data, split = settings.data_paths(config)
        self.assertEqual(data, settings.PROJECT_ROOT / "data/icd/datasets/sample/corpus.feather")
        self.assertEqual(split, settings.PROJECT_ROOT / "data/icd/splits/sample/split.feather")

    def test_legacy_configuration_falls_back_to_data_directory(self):
        config = SimpleNamespace(dir="legacy", data_filename="corpus.feather", split_filename="split.feather")
        data, split = settings.data_paths(config)
        self.assertEqual(data.parent, split.parent)

    def test_absolute_override_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(settings.repository_path(directory), Path(directory).resolve())

    def test_relative_environment_override_is_repository_relative(self):
        with patch.dict(os.environ, {"BIBM25_TEST_ICD_PATH": "custom/icd"}):
            self.assertEqual(settings.environment_path("BIBM25_TEST_ICD_PATH", "unused"), settings.PROJECT_ROOT / "custom/icd")

    def test_help_works_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", "-m", "tasks.icd.main", "--help"], cwd=settings.PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("data/icd/splits", result.stdout)

    def test_import_has_no_optional_dependency_side_effects(self):
        code = "import sys; import tasks.icd.main; import tasks.icd.src.settings; assert not any(n in sys.modules for n in ('torch','hydra','omegaconf','pandas'))"
        result = subprocess.run([sys.executable, "-S", "-c", code], cwd=settings.PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
