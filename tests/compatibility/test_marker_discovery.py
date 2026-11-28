"""Compatibility selection must not import baseline-only test dependencies."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.compatibility


def run_collection(tmp_path, source):
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / "conftest.py", tmp_path / "conftest.py"
    )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers = compatibility: public API compatibility\n"
    )
    (tmp_path / "test_selected.py").write_text(source)
    (tmp_path / "test_baseline_only.py").write_text(
        "raise ImportError('baseline-only private dependency must not be imported')\n"
    )
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "compatibility",
            "--strict-markers",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    "source",
    [
        "import pytest\npytestmark = pytest.mark.compatibility\ndef test_selected(): pass\n",
        "import pytest\n@pytest.mark.compatibility\ndef test_selected(): pass\n",
        "import pytest\n@pytest.mark.compatibility\nclass TestSelected:\n def test_selected(self): pass\n",
    ],
)
def test_marker_discovery_before_import(tmp_path, source):
    result = run_collection(tmp_path, source)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_marked_import_errors_are_not_suppressed(tmp_path):
    result = run_collection(
        tmp_path,
        "import pytest\npytestmark = pytest.mark.compatibility\nraise ImportError('selected import failure')\n",
    )
    assert result.returncode == 2
    assert "selected import failure" in result.stdout
    assert "baseline-only private dependency" not in result.stdout
