"""Regression tests for the full evaluation pipeline.

These tests run ``alignet/test.py`` as a subprocess and verify that the metric
values printed to stdout match the known-good results produced by the reference
checkpoint (``checkpoints/affine_imagenet_sat_sensoraug.ckpt``).

Requirements
------------
- GPU with CUDA available.
- Checkpoint at ``checkpoints/affine_imagenet_sat_sensoraug.ckpt``.
- MDAS data accessible at the path set in ``test_alignet_config_affine_mdas.yml``.
- DREAM data accessible at the path set in ``test_alignet_config_affine_dream.yml``.
- Test CSV files present under ``files/test/``.

Run only these tests
--------------------
    pytest tests/test_evaluation.py -v -m evaluation

Skip them in fast CI
--------------------
    pytest tests/ -m "not evaluation"

Approximate runtimes: MDAS pair ~3 min, DREAM pair ~13 min.
"""

import re
import sys
import subprocess

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_eval(config_path: str, override: str) -> dict:
    """Invoke ``alignet/test.py`` and return the parsed metric dict.

    Raises ``RuntimeError`` if the subprocess exits with a non-zero code.
    """
    cmd = [
        sys.executable, "alignet/test.py",
        "--config", config_path,
        "--override", override,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"test.py exited with code {result.returncode}.\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}"
        )
    return _parse_metrics(result.stdout)


def _parse_metrics(output: str) -> dict:
    """Extract metric values from test.py stdout.

    Matches lines of the form printed by ``on_test_epoch_end``:
        *  rmse:    11.06071
    """
    metrics = {}
    for line in output.splitlines():
        m = re.match(r"\s+\*\s+(\w+):\s+([\d.]+)", line)
        if m:
            metrics[m.group(1)] = float(m.group(2))
    return metrics


def _assert_metrics(actual: dict, expected: dict, atol: float = 0.01) -> None:
    """Assert each expected metric is within ``atol`` of the actual value.

    Using atol=0.01 (one unit in the last reported decimal) accounts for
    floating-point differences across GPU drivers or CUDA versions while still
    catching real regressions.
    """
    assert actual, (
        "No metrics were parsed from the output. "
        "Check that test.py completed successfully and printed results."
    )
    for name, exp_val in expected.items():
        assert name in actual, (
            f"Metric '{name}' not found in output. Available: {list(actual.keys())}"
        )
        diff = abs(actual[name] - exp_val)
        assert diff <= atol, (
            f"Metric '{name}' out of tolerance.\n"
            f"  Expected : {exp_val:.5f}\n"
            f"  Got      : {actual[name]:.5f}\n"
            f"  Diff     : {diff:.5f}  (atol={atol})"
        )


# ---------------------------------------------------------------------------
# Reference values
# Checkpoint : checkpoints/affine_imagenet_sat_sensoraug.ckpt
# Validated  : 2026-03-10
# ---------------------------------------------------------------------------

_MDAS_ENMAP10M_S2_10M = {
    "rmse":  11.06071,
    "mpd":   10.42518,
    "iou":    0.91636,
    "mi":     0.78130,
    "re":     0.06272,
    "h_err":  0.10102,
}

_DREAM_OPTIQUE_RADAR = {
    "rmse":  31.83196,
    "mpd":   30.36468,
    "iou":    0.78757,
    "mi":     0.48252,
    "re":     0.20723,
    "h_err":  0.29469,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.evaluation
def test_mdas_enmap10m_sentinel2_10m():
    """MDAS affine: EeteS_EnMAP_10m → EeteS_Sentinel_2_10m  (~3 min).

    Validates the full pipeline on a cross-modal pair that involves a
    hyperspectral sensor and a multispectral sensor at the same spatial
    resolution.
    """
    actual = _run_eval(
        config_path="./configs/test_alignet_config_affine_mdas.yml",
        override=(
            "datasets.test_1.path="
            "./files/test/mdas_test_affine_pairs/"
            "mdas_test_affine_EeteS_EnMAP_10m_EeteS_Sentinel_2_10m.csv"
        ),
    )
    _assert_metrics(actual, _MDAS_ENMAP10M_S2_10M)


@pytest.mark.evaluation
def test_dream_optique_radar():
    """DREAM affine: optique → radar  (~13 min).

    Validates the full pipeline on a cross-modal pair spanning an optical and
    a radar sensor — the largest modality gap in the DREAM benchmark.
    """
    actual = _run_eval(
        config_path="./configs/test_alignet_config_affine_dream.yml",
        override=(
            "datasets.test_1.path="
            "./files/test/dream_test_affine_pairs/"
            "dream_test_affine_optique_radar.csv"
        ),
    )
    _assert_metrics(actual, _DREAM_OPTIQUE_RADAR)
