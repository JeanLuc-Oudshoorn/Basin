"""Disabled screening bypasses model work, reports and WSL while admitting all sources."""

import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
for stage in ("02_data_validation", "03_transforming_input_data"):
    sys.path.insert(0, str(ROOT / "src" / stage))
from transform import load_config, validated_sources
from validate_generated_data import main, validate


class ValidationSkipTests(unittest.TestCase):
    def test_disabled_gate_admits_every_source_without_reading_reports_or_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config()
            config["validation"].update(enabled=False, auto_run=True)
            config["paths"]["processed"] = directory
            paths = [
                Path(directory) / name / "data.parquet"
                for name in ("previously_rejected", "unscreened")
            ]
            with (
                patch(
                    "transform.validation_report_path",
                    side_effect=AssertionError("must not read old scores"),
                ),
                patch(
                    "transform.file_digest",
                    side_effect=AssertionError("must not hash tables"),
                ),
                patch(
                    "validate_generated_data.validate",
                    side_effect=AssertionError("must not screen"),
                ),
            ):
                admitted, audit = validated_sources(config, paths)
            self.assertEqual(admitted, paths)
            self.assertTrue(audit["screening_skipped"])
            self.assertEqual(audit["admitted_count"], 2)
            self.assertEqual(audit["excluded"], [])
            self.assertEqual(
                json.loads((Path(directory) / "validation_selection.json").read_text()),
                audit,
            )

    def test_disabled_validator_skips_discovery_and_retry_report_lookup(self):
        config = load_config()
        config["validation"]["enabled"] = False
        with (
            patch(
                "transform.discover_source_tables",
                side_effect=AssertionError("must not scan"),
            ),
            patch(
                "transform.validation_report_path",
                side_effect=AssertionError("must not retry"),
            ),
        ):
            result = validate(config, retry_errors=True)
        self.assertTrue(result.empty)
        self.assertTrue(result.attrs["screening_skipped"])

    def test_disabled_cli_does_not_launch_wsl_or_prepare_source_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config()
            config["validation"]["enabled"] = False
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            launcher = Mock()
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "validate_generated_data.py",
                        "--config",
                        str(path),
                        "--retry-errors",
                    ],
                ),
                patch("transform.run_context", return_value=nullcontext()),
                patch(
                    "transform.write_source_manifest",
                    side_effect=AssertionError("must not index"),
                ),
                patch.dict(sys.modules, {"train": Mock(launch_wsl=launcher)}),
            ):
                main()
            launcher.assert_not_called()


if __name__ == "__main__":
    unittest.main()
