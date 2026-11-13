"""Training storage stays on its configured project drive before torch imports."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/05_model_training"))
from runtime_storage import configure_runtime_storage


class RuntimeStorageTests(unittest.TestCase):
    def test_temp_and_compiler_paths_override_inherited_locations(self):
        previous_temp = tempfile.tempdir
        try:
            with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as temporary:
                with patch.dict(
                    os.environ,
                    {
                        "TMPDIR": "C:/old-temp",
                        "TORCHINDUCTOR_CACHE_DIR": "C:/old-cache",
                    },
                ):
                    paths = configure_runtime_storage(temporary)
                    for name, value in paths.items():
                        self.assertTrue(
                            Path(value).is_relative_to(Path(temporary).resolve())
                        )
                        self.assertEqual(os.environ[name], value)
                        self.assertTrue(Path(value).is_dir())
                    with tempfile.NamedTemporaryFile() as file:
                        self.assertTrue(
                            Path(file.name).is_relative_to(Path(temporary).resolve())
                        )
                    self.assertTrue(
                        Path(os.environ["CUDA_COREDUMP_FILE"]).is_relative_to(
                            Path(temporary).resolve()
                        )
                    )
        finally:
            tempfile.tempdir = previous_temp


if __name__ == "__main__":
    unittest.main()
