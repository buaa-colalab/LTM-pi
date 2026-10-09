import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseInstallContractTest(unittest.TestCase):
    def test_direct_reference_and_datasets_compatibility_are_declared(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertTrue(metadata["tool"]["hatch"]["metadata"]["allow-direct-references"])
        self.assertIn("datasets<4", metadata["project"]["dependencies"])

    def test_prepare_writes_norm_stats_to_configured_assets_root(self) -> None:
        prepare = (ROOT / "bin" / "prepare.sh").read_text()
        self.assertIn('--output-dir "${ROBOMME_ASSETS_ROOT}/${ASSET_ID}"', prepare)


if __name__ == "__main__":
    unittest.main()
