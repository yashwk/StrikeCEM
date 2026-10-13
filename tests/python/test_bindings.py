import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from strikecem import (
    Database,
    QueryResult,
    QueryRequest,
    find_library,
    ChannelNotFoundError,
    DesignMismatchError,
    DatabaseCorruptError,
    DatabaseOpenError,
)

class TestStrikeCemPythonSdk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workdir = tempfile.TemporaryDirectory()
        cls.work_path = Path(cls.workdir.name)
        cls.repo_root = Path(__file__).resolve().parent.parent.parent
        cls.strikecem_bin = cls.repo_root / "build" / "release" / "strikecem"
        cls.plate_stl = cls.repo_root / "examples" / "plate.stl"

        # Generate test configuration and run strikecem to create test .scem and .h5 files
        cls.scem_db_path = cls.work_path / "target.scem"
        cls.h5_db_path = cls.work_path / "target.h5"

        cfg_json = f"""{{
  "scem_schema_version": "1.0",
  "model": {{"path": "{cls.plate_stl.as_posix()}", "units": "m"}},
  "frequency": {{"frequency_hz": 10000000000}},
  "angles": {{
    "azimuth": {{"start": 0, "stop": 90, "step": 30}},
    "elevation": {{"start": -90, "stop": -90, "step": 1}}
  }},
  "polarization": ["HH", "VV"],
  "solver": {{"type": "PO"}},
  "output": {{"path": "{cls.scem_db_path.as_posix()}", "format": "scem"}},
  "integration": {{
    "designer": {{
      "design_id": "test-aerospace-target",
      "revision": "v1.2"
    }}
  }}
}}"""
        cfg_scem_file = cls.work_path / "cfg_scem.json"
        cfg_scem_file.write_text(cfg_json)

        # Run strikecem to generate .scem
        res = subprocess.run(
            [str(cls.strikecem_bin), "run", str(cfg_scem_file)],
            capture_output=True,
            text=True,
            check=True
        )

        # Generate .h5
        cfg_h5_file = cls.work_path / "cfg_h5.json"
        cfg_h5_file.write_text(cfg_json.replace(cls.scem_db_path.as_posix(), cls.h5_db_path.as_posix()).replace('"scem"', '"hdf5"'))
        subprocess.run(
            [str(cls.strikecem_bin), "run", str(cfg_h5_file)],
            capture_output=True,
            text=True,
            check=True
        )

    @classmethod
    def tearDownClass(cls):
        cls.workdir.cleanup()

    def test_01_find_library(self):
        lib_path = find_library()
        self.assertTrue(os.path.isfile(lib_path))
        self.assertIn("libStrikeCEM", lib_path)

    def test_02_database_open_close(self):
        db = Database.open(str(self.scem_db_path))
        self.assertFalse(db._closed)
        db.close()
        self.assertTrue(db._closed)
        with self.assertRaises(RuntimeError):
            _ = db.frequencies

    def test_03_context_manager(self):
        with Database.open(str(self.scem_db_path)) as db:
            self.assertEqual(len(db.frequencies), 1)
        self.assertTrue(db._closed)

    def test_04_metadata_scem(self):
        with Database.open(str(self.scem_db_path)) as db:
            self.assertEqual(db.design_id, "test-aerospace-target")
            self.assertEqual(db.design_revision, "v1.2")
            self.assertEqual(db.frequencies, [10e9])
            self.assertEqual(db.polarizations, ["HH", "VV"])

    def test_05_metadata_hdf5(self):
        with Database.open(str(self.h5_db_path)) as db:
            self.assertEqual(db.design_id, "test-aerospace-target")
            self.assertEqual(db.design_revision, "v1.2")
            self.assertEqual(db.frequencies, [10e9])
            self.assertEqual(db.polarizations, ["HH", "VV"])

    def test_06_verify_design(self):
        with Database.open(str(self.scem_db_path)) as db:
            self.assertTrue(db.verify_design("test-aerospace-target", "v1.2"))
            with self.assertRaises(DesignMismatchError):
                db.verify_design("wrong-target", "v1.2")
            with self.assertRaises(DesignMismatchError):
                db.verify_design("test-aerospace-target", "v9.9")

    def test_07_query_exact_and_interpolated(self):
        with Database.open(str(self.scem_db_path)) as db:
            # Exact grid point
            res_0 = db.query(az_deg=0.0, el_deg=-90.0, freq_hz=10e9, pol="HH")
            self.assertIsInstance(res_0, QueryResult)
            self.assertIsInstance(res_0.scattering, complex)
            self.assertGreaterEqual(res_0.rcs_sqm, 0.0)

            # Interpolated query (midway between 0 and 30 deg)
            res_mid = db.query(az_deg=15.0, el_deg=-90.0, freq_hz=10e9, pol="HH")
            self.assertIsInstance(res_mid, QueryResult)

    def test_08_channel_not_found(self):
        with Database.open(str(self.scem_db_path)) as db:
            with self.assertRaises(ChannelNotFoundError):
                db.query(az_deg=0.0, el_deg=-90.0, freq_hz=10e9, pol="RHCP")

    def test_09_query_batch(self):
        with Database.open(str(self.scem_db_path)) as db:
            reqs = [
                QueryRequest(azimuth_deg=0.0, elevation_deg=-90.0, frequency_hz=10e9, polarization="HH"),
                QueryRequest(azimuth_deg=30.0, elevation_deg=-90.0, frequency_hz=10e9, polarization="HH"),
                QueryRequest(azimuth_deg=60.0, elevation_deg=-90.0, frequency_hz=10e9, polarization="VV"),
            ]
            results = db.query_batch(reqs)
            self.assertEqual(len(results), 3)
            for r in results:
                self.assertIsInstance(r, QueryResult)

            # Tuple format
            tuples = [
                (0.0, -90.0, 10e9, "HH"),
                (30.0, -90.0, 10e9, "VV")
            ]
            t_res = db.query_batch(tuples)
            self.assertEqual(len(t_res), 2)

    def test_10_query_batch_numpy(self):
        with Database.open(str(self.scem_db_path)) as db:
            az = [0.0, 30.0, 60.0]
            el = [-90.0, -90.0, -90.0]
            freq = [10e9, 10e9, 10e9]
            out = db.query_batch_numpy(az, el, freq, "HH")
            self.assertIn("rcs_sqm", out)
            self.assertIn("rcs_dbsm", out)
            self.assertIn("scattering_real", out)
            self.assertIn("scattering_imag", out)
            self.assertEqual(len(out["rcs_sqm"]), 3)

    def test_11_nonexistent_file(self):
        with self.assertRaises((DatabaseOpenError, DatabaseCorruptError)):
            Database.open("/nonexistent/path/db.scem")

if __name__ == "__main__":
    unittest.main()
