"""
(AI Generated)
Regression tests for fetching confirmed-planet literature data from the KOI dataset.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from get_literature_data import (  # noqa: E402
    CONFIRMED_COLUMNS,
    LiteratureDataError,
    _build_koi_query,
    fetch_koi_table,
    get_literature_data,
    koi_rows_to_confirmed,
    parse_args,
)

# A trimmed stand-in for what the NASA Exoplanet Archive TAP service returns
# for `SELECT ... FROM cumulative WHERE lower(kepler_name) like 'kepler-5 %'`.
KEPLER_5_RAW = pd.DataFrame(
    [
        {
            "kepid": 8191672,
            "kepoi_name": "K00018.01",
            "kepler_name": "Kepler-5 b",
            "koi_disposition": "CONFIRMED",
            "koi_period": 3.548465446,
            "koi_time0bk": 122.9014315,
            "koi_duration": 4.56904,
            "koi_depth": 7251.5,
            "koi_ror": 0.078272,
            "koi_impact": 0.075,
            "koi_incl": 89.33,
            "koi_dor": 6.4128,
            "koi_eccen": 0,
            "koi_prad": 14.92,
            "koi_srad": 1.746,
            "koi_max_sngle_ev": 165.75914,
            "koi_max_mult_ev": 2268.56,
        }
    ]
)

# A multi-planet system, with a non-confirmed KOI mixed in to exercise the
# disposition filter (mirrors how Kepler-90 looks in the real cumulative table).
KEPLER_90_RAW = pd.DataFrame(
    [
        {
            "kepid": 11442793,
            "kepoi_name": "K00351.06",
            "kepler_name": "Kepler-90 b",
            "koi_disposition": "CONFIRMED",
            "koi_period": 7.00821787,
            "koi_time0bk": 137.67164,
            "koi_duration": 3.812,
            "koi_depth": 116.4,
            "koi_ror": 0.009852,
            "koi_impact": 0.156,
            "koi_incl": 89.36,
            "koi_dor": 14.03,
            "koi_eccen": 0,
            "koi_prad": 1.29,
            "koi_srad": 1.2,
            "koi_max_sngle_ev": 4.745033,
            "koi_max_mult_ev": 14.816139,
        },
        {
            "kepid": 11442793,
            "kepoi_name": "K00351.01",
            "kepler_name": "Kepler-90 h",
            "koi_disposition": "CONFIRMED",
            "koi_period": 331.5972732,
            "koi_time0bk": 140.498929,
            "koi_duration": 14.404,
            "koi_depth": 8322.1,
            "koi_ror": 0.083115,
            "koi_impact": 0.02,
            "koi_incl": 89.99,
            "koi_dor": 190.456,
            "koi_eccen": 0,
            "koi_prad": 10.89,
            "koi_srad": 1.2,
            "koi_max_sngle_ev": 183.05147,
            "koi_max_mult_ev": 249.14,
        },
        {
            "kepid": 11442793,
            "kepoi_name": "K00351.08",
            "kepler_name": None,
            "koi_disposition": "FALSE POSITIVE",
            "koi_period": 1.2,
            "koi_time0bk": 100.0,
            "koi_duration": 1.0,
            "koi_depth": 10.0,
            "koi_ror": 0.001,
            "koi_impact": 0.1,
            "koi_incl": 89.0,
            "koi_dor": 5.0,
            "koi_eccen": 0,
            "koi_prad": 0.5,
            "koi_srad": 1.2,
            "koi_max_sngle_ev": 1.0,
            "koi_max_mult_ev": 1.0,
        },
    ]
)


class BuildKoiQueryTests(unittest.TestCase):
    def test_query_matches_star_case_insensitively_with_trailing_space(self):
        query = _build_koi_query("Kepler-5")
        self.assertIn("lower(kepler_name) like 'kepler-5 %'", query)

    def test_query_does_not_let_kepler_5_match_kepler_50(self):
        query = _build_koi_query("Kepler-5")
        self.assertNotIn("kepler-50", query)

    def test_query_escapes_single_quotes(self):
        query = _build_koi_query("O'Brien-1")
        self.assertIn("o''brien-1", query)


class KoiRowsToConfirmedTests(unittest.TestCase):
    def test_single_planet_row_matches_expected_columns_and_units(self):
        confirmed = koi_rows_to_confirmed(KEPLER_5_RAW)

        self.assertEqual(list(confirmed.columns), CONFIRMED_COLUMNS)
        self.assertEqual(len(confirmed), 1)

        row = confirmed.iloc[0]
        self.assertEqual(row["target"], "Kepler-5b")
        self.assertAlmostEqual(row["period_days"], 3.548465446)
        self.assertAlmostEqual(row["t0"], 122.9014315 + 2_454_833.0)
        self.assertAlmostEqual(row["duration_hours"], 4.56904)
        self.assertAlmostEqual(row["duration_days"], 4.56904 / 24.0)
        self.assertAlmostEqual(row["depth_mean_per_transit"], 7251.5 / 1_000_000.0)
        self.assertAlmostEqual(row["planet_radius_rjup"], 14.92 / 11.209, places=6)
        self.assertEqual(row["eccentricity"], 0.0)
        self.assertIsInstance(row["eccentricity"], float)

    def test_multi_planet_system_is_sorted_by_period_and_drops_non_confirmed(self):
        confirmed = koi_rows_to_confirmed(KEPLER_90_RAW)

        # The FALSE POSITIVE / unnamed row must be excluded.
        self.assertEqual(confirmed["target"].tolist(), ["Kepler-90b", "Kepler-90h"])
        self.assertTrue(confirmed["period_days"].is_monotonic_increasing)

    def test_empty_input_returns_empty_confirmed_frame(self):
        confirmed = koi_rows_to_confirmed(pd.DataFrame())
        self.assertTrue(confirmed.empty)
        self.assertEqual(list(confirmed.columns), CONFIRMED_COLUMNS)

    def test_no_confirmed_rows_returns_empty_confirmed_frame(self):
        raw = pd.DataFrame(
            [{**KEPLER_5_RAW.iloc[0].to_dict(), "koi_disposition": "CANDIDATE"}]
        )
        confirmed = koi_rows_to_confirmed(raw)
        self.assertTrue(confirmed.empty)

    def test_duplicate_kepler_names_are_deduplicated(self):
        duplicated = pd.concat([KEPLER_5_RAW, KEPLER_5_RAW], ignore_index=True)
        confirmed = koi_rows_to_confirmed(duplicated)
        self.assertEqual(len(confirmed), 1)


class GetLiteratureDataTests(unittest.TestCase):
    def test_planet_target_saves_under_star_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path, confirmed = get_literature_data(
                "Kepler-5b",
                out_dir=tmp_dir,
                fetch=lambda star, **_: KEPLER_5_RAW,
            )

            self.assertEqual(out_path, Path(tmp_dir) / "Kepler-5-confirmed.csv")
            self.assertTrue(out_path.is_file())
            saved = pd.read_csv(out_path)
            self.assertEqual(saved["target"].tolist(), ["Kepler-5b"])
            self.assertEqual(list(confirmed.columns), CONFIRMED_COLUMNS)

    def test_star_target_resolves_the_same_as_a_planet_target(self):
        captured_stars: list[str] = []

        def fake_fetch(star: str, **_: object) -> pd.DataFrame:
            captured_stars.append(star)
            return KEPLER_5_RAW

        with tempfile.TemporaryDirectory() as tmp_dir:
            get_literature_data("Kepler-5", out_dir=tmp_dir, fetch=fake_fetch)
            get_literature_data("Kepler-5b", out_dir=tmp_dir, fetch=fake_fetch)

        self.assertEqual(captured_stars, ["Kepler-5", "Kepler-5"])

    def test_no_matching_rows_raises_literature_data_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(LiteratureDataError):
                get_literature_data(
                    "Kepler-99999",
                    out_dir=tmp_dir,
                    fetch=lambda star, **_: pd.DataFrame(),
                )

    def test_non_kepler_mission_is_rejected_before_saving(self):
        def unsupported_fetch(star: str, *, mission: str = "Kepler") -> pd.DataFrame:
            return fetch_koi_table(star, mission=mission)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(NotImplementedError):
                get_literature_data(
                    "Kepler-5",
                    mission="TESS",
                    out_dir=tmp_dir,
                    fetch=unsupported_fetch,
                )


class ParseArgsTests(unittest.TestCase):
    def test_defaults_to_kepler_mission_and_confirmed_dir(self):
        args = parse_args(["Kepler-5b"])
        self.assertEqual(args.target, "Kepler-5b")
        self.assertEqual(args.mission, "Kepler")
        self.assertEqual(args.out_dir.name, "confirmed")

    def test_mission_and_out_dir_are_overridable(self):
        args = parse_args(["Kepler-5b", "--mission", "K2", "--out-dir", "/tmp/somewhere"])
        self.assertEqual(args.mission, "K2")
        self.assertEqual(args.out_dir, Path("/tmp/somewhere"))


if __name__ == "__main__":
    unittest.main()
