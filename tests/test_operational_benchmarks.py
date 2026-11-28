import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from loguru import logger
from scipy.special import ndtri

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "11_operational_benchmarks"))
import operational_datasets as od  # noqa: E402
import operational_scoring as scoring  # noqa: E402
import prepare_cases  # noqa: E402

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
OPTIONS = prepare_cases.settings(
    {
        "operational_benchmarks": {
            "horizon": 8,
            "context_lengths": [16, 32],
            "forecast": {"origins": {"cats": 3, "ucr": 3}, "anomaly_margin": 5},
            "anomaly": {"lead_points": [2, 5]},
        }
    }
)


def setUpModule():
    for name in ("operational_benchmarks", "transform"):
        logger.disable(name)


def tearDownModule():
    for name in ("operational_benchmarks", "transform"):
        logger.enable(name)


def series(length=400, anomalies=((200, 230),), dataset="ucr", companions=0):
    values = np.sin(np.arange(length) / 5.0)
    others = {
        f"other{n}": values * 0 + n + np.arange(length) for n in range(companions)
    }
    return od.Series(
        dataset=dataset,
        name="example",
        category="electricity",
        description="",
        values=values,
        anomalies=[od.Anomaly(s, e, f"a{n}") for n, (s, e) in enumerate(anomalies)],
        abnormal=list(anomalies),
        companions=others,
    )


def normal_quantiles(median, sigma):
    """Quantiles [cases, levels, steps] of normal forecasts."""
    return median[:, None, :] + ndtri(np.array(LEVELS))[None, :, None] * sigma


class DatasetTests(unittest.TestCase):
    def test_only_machine_grid_and_weather_families_are_operational(self):
        expected = {
            "PowerDemand3": "electricity",
            "DISTORTEDPowerDemand1": "electricity",
            "Italianpowerdemand": "electricity",
            "NOISECIMIS44AirTemperature4": "weather",
            "TkeepForthMARS": "spacecraft_telemetry",
            "DISTORTEDsel840mECG1": "physiology",
            "mit14046longtermecg": "physiology",
            "1sddb40": "physiology",
            "qtdbSel1005V": "physiology",
            "s20101mML2": "physiology",
            "ltstdbs30791AI": "physiology",
            "resperation11": "physiology",
            "tilt12744mtable": "physiology",
            "InternalBleeding16": "physiology",
            "CHARISfive": "physiology",
            "BIDMC1": "physiology",
            "taichidbS0715Master": "physiology",
            "NOISEgaitHunt2": "human_motion",
            "park3m": "human_motion",
            "WalkingAceleration5": "human_motion",
            "GP711MarkerLFM5z3": "human_motion",
            "weallwalk": "human_motion",
            "Lab2Cmac011215EPG4": "biology",
            "insectEPG2": "biology",
            "MesoplodonDensirostris": "biology",
            "SomethingNew": "unknown",
        }
        for name, category in expected.items():
            self.assertEqual(od.ucr_family(name)[2], category, name)
        self.assertEqual(od.ucr_family("DISTORTEDECG4")[1], "distorted")
        self.assertNotIn("physiology", od.OPERATIONAL)
        self.assertNotIn("unknown", od.OPERATIONAL)

    def test_ucr_files_give_zero_based_anomaly_positions_in_both_layouts(self):
        with tempfile.TemporaryDirectory() as folder:
            nested = (
                Path(folder) / "archive" / "FilesAreInHere" / "UCR_Anomaly_FullData"
            )
            nested.mkdir(parents=True)
            values = np.arange(50, dtype=float)
            (nested / "001_UCR_Anomaly_PowerDemand1_10_21_25.txt").write_text(
                "\n".join(f"   {v:.7e}" for v in values)
            )
            (nested / "002_UCR_Anomaly_ECG1_10_30_30.txt").write_text(
                "  ".join(f"{v:.7e}" for v in values)
            )
            listing = od.ucr_listing(Path(folder) / "archive")
            self.assertEqual([r["operational"] for r in listing], [True, False])
            first, second = (od.load_ucr(r) for r in listing)
        # The name says points 21 to 25, counted from 1, both included.
        self.assertEqual((first.anomalies[0].start, first.anomalies[0].end), (20, 25))
        np.testing.assert_array_equal(first.values, values)
        np.testing.assert_array_equal(second.values, values)
        self.assertEqual(second.abnormal, [(29, 30)])

    def test_cloud_series_are_put_on_an_even_grid_with_their_anomalies(self):
        with tempfile.TemporaryDirectory() as folder:
            for domain in ("data-ingress-rate", "consumer-purchase-rate"):
                (Path(folder) / "data" / domain).mkdir(parents=True)
            rows = ["TimeStamp,Value,Label"]
            for hour in range(12):
                if hour == 7:
                    continue  # a moment without a row
                label = 1 if hour in (3, 4, 10) else 0
                rows.append(f'"2018-04-25T{hour:02d}:00:00Z",{hour * 1.5},{label}')
                if hour == 2:
                    rows.append('"2018-04-25T02:00:00Z",999,1')  # a repeated moment
            rows[7] = '"2018-04-25T05:00:00Z",,0'  # a row without a value
            (Path(folder) / "data/data-ingress-rate/ingress-01.csv").write_text(
                "\n".join(rows)
            )
            (Path(folder) / "data/consumer-purchase-rate/purchase-01.csv").write_text(
                "\n".join(rows)
            )
            listing = od.cloud_listing(folder)
            self.assertEqual(
                [(r["category"], r["operational"]) for r in listing],
                [("commercial", False), ("cloud_service_telemetry", True)],
            )
            series_ = od.load_cloud(listing[1])
            with self.assertRaisesRegex(FileNotFoundError, "git clone"):
                od.cloud_listing(Path(folder) / "missing")
        expected = np.arange(12) * 1.5
        expected[[5, 7]] = np.nan
        np.testing.assert_array_equal(series_.values, expected)
        self.assertEqual(series_.name, "data-ingress-rate/ingress-01")
        self.assertEqual(
            [(a.start, a.end) for a in series_.anomalies], [(3, 5), (10, 11)]
        )
        self.assertEqual(len({a.label for a in series_.anomalies}), 2)

    def test_caster_gives_every_sensor_and_marks_where_it_is_not_casting(self):
        rows = 40
        frame = pd.DataFrame(
            {
                "Time (s)": np.arange(rows) * 0.5,
                "Casting speed (m/min)": 7.0,
                "Pyrometer reading (K)": 1350.0,
                "Tundish weight (ton)": 12.0,
                "Superheat (K)": 25.0,
                "Mold level (mm)": 140.0 + np.arange(rows),
                "Zone 1 water flow (L/s)": np.arange(rows, dtype=float),
            }
        )
        frame.loc[:5, "Pyrometer reading (K)"] = 255.0  # the strand has not arrived
        frame.loc[30:, "Mold level (mm)"] = 0.5  # a breakout empties the mold
        frame.iloc[20, 1:] = np.nan  # one row was not recorded
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "Continuous Caster Operating Profile 02 (CCOP-02).csv"
            frame.to_csv(path, index=False)
            (record,) = od.caster_listing(folder)
            sensors = od.load_caster(record)
            coarse = od.load_caster(record, samples_per_point=4)
            with self.assertRaisesRegex(FileNotFoundError, "10.1184"):
                od.caster_listing(Path(folder) / "missing")
        self.assertEqual(len(sensors), 6)
        flow = sensors[-1]
        self.assertEqual(flow.name, "CCOP-02/Zone 1 water flow")
        self.assertTrue(flow.operational)
        self.assertEqual(flow.anomalies, [])  # no labels: forecast cases only
        # Not casting before the strand arrives and from the breakout on; the
        # row that was not recorded is missing, not a stop.
        self.assertEqual(flow.abnormal, [(0, 6), (30, 40)])
        self.assertTrue(np.isnan(flow.values[20]))
        self.assertEqual(len(flow.companions), 5)
        self.assertNotIn("Zone 1 water flow", flow.companions)
        self.assertEqual(coarse[-1].abnormal, [(0, 2), (7, 10)])
        np.testing.assert_allclose(coarse[-1].values[:2], [1.5, 5.5])
        cases, _ = prepare_cases.build_cases(
            sensors,
            prepare_cases.settings(
                {
                    "operational_benchmarks": {
                        "horizon": 4,
                        "context_lengths": [4],
                        "forecast": {"origins": 20, "anomaly_margin": 1},
                        "anomaly": {"lead_points": [1]},
                    }
                }
            ),
        )
        self.assertTrue(cases)
        for case in cases:
            self.assertEqual(case["task"], "forecast")
            start, end = case["origin"] - 4, case["origin"] + 4
            self.assertTrue(7 <= start and (end <= 20 or 21 <= start) and end <= 29)
            self.assertEqual(len(case["covariates"]), 5)

    def cats_folder(self, folder):
        rows = 120
        stamps = pd.date_range("2023-01-01", periods=rows, freq="s")
        labelled = np.zeros(rows)
        labelled[40:50] = labelled[90:100] = 1
        frame = pd.DataFrame(
            {
                "timestamp": stamps.strftime("%Y-%m-%d %H:%M:%S"),
                "bed1": np.arange(rows, dtype=float),
                "cfo1": np.arange(rows, dtype=float) * 2,
                "y": labelled,
            }
        )
        frame.to_csv(Path(folder) / "data.csv", index=False)
        (Path(folder) / "metadata.csv").write_text(
            "start_time,end_time,root_cause,affected,category\n"
            "2023-01-01 00:00:40,2023-01-01 00:00:50,bed1,['cfo1'],3\n"
            "2023-01-01 00:01:30,2023-01-01 00:01:40,bso1,['cfo1'],7\n"
        )

    def test_cats_gives_each_channel_its_anomalies_and_every_abnormal_range(self):
        with tempfile.TemporaryDirectory() as folder:
            self.cats_folder(folder)
            bed1, cfo1 = od.load_cats(folder, channels=("bed1", "cfo1"))
            coarse = od.load_cats(folder, channels=("bed1",), seconds_per_point=8)[0]
            (Path(folder) / "metadata.csv").write_text(
                "start_time,end_time,root_cause,affected,category\n"
                "2023-01-01 00:00:41,2023-01-01 00:00:50,bed1,['cfo1'],3\n"
            )
            with self.assertRaisesRegex(ValueError, "disagree"):
                od.load_cats(folder, channels=("bed1",))
        self.assertEqual(
            [(a.start, a.end, a.role) for a in bed1.anomalies], [(40, 50, "root_cause")]
        )
        self.assertEqual([a.role for a in cfo1.anomalies], ["affected", "affected"])
        # Normal data of bed1 must also avoid the anomaly that starts elsewhere.
        self.assertEqual(bed1.abnormal, [(40, 50), (90, 100)])
        # The other channels of the system come along as possible covariates.
        self.assertEqual(list(bed1.companions), ["cfo1"])
        np.testing.assert_array_equal(bed1.companions["cfo1"], cfo1.values)
        self.assertEqual(len(coarse.values), 15)
        self.assertAlmostEqual(coarse.values[0], np.arange(8).mean())
        # Seconds 40 to 49 touch the points 5 and 6 of eight seconds each.
        self.assertEqual((coarse.anomalies[0].start, coarse.anomalies[0].end), (5, 7))


class CaseTests(unittest.TestCase):
    def test_forecast_cases_stay_clear_of_anomalies_and_their_margin(self):
        cases, _ = prepare_cases.build_cases([series()], OPTIONS)
        forecasts = [c for c in cases if c["task"] == "forecast"]
        self.assertEqual(len(forecasts), 6)  # three for each context length
        for case in forecasts:
            start = case["origin"] - case["context_length"]
            end = case["origin"] + OPTIONS["horizon"]
            self.assertTrue(end <= 200 - 5 or start >= 230 + 5, (start, end))
            self.assertEqual(len(case["values"]), case["context_length"] + 8)
        self.assertEqual([c["case"] for c in cases], list(range(len(cases))))

    def test_forecast_cases_are_a_horizon_apart_and_skip_missing_values(self):
        source = series(length=120, anomalies=())
        source.values[60] = np.nan
        cases, _ = prepare_cases.build_cases(
            [source],
            prepare_cases.settings(
                {
                    "operational_benchmarks": {
                        "horizon": 8,
                        "context_lengths": [16],
                        "forecast": {"origins": 50, "anomaly_margin": {"ucr": 5}},
                        "anomaly": {"lead_points": [2]},
                    }
                }
            ),
        )
        origins = [c["origin"] for c in cases]
        self.assertGreater(len(origins), 5)
        self.assertTrue(all(b - a >= 8 for a, b in zip(origins, origins[1:])))
        for origin in origins:
            self.assertFalse(origin - 16 <= 60 < origin + 8)

    def test_anomaly_cases_start_before_the_anomaly_and_judge_only_its_points(self):
        cases, skipped = prepare_cases.build_cases(
            [series(anomalies=((200, 203), (260, 300)))], OPTIONS
        )
        found = {
            (c["anomaly"], c["context_length"], c["lead"]): c
            for c in cases
            if c["task"] == "anomaly"
        }
        self.assertEqual(len(found), 8)
        for (label, context, lead), case in found.items():
            start = 200 if label == "a0" else 260
            # The context ends `lead` points before the anomaly: it holds none of it.
            self.assertEqual(case["origin"], start - lead)
            self.assertEqual(case["span_start"], lead)
            np.testing.assert_array_equal(
                case["values"],
                np.float32(
                    np.sin(np.arange(start - lead - context, start - lead + 8) / 5)
                ),
            )
        # Three points long: all judged. Forty points long: those inside the forecast.
        self.assertEqual(found[("a0", 16, 2)]["span_length"], 3)
        self.assertEqual(found[("a1", 16, 2)]["span_length"], 6)
        self.assertEqual(found[("a1", 16, 5)]["span_length"], 3)
        self.assertEqual(found[("a1", 16, 5)]["anomaly_length"], 40)
        self.assertEqual(skipped, {})

    def test_no_anomaly_case_when_an_earlier_anomaly_would_be_in_the_context(self):
        cases, skipped = prepare_cases.build_cases(
            [series(anomalies=((200, 210), (230, 240)))], OPTIONS
        )
        made = {
            (c["anomaly"], c["context_length"], c["lead"])
            for c in cases
            if c["task"] == "anomaly"
        }
        # 20 normal points lie between the two: enough for a context of 16
        # with a lead of 2, not with a lead of 5 and not for a context of 32.
        self.assertIn(("a1", 16, 2), made)
        self.assertNotIn(("a1", 16, 5), made)
        self.assertNotIn(("a1", 32, 2), made)
        self.assertEqual(
            skipped, {"ucr: another anomaly lies right before this one": 3}
        )

    def test_an_anomaly_that_ended_earlier_may_lie_deep_in_a_long_context(self):
        options = prepare_cases.settings(
            {
                "operational_benchmarks": {
                    "horizon": 8,
                    "context_lengths": [16, 32],
                    "forecast": {"origins": {"cats": 3, "ucr": 3}, "anomaly_margin": 5},
                    "anomaly": {"lead_points": [2], "normal_points_before": 16},
                }
            }
        )
        cases, skipped = prepare_cases.build_cases(
            [series(anomalies=((200, 210), (230, 240)))], options
        )
        marks = {
            (c["anomaly"], c["context_length"]): c["earlier_anomaly"]
            for c in cases
            if c["task"] == "anomaly"
        }
        # The context of 32 reaches back to point 196, into the first anomaly;
        # its last 16 points and the lead are normal.
        self.assertEqual(marks[("a1", 32)], "yes")
        self.assertEqual(marks[("a1", 16)], "no")
        self.assertEqual(marks[("a0", 32)], "no")
        self.assertEqual(skipped, {})

    def test_cases_get_random_covariates_for_the_context_only(self):
        source = series(companions=7)
        cases, _ = prepare_cases.build_cases([source], OPTIONS)
        again, _ = prepare_cases.build_cases([source], OPTIONS)
        chosen = set()
        for case, same in zip(cases, again):
            names = case["covariates"]
            self.assertEqual(len(set(names)), 5)  # five different ones, the most
            self.assertEqual(names, same["covariates"])  # the same on every run
            chosen.add(tuple(names))
            values = case["covariate_values"].reshape(5, case["context_length"])
            start = case["origin"] - case["context_length"]
            for name, row in zip(names, values):
                np.testing.assert_array_equal(
                    row, np.float32(source.companions[name][start : case["origin"]])
                )
        self.assertGreater(len(chosen), 5)  # not the same five for every case
        plain, _ = prepare_cases.build_cases([series()], OPTIONS)
        self.assertEqual(plain[0]["covariates"], [])
        self.assertEqual(len(plain[0]["covariate_values"]), 0)
        with self.assertRaisesRegex(ValueError, "0 to 5"):
            prepare_cases.settings(
                {"operational_benchmarks": {"covariates": {"count": 6}}}
            )

    def test_settings_are_checked(self):
        for section, message in (
            ({"horizons": 5}, "Unknown settings"),
            ({"anomaly": {"leads": [1]}}, "Unknown settings"),
            ({"anomaly": {"lead_points": [112]}}, "begin inside the forecast"),
            ({"anomaly": {"quantiles": [0.6, 0.9]}}, "around 0.5"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                prepare_cases.settings({"operational_benchmarks": section})

    def test_cases_are_written_and_read_back_with_their_record(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {
                "paths": {"operational_benchmark_data": str(Path(folder) / "cases")},
                "operational_benchmarks": {},
            }
            with mock.patch.object(prepare_cases, "ucr_listing", return_value=[]):
                with mock.patch.object(
                    prepare_cases, "build_cases", lambda *_: prepare_cases_build()
                ):
                    record = prepare_cases.prepare(config, datasets=("ucr",), log=str)
            directory = prepare_cases.data_directory(config)
            cases, read = prepare_cases.load_cases(directory)
            self.assertEqual(read["cases"], len(cases))
            self.assertEqual(read["cases_sha256"], record["cases_sha256"])
            self.assertEqual(cases["values"].iloc[0].dtype, np.float32)
            (directory / "cases.parquet").write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "prepare again"):
                prepare_cases.load_cases(directory)


BUILD = prepare_cases.build_cases


def prepare_cases_build():
    return BUILD([series()], OPTIONS)


class ScoringTests(unittest.TestCase):
    def test_interval_is_the_two_quantiles_or_a_normal_one_through_them(self):
        median = np.array([[10.0, 20.0]])
        quantiles = normal_quantiles(median, 3.0)
        quantiles[:, 8] += 5  # a long upper tail: the 90% quantile lies further out
        anomaly = dict(OPTIONS["anomaly"])
        self.assertEqual(anomaly["interval"], "quantiles")
        got_median, lower, upper = scoring.intervals(quantiles, LEVELS, anomaly)
        np.testing.assert_allclose(got_median, median)
        np.testing.assert_allclose(lower, median - 1.2815515655 * 3)
        np.testing.assert_allclose(upper, median + 1.2815515655 * 3 + 5)
        # The normal interval: two standard deviations of the normal
        # distribution that has the same 10% and 90% quantiles.
        anomaly.update(interval="normal", interval_sigmas=2.0)
        _, lower, upper = scoring.intervals(
            normal_quantiles(median, 3.0), LEVELS, anomaly
        )
        np.testing.assert_allclose(lower, median - 6)
        np.testing.assert_allclose(upper, median + 6)
        anomaly["quantiles"] = [0.05, 0.95]
        with self.assertRaisesRegex(ValueError, "needs 0.05"):
            scoring.intervals(quantiles, LEVELS, anomaly)
        with self.assertRaisesRegex(ValueError, "quantiles.*or.*normal"):
            prepare_cases.settings(
                {"operational_benchmarks": {"anomaly": {"interval": "sigma"}}}
            )

    def test_points_are_judged_against_the_bound_on_their_side(self):
        got = scoring.surprise(
            np.array([1.0, 1.0, 5.0, 0.0, 9.0]),
            np.array([1.0, 2.0, 1.0, 2.0, 5.0]),  # median
            np.array([1.0, 2.0, -1.0, 1.0, 1.0]),  # lower bound
            np.array([1.0, 2.0, 3.0, 9.0, 7.0]),  # upper bound
        )
        # On the median; off an interval without width; twice the way to the
        # upper bound; twice the way to the lower bound; twice again.
        np.testing.assert_array_equal(got, [0.0, np.inf, 2.0, 2.0, 2.0])

    def test_a_window_can_need_more_than_one_point_outside(self):
        scores = np.array([[0.5, 2.0, 0.5, 0.5], [2.0, 2.0, 2.0, 0.5], [0.5] * 4])
        np.testing.assert_array_equal(scoring.flagged(scores, 0.0), [True, True, False])
        np.testing.assert_array_equal(
            scoring.flagged(scores, 0.5), [False, True, False]
        )

    def frame(self):
        """Two normal windows and two anomaly cases of one series, context 16."""
        context = np.tile([0.0, 1.0], 8)  # mean absolute step 1
        normal = np.zeros(8)
        spike = np.zeros(8)
        spike[4] = 9.0  # one anomalous point, at step 4
        early = np.zeros(8)
        early[1] = 9.0  # a spike in the normal lead, before the labelled anomaly
        rows = []
        for number, (task, future, start, length) in enumerate(
            (
                ("forecast", normal, -1, 0),
                ("forecast", early, -1, 0),
                ("anomaly", spike, 4, 2),
                ("anomaly", early, 4, 2),
            )
        ):
            rows.append(
                {
                    "case": number,
                    "dataset": "ucr",
                    "series": "example",
                    "category": "electricity",
                    "variant": "original",
                    "task": task,
                    "context_length": 16,
                    "origin": 100 + number,
                    "anomaly": f"a{number}" if task == "anomaly" else "",
                    "role": "anomaly" if task == "anomaly" else "",
                    "kind": "",
                    "lead": start,
                    "span_start": start,
                    "span_length": length,
                    "anomaly_length": length,
                    "values": np.concatenate([context, future]).astype(np.float32),
                }
            )
        return pd.DataFrame(rows)

    def test_detection_false_alarms_and_forecast_error(self):
        cases = self.frame()
        # Every model forecast: median 1, standard deviation 1, so the 10% and
        # 90% quantiles are 1.28 below and above the median.
        quantiles = normal_quantiles(np.ones((4, 8)), 1.0)
        options = {**OPTIONS, "horizon": 8}
        table = scoring.score_model(cases, quantiles, LEVELS, options).set_index("case")
        # All points are 1 off the median; the context's mean step is 1.
        self.assertAlmostEqual(table.loc[0, "mae"], 1.0)
        self.assertAlmostEqual(table.loc[0, "outside"], 0.0)
        self.assertAlmostEqual(table.loc[1, "outside"], 1 / 8)
        # The spike at step 4 lies in the anomaly's steps 4 and 5: detected.
        self.assertEqual(table.loc[2, "detected"], 1.0)
        self.assertAlmostEqual(table.loc[2, "peak"], 8 / 1.2815515655)
        self.assertEqual(table.loc[2, "flagged_before"], 0.0)
        # A spike at step 1 is outside the anomaly's steps: not a detection.
        self.assertEqual(table.loc[3, "detected"], 0.0)
        self.assertEqual(table.loc[3, "flagged_before"], 1.0)
        # Neither normal window has a flagged point in steps 4 and 5.
        self.assertEqual(table.loc[2, "false_alarm"], 0.0)
        self.assertEqual(table.loc[2, "separation"], 1.0)
        self.assertEqual(table.loc[3, "separation"], 0.5)  # as surprising as normal
        self.assertTrue(np.isnan(table.loc[0, "detected"]))

        # A model with a tenth of the spread flags every point of normal data.
        narrow = scoring.score_model(
            cases, normal_quantiles(np.ones((4, 8)), 0.1), LEVELS, options
        ).set_index("case")
        self.assertEqual(narrow.loc[3, "detected"], 1.0)
        self.assertEqual(narrow.loc[3, "false_alarm"], 1.0)

        results = scoring.summarise(
            cases,
            {"naive": table.reset_index(), "narrow": narrow.reset_index()},
            options,
        )
        anomaly = results["anomaly"]["ucr"]
        # One of two anomalies flagged, no normal window flagged.
        naive = anomaly["naive"]["overall"]
        self.assertEqual(
            (naive["recall"], naive["precision"], naive["false_alarm"]), (0.5, 1.0, 0.0)
        )
        self.assertAlmostEqual(naive["f1"], 2 / 3)
        self.assertEqual(naive["anomalies"], 2)
        # Flags everything: every anomaly found, half of what it flags is normal.
        everything = anomaly["narrow"]["overall"]
        self.assertEqual((everything["recall"], everything["precision"]), (1.0, 0.5))
        self.assertAlmostEqual(everything["f1"], 2 / 3)
        forecast = results["forecast"]["ucr"]
        self.assertEqual(forecast["naive"]["overall"]["cases"], 2)
        # 1 for the flat window; (7 + 8) / 8 for the one with the spike.
        self.assertAlmostEqual(forecast["naive"]["overall"]["mase_8_median"], 1.4375)
        self.assertEqual(forecast["narrow"]["overall"]["beats_naive"], 0.0)
        page = scoring.leaderboard(results, options, {"Basin": "epoch 3"})
        self.assertIn("### UCR (operational series)", page)
        self.assertIn("| naive | 2 | 0.667 | 100.0% | 50.0% | 0.0% |", page)
        json.dumps(results)  # everything can be saved

    def test_breakdowns_are_in_numeric_order(self):
        values = sorted(["10", "2", "1", "noise"], key=scoring._order)
        self.assertEqual(values, ["1", "2", "10", "noise"])


class StageTests(unittest.TestCase):
    def setUp(self):
        if not (ROOT / "benchmarks/fakebench-ts/src/fakebench_ts").is_dir():
            self.skipTest("the FakeBench-TS submodule is not checked out")
        import operational_benchmarks

        self.stage = operational_benchmarks

    def test_models_are_called_with_one_context_length_at_a_time(self):
        cases, _ = prepare_cases.build_cases([series()], OPTIONS)
        cases = pd.DataFrame(cases)
        calls = []

        def forecaster(contexts, horizon):
            calls.append({len(c) for c in contexts})
            return np.stack(
                [np.tile(c[-1], (9, horizon)) + np.arange(9)[:, None] for c in contexts]
            )

        quantiles = self.stage.forecast_cases(forecaster, cases, 8, batch_size=4)
        self.assertTrue(all(len(lengths) == 1 for lengths in calls))
        for position, case in enumerate(cases.itertuples()):
            last = case.values[case.context_length - 1]
            np.testing.assert_allclose(quantiles[position, 4], last + 4, rtol=1e-6)
        with self.assertRaisesRegex(ValueError, "needs quantile forecasts"):
            self.stage.forecast_cases(
                lambda contexts, horizon: np.zeros((len(contexts), horizon)),
                cases,
                8,
                batch_size=4,
            )

    def test_covariates_go_only_to_models_that_take_them(self):
        cases, _ = prepare_cases.build_cases([series(companions=7)], OPTIONS)
        cases = pd.DataFrame(cases)
        seen = []

        def forecaster(contexts, horizon, past_covariates=None):
            seen.append((len(contexts[0]), past_covariates))
            return np.zeros((len(contexts), 9, horizon))

        self.stage.forecast_cases(forecaster, cases, 8, batch_size=4)
        self.assertTrue(all(past is None for _, past in seen))
        seen.clear()
        forecaster.uses_covariates = True
        self.stage.forecast_cases(forecaster, cases, 8, batch_size=len(cases))
        for length, past in seen:
            members = cases[cases["context_length"] == length]
            self.assertEqual(past.shape, (len(members), 5, length))
            np.testing.assert_array_equal(
                past[0], members["covariate_values"].iloc[0].reshape(5, length)
            )

    def test_a_run_with_the_naive_baseline_writes_every_result_file(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {
                "paths": {
                    "operational_benchmark_data": str(Path(folder) / "cases"),
                    "operational_benchmarks": str(Path(folder) / "results"),
                },
                "training": {"cpu_threads": 4},
                "operational_benchmarks": {
                    "horizon": 8,
                    "context_lengths": [16, 32],
                    "forecast": {"origins": {"cats": 3, "ucr": 3}, "anomaly_margin": 5},
                    "anomaly": {"lead_points": [2, 5]},
                },
            }
            with mock.patch.object(prepare_cases, "ucr_listing", return_value=[]):
                with mock.patch.object(
                    prepare_cases, "build_cases", lambda *_: prepare_cases_build()
                ):
                    prepare_cases.prepare(config, datasets=("ucr",), log=str)
            arguments = mock.Mock(models=["naive"], batch_size=None, limit=None)
            output = self.stage.run(config, arguments)
            names = {p.name for p in output.iterdir()}
            self.assertEqual(
                names,
                {
                    "forecasts",
                    "run.json",
                    "results.json",
                    "leaderboard.md",
                    "per_case.parquet",
                    "examples.html",
                },
            )
            results = json.loads((output / "results.json").read_text())
            self.assertEqual(results["anomaly"]["ucr"]["naive"]["overall"]["cases"], 4)
            before = (output / "leaderboard.md").read_text()
            self.stage.rescore(config, output.name)
            self.assertEqual((output / "leaderboard.md").read_text(), before)
            # Another interval only changes the scores: the same cases serve.
            config["operational_benchmarks"]["anomaly"]["interval"] = "normal"
            self.stage.rescore(config, output.name)
            self.assertIn(
                "standard deviations", (output / "leaderboard.md").read_text()
            )
            self.assertNotIn("standard deviations", before)
            # Cases prepared with other settings are refused, not silently used.
            config["operational_benchmarks"]["anomaly"]["lead_points"] = [3]
            with self.assertRaisesRegex(ValueError, "Prepare again"):
                self.stage.run(config, arguments)

    def test_holt_winters_gives_an_interval_that_holds_most_of_a_random_walk(self):
        rng = np.random.default_rng(0)
        walks = np.cumsum(rng.normal(size=(40, 312)), axis=1)
        forecaster = self.stage.holt_winters(workers=1)
        quantiles = forecaster([walk[:200] for walk in walks], 112)
        self.assertEqual(quantiles.shape, (40, 9, 112))
        self.assertTrue((np.diff(quantiles, axis=1) > 0).all())
        # The interval widens with the step, as the uncertainty of a walk does.
        width = quantiles[:, 8] - quantiles[:, 0]
        self.assertTrue((width[:, -1] > 3 * width[:, 0]).all())
        future = walks[:, 200:]
        inside = (future >= quantiles[:, 0]) & (future <= quantiles[:, 8])
        self.assertGreater(inside.mean(), 0.65)
        self.assertLess(inside.mean(), 0.92)

    def test_holt_winters_runs_without_a_gpu_and_is_scored_per_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {
                "paths": {
                    "operational_benchmark_data": str(Path(folder) / "cases"),
                    "operational_benchmarks": str(Path(folder) / "results"),
                },
                "training": {"cpu_threads": 4},
                "operational_benchmarks": {
                    "horizon": 8,
                    "context_lengths": [16, 32],
                    "forecast": {"origins": {"cats": 3, "ucr": 3}, "anomaly_margin": 5},
                    "anomaly": {"lead_points": [2, 5]},
                    "holt_winters_workers": 1,
                },
            }
            with mock.patch.object(prepare_cases, "ucr_listing", return_value=[]):
                with mock.patch.object(
                    prepare_cases, "build_cases", lambda *_: prepare_cases_build()
                ):
                    prepare_cases.prepare(config, datasets=("ucr",), log=str)
            arguments = mock.Mock(
                models=["naive", "holtwinters"], batch_size=None, limit=None
            )
            output = self.stage.run(config, arguments)
            results = json.loads((output / "results.json").read_text())
            for task in ("forecast", "anomaly"):
                self.assertEqual(sorted(results[task]["ucr"]), ["holtwinters", "naive"])
            self.assertIn("holtwinters", self.stage.CPU_MODELS)
            self.assertIn("| holtwinters |", (output / "leaderboard.md").read_text())


if __name__ == "__main__":
    unittest.main()
