"""The two public anomaly datasets as plain series with labelled anomalies.

CATS (Controlled Anomalies Time Series, Solenix): one simulated system sampled
at 1 Hz for 5 million seconds. 4 operator commands, 3 external stimuli and 10
telemetry readings, without noise. 200 injected anomalies, each with the
telemetry channel it starts in (root cause) and the one it spreads to
(affected). The first million rows have none.

UCR Time Series Anomaly Archive 2021: 250 separate univariate series, each
with exactly one labelled anomaly. Most of them are recordings of a body
(electrocardiograms, respiration, blood pressure, gait). `UCR_FAMILIES`
records for every family what it measures; only the families that measure a
machine, an electricity grid or the weather count as operational data.

Microsoft Cloud Monitoring Dataset: 60 separate series from production
telemetry of Microsoft services, per minute or per hour, with anomalies
labelled by the engineers who run them. Most have several anomalies, some
none. `CLOUD_DOMAINS` records what each folder measures; the counts of in-app
purchases are commercial data and are left out.

Micromill Continuous Caster (Carnegie Mellon University): 22 sensors of an
industrial steel caster, twice a second, in four recordings of one to three
hours. It has no anomaly labels, so it serves the forecast task only. One
recording ends in a breakout and a stop and one begins with a start-up; those
periods are not normal operation and no case is taken from them.

Positions are zero-based and ranges are half-open: an anomaly covers
`values[start:end]`.
"""

import ast
import csv
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

CATS_COMMANDS = ("aimp", "amud", "adbr", "adfl")
CATS_STIMULI = ("arnd", "asin1", "asin2")
CATS_TELEMETRY = (
    "bed1",
    "bed2",
    "bfo1",
    "bfo2",
    "bso1",
    "bso2",
    "bso3",
    "ced1",
    "cfo1",
    "cso1",
)

# Categories that count as operational data: measurements of a machine, a
# plant, an electricity grid or (borderline, but included) the weather.
OPERATIONAL = (
    "simulated_system",
    "electricity",
    "weather",
    "spacecraft_telemetry",
    "cloud_service_telemetry",
    "software_crash_rate",
    "industrial_process",
)

# The caster is casting when all of these hold (sensor name, lowest value):
# the strand moves, there is steel in the mold and the tundish, the strand has
# reached the pyrometer, and the superheat sensor gives a reading.
CASTER_CASTING = (
    ("Casting speed", 1.0),  # m/min
    ("Mold level", 50.0),  # mm
    ("Tundish weight", 1.0),  # tonne
    ("Pyrometer reading", 1000.0),  # K
    ("Superheat", 0.0),  # K
)

# What each folder of the Microsoft Cloud Monitoring Dataset measures, from
# the notes in its metadata files.
CLOUD_DOMAINS = {
    "application-crash-rate-1": (
        "software_crash_rate",
        "crash rate of a Windows application, per hour",
    ),
    "application-crash-rate-2": (
        "software_crash_rate",
        "crash rate of a Windows application, per hour",
    ),
    "consumer-purchase-rate": (
        "commercial",
        "count of in-app purchases in one market, per hour",
    ),
    "data-ingress-rate": (
        "cloud_service_telemetry",
        "rate of data flowing into a cloud service, per minute",
    ),
    "ecommerce-api-incoming-rps": (
        "cloud_service_telemetry",
        "requests per second arriving at an e-commerce API, per hour",
    ),
    "middle-tier-api-dependency-latency": (
        "cloud_service_telemetry",
        "latency of a cloud service's calls to its backend services, per hour",
    ),
    "mongodb-application-rps": (
        "cloud_service_telemetry",
        "query rate to a MongoDB cluster from one application, per minute",
    ),
    "mongodb-machine-rps": (
        "cloud_service_telemetry",
        "query rate to one MongoDB server, per minute",
    ),
    "service-unavailable": (
        "cloud_service_telemetry",
        "'service unavailable' answers of a cloud service, per minute",
    ),
}

# What each family of UCR series measures, from the archive's own description
# (UCR_AnomalyDataSets.pptx). The pattern is matched against the series name
# without the DISTORTED or NOISE prefix; the first match counts.
UCR_FAMILIES = (
    (
        r"PowerDemand|Italianpowerdemand",
        "electricity",
        "Italian electricity demand, 1995 to 1998",
    ),
    (
        r"CIMIS44AirTemperature",
        "weather",
        "hourly air temperature at CIMIS weather station 44 (Riverside)",
    ),
    (
        r"Tkeep\w+MARS",
        "spacecraft_telemetry",
        "telemetry channel of a NASA spacecraft (KDD 2018)",
    ),
    (r"gait|park3m", "human_motion", "force plate under a walking patient"),
    (r"WalkingAceleration", "human_motion", "accelerometer worn while walking"),
    (r"GP711Marker", "human_motion", "motion-capture marker on a walker's shoe"),
    (r"weallwalk", "human_motion", "inertial sensors carried while walking"),
    (
        r"ECG|sddb|apneaecg|ltstdb|qtdb|sel840|s20101|longtermecg|STAFFIII|Fantasia",
        "physiology",
        "electrocardiogram of a patient",
    ),
    (r"CHARIS", "physiology", "heart and pressure signals of a brain-injury patient"),
    (r"BIDMC", "physiology", "pulse signal (plethysmogram) of a patient"),
    (r"resperation", "physiology", "respiration of a sleeping patient"),
    (r"tilt", "physiology", "arterial blood pressure of a person on a tilt table"),
    (r"InternalBleeding", "physiology", "arterial blood pressure of pigs"),
    (r"taichidb", "physiology", "recording from a physiology database (Tai Chi)"),
    (r"insectEPG|Lab2Cmac", "biology", "feeding signal of an insect"),
    (r"MesoplodonDensirostris", "biology", "accelerometer on a beaked whale"),
)
UCR_NAME = re.compile(
    r"(?P<number>\d{3})_UCR_Anomaly_(?P<name>[A-Za-z0-9]+)"
    r"_(?P<train>\d+)_(?P<begin>\d+)_(?P<end>\d+)\.txt"
)


@dataclass
class Anomaly:
    start: int
    end: int
    label: str  # a name that is unique within the dataset
    role: str = "anomaly"  # for CATS: root_cause or affected
    kind: str = ""  # for CATS: the anomaly category in the metadata


@dataclass
class Series:
    dataset: str
    name: str
    category: str
    description: str
    values: np.ndarray
    anomalies: list  # the anomalies to detect in this series
    # Every range that is not normal behaviour, also where another channel of
    # the same system is the one that is anomalous.
    abnormal: list = field(default_factory=list)
    variant: str = "original"  # UCR: original, distorted or noise
    # Other series measured at the same moments (name -> values), from which a
    # case's covariates are drawn. CATS: the system's other 16 channels.
    companions: dict = field(default_factory=dict)

    @property
    def operational(self):
        return self.category in OPERATIONAL


def ucr_family(name):
    """(base name, variant, category, description) of a UCR series name."""
    variant, base = "original", name
    for prefix in ("DISTORTED", "NOISE"):
        if name.startswith(prefix):
            variant, base = prefix.lower(), name[len(prefix) :]
    for pattern, category, description in UCR_FAMILIES:
        if re.search(pattern, base):
            return base, variant, category, description
    return base, variant, "unknown", "not described in the archive"


def ucr_folder(path):
    """The folder with the 250 text files, wherever the archive was unpacked."""
    path = Path(path)
    for folder in (path, *sorted(p.parent for p in path.rglob("001_UCR_Anomaly_*"))):
        if any(folder.glob("001_UCR_Anomaly_*.txt")):
            return folder
    raise FileNotFoundError(f"No UCR anomaly series (NNN_UCR_Anomaly_*.txt) in {path}")


def ucr_listing(path):
    """One record per file, read from the file names alone."""
    records = []
    for file in sorted(ucr_folder(path).glob("*_UCR_Anomaly_*.txt")):
        match = UCR_NAME.fullmatch(file.name)
        if not match:
            raise ValueError(f"Unexpected UCR file name: {file.name}")
        base, variant, category, description = ucr_family(match["name"])
        records.append(
            {
                "file": file,
                "name": f"{match['number']}_{match['name']}",
                "base": base,
                "variant": variant,
                "category": category,
                "description": description,
                "operational": category in OPERATIONAL,
                # The names count from 1 and include the last anomalous point.
                "anomaly": (int(match["begin"]) - 1, int(match["end"])),
                "train_end": int(match["train"]),
            }
        )
    return records


def load_ucr(record):
    # Most files have one value per line; a few have all values on one line.
    values = np.array(record["file"].read_text().split(), dtype=np.float64)
    start, end = record["anomaly"]
    if not 0 <= start < end <= len(values):
        raise ValueError(f"{record['file'].name}: anomaly outside the series")
    return Series(
        dataset="ucr",
        name=record["name"],
        category=record["category"],
        description=record["description"],
        values=values,
        anomalies=[Anomaly(start, end, record["name"])],
        abnormal=[(start, end)],
        variant=record["variant"],
    )


def cloud_listing(path):
    """One record per series of the Microsoft Cloud Monitoring Dataset."""
    root = Path(path) / "data"
    if not root.is_dir():
        raise FileNotFoundError(
            f"No Cloud Monitoring Dataset in {path}. Fetch it with: git clone "
            f"https://github.com/microsoft/cloud-monitoring-dataset {path}"
        )
    records = []
    for file in sorted(root.glob("*/*.csv")):
        category, description = CLOUD_DOMAINS.get(
            file.parent.name, ("unknown", "not described")
        )
        records.append(
            {
                "file": file,
                "name": f"{file.parent.name}/{file.stem}",
                "category": category,
                "description": description,
                "operational": category in OPERATIONAL,
            }
        )
    return records


def load_cloud(record):
    """The series on an even time grid; moments without a value are missing (NaN).

    A few files repeat some timestamps or skip some; the first row of a
    timestamp counts.
    """
    import pandas as pd

    frame = pd.read_csv(record["file"])
    frame["TimeStamp"] = pd.to_datetime(frame["TimeStamp"], utc=True)
    frame = frame.drop_duplicates("TimeStamp").set_index("TimeStamp").sort_index()
    step = frame.index.to_series().diff().mode()[0]
    frame = frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq=step))
    labelled = frame["Label"].fillna(0).to_numpy() > 0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], labelled, [0]])))
    ranges = [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2])]
    return Series(
        dataset="cloud",
        name=record["name"],
        category=record["category"],
        description=record["description"],
        values=frame["Value"].to_numpy(dtype=np.float64),
        anomalies=[
            Anomaly(start, end, f"{record['name']}#{number}")
            for number, (start, end) in enumerate(ranges)
        ],
        abnormal=ranges,
    )


def caster_listing(path):
    """One record per recording of the Micromill caster dataset."""
    files = sorted(Path(path).rglob("*CCOP-*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No caster recordings (CCOP-*.csv) in {path}. Download and unpack "
            "https://doi.org/10.1184/R1/33432847 there."
        )
    return [
        {"file": file, "name": re.search(r"CCOP-\d+", file.name)[0]} for file in files
    ]


def load_caster(record, samples_per_point=1):
    """One series per sensor of a recording, each with the other 21 as companions.

    `abnormal` is every stretch in which the caster is not casting (see
    `CASTER_CASTING`): before a start-up has settled, and from a breakout on.
    With `samples_per_point` above 1 every point is the mean of that many
    samples (two per second), abnormal when any of them is.
    """
    import pandas as pd

    frame = pd.read_csv(record["file"])
    seconds = frame.pop(frame.columns[0]).to_numpy()
    if not np.allclose(np.diff(seconds), 0.5):
        raise ValueError(f"{record['file'].name}: expected one row every 0.5 s")
    # "Casting speed (m/min)" -> "Casting speed"
    frame.columns = [re.sub(r"\s*\(.*\)$", "", name) for name in frame.columns]
    casting = np.ones(len(frame), dtype=bool)
    for name, lowest in CASTER_CASTING:
        casting &= frame[name].to_numpy() > lowest  # a missing value is not casting
    casting |= frame.isna().all(axis=1).to_numpy()  # a missing row is only missing
    step = int(samples_per_point)
    if step < 1:
        raise ValueError("samples_per_point must be a whole number of at least 1")
    points = len(frame) // step
    stopped = ~casting[: points * step].reshape(points, step).all(axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], stopped, [0]])))
    abnormal = [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2])]
    values = {
        name: frame[name]
        .to_numpy(dtype=np.float64)[: points * step]
        .reshape(points, step)
        .mean(axis=1)
        for name in frame.columns
    }
    return [
        Series(
            dataset="caster",
            name=f"{record['name']}/{name}",
            category="industrial_process",
            description=f"sensor of an industrial steel caster, one point per "
            f"{step / 2:g} s",
            values=values[name],
            anomalies=[],
            abnormal=abnormal,
            companions={n: v for n, v in values.items() if n != name},
        )
        for name in frame.columns
    ]


def cats_anomalies(path, first):
    """The rows of metadata.csv as positions; `first` is the first timestamp."""
    rows = []
    with open(Path(path) / "metadata.csv", newline="", encoding="utf-8") as file:
        for number, row in enumerate(csv.DictReader(file)):
            start, end = (
                int((datetime.fromisoformat(row[key]) - first).total_seconds())
                for key in ("start_time", "end_time")
            )
            rows.append(
                {
                    "label": f"anomaly_{number:03d}",
                    "start": start,
                    "end": end,  # the end time is the first normal second
                    "root_cause": row["root_cause"],
                    "affected": list(ast.literal_eval(row["affected"])),
                    "kind": row["category"],
                }
            )
    return rows


def load_cats(path, channels=CATS_TELEMETRY, seconds_per_point=1):
    """One series per telemetry channel, with the anomalies that involve it.

    Every series also carries the system's other channels (commands, stimuli
    and the other telemetry) as `companions`. With `seconds_per_point` above 1
    every point is the mean of that many seconds, and a point counts as
    anomalous when any of its seconds is.
    """
    import pyarrow as pa
    import pyarrow.csv as pc

    with open(Path(path) / "data.csv", encoding="utf-8") as file:
        header = file.readline().strip().split(",")
    known = (*CATS_COMMANDS, *CATS_STIMULI, *CATS_TELEMETRY)
    columns = [name for name in header if name in known or name in channels]
    table = pc.read_csv(
        Path(path) / "data.csv",
        convert_options=pc.ConvertOptions(
            include_columns=["timestamp", "y", *columns],
            column_types={name: pa.float32() for name in ("y", *columns)},
        ),
    )
    seconds = table["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)
    if not (np.diff(seconds) == 1).all():
        raise ValueError("CATS is expected to have one row per second")
    rows = cats_anomalies(
        path, datetime(1970, 1, 1) + timedelta(seconds=int(seconds[0]))
    )
    labelled = table["y"].to_numpy() > 0
    marked = np.zeros(len(labelled), dtype=bool)
    for row in rows:
        marked[row["start"] : row["end"]] = True
    if not np.array_equal(marked, labelled):
        raise ValueError("CATS metadata.csv and the y column of data.csv disagree")
    step = int(seconds_per_point)
    if step < 1:
        raise ValueError("seconds_per_point must be a whole number of at least 1")
    points = len(labelled) // step
    for row in rows:
        row["start"], row["end"] = row["start"] // step, -(-row["end"] // step)
    abnormal = sorted((row["start"], min(row["end"], points)) for row in rows)
    values = {
        name: table[name]
        .to_numpy()
        .astype(np.float64)[: points * step]
        .reshape(points, step)
        .mean(axis=1)
        for name in columns
    }
    series = []
    for channel in channels:
        anomalies = [
            Anomaly(
                row["start"],
                row["end"],
                row["label"],
                "root_cause" if row["root_cause"] == channel else "affected",
                row["kind"],
            )
            for row in rows
            if row["root_cause"] == channel or channel in row["affected"]
        ]
        series.append(
            Series(
                dataset="cats",
                name=channel,
                category="simulated_system",
                description="telemetry reading of a simulated system, no noise, "
                f"one point per {step} s",
                values=values[channel],
                companions={n: v for n, v in values.items() if n != channel},
                anomalies=anomalies,
                abnormal=abnormal,
            )
        )
    return series
