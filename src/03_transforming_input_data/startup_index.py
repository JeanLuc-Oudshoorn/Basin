"""The few record fields that training start-up reads, kept in flat arrays.

Before the first update, training passes over the corpus seven times: twice to
split it, once to check that validation tables are not trained on, twice to
plan batches and twice to compute the digests a checkpoint is bound to. The
record metadata is JSON, about 800 bytes per record. For 23 million records
that is 24 GiB, more than the page cache holds, so each pass read all of it
from disk again: 46 minutes of a 58-minute start.

Those passes use ten small fields. This index holds exactly those fields for
every record, in arrays that are memory mapped, so a pass costs seconds. It is
built with one pass over the metadata, saved beside the corpus and reused by
every later start. A record's other fields are still read from its shard.

Layout: records of one table follow each other and share their source fields,
which are stored once per run of equal values. Fields with few distinct values
are stored as codes into a list of those values. Record ids are stored as text.

Start-up reaches the index through `scan`, `light` and `fetch` below. They take
the record sequences of sharded_records.py as they are: that file is part of
the identity of a transformed corpus, so it is left unchanged.
"""

import hashlib
import json
import os
import shutil
import sys
from array import array
from bisect import bisect_right
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from loguru import logger
from sharded_records import EncodedRecords, RecordRef, ShardBatch, ShardedRecords
from transform_progress import atomic_write

FORMAT = "startup_index_v1"
DIRECTORY = ".startup_index"
# Text shared by the consecutive records of one table; text of each record.
TABLE_TEXT = ("source", "source_key", "source_sha256")
RECORD_TEXT = ("id",)
# Any JSON value; indexed while the number of distinct values stays small.
CODED = (
    "source_rows",
    "split",
    "cutoff",
    "context_rows",
    "label_rows",
    "roles",
    "_shard_batch_keys",
)
VOCABULARY_LIMIT = 1 << 20
ABSENT = object()  # the record does not have this field
_ENCODING = ("utf-8", "surrogatepass")  # every str read from JSON survives it


def corpus_key(shards):
    """Identity of the indexed corpus: its shards, in order."""
    content = [FORMAT, [[shard["sha256"], shard["record_count"]] for shard in shards]]
    return hashlib.sha256(json.dumps(content).encode()).hexdigest()


class _File:
    """Writes one index file and digests it, so it need not be read back."""

    def __init__(self, path):
        self.path, self.stream = path, path.open("wb")
        self.sha256, self.bytes = hashlib.sha256(), 0

    def write(self, data):
        self.sha256.update(data)
        self.bytes += len(data)
        self.stream.write(data)

    def close(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        return {"sha256": self.sha256.hexdigest(), "bytes": self.bytes}


def _save(path, data):
    file = _File(path)
    file.write(data)
    return file.close()


class _TextWriter:
    def __init__(self, directory, name):
        self.directory, self.name = directory, name
        self.blob = _File(directory / f"{name}.bin")
        self.ends, self.missing = array("Q"), bytearray()
        self.pending, self.failed = bytearray(), False

    def add(self, value):
        if self.failed:
            return
        if value is ABSENT:
            self.missing.append(1)
        elif type(value) is str:
            self.missing.append(0)
            self.pending += value.encode(*_ENCODING)
        else:
            self.failed = True
            return
        self.ends.append(self.blob.bytes + len(self.pending))
        if len(self.pending) >= 1 << 20:
            self.blob.write(bytes(self.pending))
            self.pending.clear()

    def close(self):
        """Files written for this field, or None when a value was not text."""
        self.blob.write(bytes(self.pending))
        files = {f"{self.name}.bin": self.blob.close()}
        if self.failed:
            return None
        path = self.directory / f"{self.name}.ends.u64"
        files[path.name] = _save(path, self.ends.tobytes())
        if any(self.missing):
            path = self.directory / f"{self.name}.missing.u8"
            files[path.name] = _save(path, bytes(self.missing))
        return files


class _CodeWriter:
    def __init__(self):
        self.codes, self.vocabulary = array("i"), []
        self.scalars, self.texts, self.failed = {}, {}, False

    def add(self, value):
        if self.failed:
            return
        if value is ABSENT:
            self.codes.append(-1)
            return
        # bool is not int here: True would otherwise share the code of 1.
        if type(value) is int or type(value) is str:
            known, key = self.scalars, value
        else:
            known, key = self.texts, json.dumps(value)
        code = known.get(key)
        if code is None:
            code = known[key] = len(self.vocabulary)
            self.vocabulary.append(key if known is self.texts else json.dumps(key))
            if len(self.vocabulary) > VOCABULARY_LIMIT:
                self.failed = True
                self.codes = self.vocabulary = self.scalars = self.texts = None
                return
        self.codes.append(code)


class _Text:
    """One text field: the values end to end, and where each one ends."""

    def __init__(self, blob, ends, missing):
        self.blob, self.ends, self.missing = blob, ends, missing

    def values(self, start, stop):
        first = int(self.ends[start - 1]) if start else 0
        ends = (self.ends[start:stop].astype(np.int64) - first).tolist()
        data = self.blob[first : first + ends[-1]].tobytes()
        starts = [0, *ends[:-1]]
        text = data.decode(*_ENCODING)
        if len(text) == len(data):  # ASCII: a byte offset is a character offset
            values = [text[a:b] for a, b in zip(starts, ends)]
        else:
            values = [data[a:b].decode(*_ENCODING) for a, b in zip(starts, ends)]
        if self.missing is not None:
            for position in np.flatnonzero(self.missing[start:stop]).tolist():
                values[position] = ABSENT
        return values


def _array(path, dtype):
    """Map the file; Windows cannot replace or delete a file that is mapped."""
    if os.name == "nt" or not path.stat().st_size:
        return np.fromfile(path, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r")


class StartupIndex:
    def __init__(self, directory, manifest):
        self.directory, self.key = directory, manifest["corpus"]
        self.fields = tuple(manifest["fields"])
        # What a light record answers without its shard. A whole record has
        # "batch" in place of the saved tensor names.
        self.known = frozenset(self.fields) | (
            {"batch"} if "_shard_batch_keys" in self.fields else frozenset()
        )
        records = manifest["records"]
        self.table = _array(directory / "table.i32", np.int32)
        if len(self.table) != records:
            raise ValueError("Start-up index does not cover the corpus")
        self.texts, self.coded = {}, {}
        for name in self.fields:
            if name in CODED:
                codes = _array(directory / f"{name}.i32", np.int32)
                values = [json.loads(text) for text in manifest["vocabularies"][name]]
                self.coded[name] = (codes, [*values, ABSENT])  # code -1 is ABSENT
                if len(codes) != records:
                    raise ValueError(f"Start-up index field {name} is incomplete")
                continue
            ends = _array(directory / f"{name}.ends.u64", np.uint64)
            missing = directory / f"{name}.missing.u8"
            self.texts[name] = _Text(
                _array(directory / f"{name}.bin", np.uint8),
                ends,
                _array(missing, np.uint8)
                if missing.name in manifest["files"]
                else None,
            )
            expected = manifest["tables"] if name in TABLE_TEXT else records
            if len(ends) != expected:
                raise ValueError(f"Start-up index field {name} is incomplete")

    @classmethod
    def open(cls, corpus_directory, shards):
        """The saved index of this corpus, or None when there is no usable one."""
        directory = Path(corpus_directory) / DIRECTORY
        started = perf_counter()
        try:
            manifest = json.loads((directory / "index.json").read_text("utf-8"))
            if (
                manifest["format"] != FORMAT
                or manifest["corpus"] != corpus_key(shards)
                or manifest["byteorder"] != sys.byteorder
            ):
                raise ValueError("it was built for another corpus or format")
            total = 0
            for name, expected in manifest["files"].items():
                with (directory / name).open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected["sha256"]:
                    raise ValueError(f"{name} does not match its checksum")
                total += expected["bytes"]
            index = cls(directory, manifest)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.warning("Start-up index at {} is not usable: {}", directory, error)
            return None
        logger.info(
            "Start-up index: {:,} records, {:.2f} GiB, verified in {:.1f}s; "
            "start-up passes read it instead of the record metadata",
            manifest["records"],
            total / 2**30,
            perf_counter() - started,
        )
        return index

    @classmethod
    def build(cls, corpus_directory, shards, metadata):
        """Index every record; `metadata(i)` returns the records of shard i."""
        corpus_directory = Path(corpus_directory)
        records = sum(shard["record_count"] for shard in shards)
        logger.info(
            "Start-up index: building for {:,} records with one pass over the "
            "record metadata; later starts reuse it",
            records,
        )
        if array("i").itemsize != 4 or array("Q").itemsize != 8:
            raise OSError("Start-up index needs 32-bit int and 64-bit long long")
        started = perf_counter()
        temporary = corpus_directory / f"{DIRECTORY}.{uuid4().hex}.tmp"
        temporary.mkdir(parents=True)
        texts = {}
        try:
            for name in TABLE_TEXT + RECORD_TEXT:
                texts[name] = _TextWriter(temporary, name)
            coded = {name: _CodeWriter() for name in CODED}
            table, tables, previous = array("i"), 0, None
            for shard in range(len(shards)):
                for record in metadata(shard):
                    source = tuple(record.get(name, ABSENT) for name in TABLE_TEXT)
                    if source != previous:
                        for name, value in zip(TABLE_TEXT, source):
                            texts[name].add(value)
                        previous = source
                        tables += 1
                    table.append(tables - 1)
                    for name in RECORD_TEXT:
                        texts[name].add(record.get(name, ABSENT))
                    for name, writer in coded.items():
                        writer.add(record.get(name, ABSENT))
            if len(table) != records:
                raise ValueError("Record metadata does not cover the corpus")
            files = {"table.i32": _save(temporary / "table.i32", table.tobytes())}
            fields, vocabularies = [], {}
            for name, writer in texts.items():
                written = writer.close()
                if written is None:
                    (temporary / f"{name}.bin").unlink()
                else:
                    fields.append(name)
                    files.update(written)
            for name, writer in coded.items():
                if not writer.failed:
                    fields.append(name)
                    vocabularies[name] = writer.vocabulary
                    path = temporary / f"{name}.i32"
                    files[path.name] = _save(path, writer.codes.tobytes())
            skipped = [
                name for name in TABLE_TEXT + RECORD_TEXT + CODED if name not in fields
            ]
            if skipped:
                logger.warning(
                    "Start-up index: {} not indexed (not text, or too many distinct "
                    "values); start-up reads these from the record metadata",
                    skipped,
                )
            manifest = {
                "format": FORMAT,
                "corpus": corpus_key(shards),
                "byteorder": sys.byteorder,
                "records": records,
                "tables": tables,
                "fields": fields,
                "vocabularies": vocabularies,
                "files": files,
            }
            _save(temporary / "index.json", json.dumps(manifest).encode("utf-8"))
            final = corpus_directory / DIRECTORY
            if final.exists():
                stale = corpus_directory / f"{DIRECTORY}.{uuid4().hex}.old"
                os.replace(final, stale)
                shutil.rmtree(stale, ignore_errors=True)
            os.replace(temporary, final)
        finally:
            for writer in texts.values():
                writer.blob.stream.close()
            shutil.rmtree(temporary, ignore_errors=True)
        logger.info(
            "Start-up index: built in {:.1f}s, {:.2f} GiB at {}",
            perf_counter() - started,
            sum(file["bytes"] for file in files.values()) / 2**30,
            final,
        )
        return cls(final, manifest)

    def columns(self, start, stop):
        """(field, values) for records start..stop-1; ABSENT where a record has none."""
        if start == stop:
            return []
        columns = []
        tables = self.table[start:stop].tolist()
        first = tables[0]
        for name in TABLE_TEXT:
            if name in self.texts:
                values = self.texts[name].values(first, tables[-1] + 1)
                columns.append((name, [values[table - first] for table in tables]))
        for name in RECORD_TEXT:
            if name in self.texts:
                columns.append((name, self.texts[name].values(start, stop)))
        for name, (codes, values) in self.coded.items():
            columns.append(
                (name, list(map(values.__getitem__, codes[start:stop].tolist())))
            )
        return columns


class LightRecord(dict):
    """The indexed fields of one record, for the passes that run before training.

    Reading a field costs no disk access. Any other field is read from the
    record's shard on request, so `record[key]`, `record.get(key)` and
    `key in record` answer as the whole record does. The dictionary itself
    holds only the indexed fields and therefore refuses to be listed or
    copied. Its values are shared between records: do not change them.
    """

    __slots__ = ("owner", "index", "indexed")

    def __missing__(self, key):
        if key in self.indexed:
            raise KeyError(key)  # an indexed field that this record lacks
        return self.owner[self.index][key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        if dict.__contains__(self, key):
            return True
        return key not in self.indexed and key in self.owner[self.index]

    def _partial(self, *_):
        raise TypeError(
            "A light record holds only its indexed fields; read the whole record "
            "from its sequence to list or copy it"
        )

    __iter__ = __len__ = keys = values = items = copy = _partial


def _whole(raw, shard, local, metadata):
    """One record from its saved metadata, as ShardedRecords.__getitem__ makes it."""
    record = dict(metadata)
    keys = record.pop("_shard_batch_keys", None)
    if keys is not None:
        record["batch"] = ShardBatch(raw, shard, local, tuple(keys))
    return record


def _saved_selection(path, selection):
    """Record metadata an earlier `fetch` saved for this selection, or None.

    The file is a line naming the corpus, the selection and the checksum of
    what follows, then the records as one JSON list.
    """
    try:
        header, _, content = path.read_bytes().partition(b"\n")
        if json.loads(header) == [*selection, hashlib.sha256(content).hexdigest()]:
            return json.loads(content)
    except (OSError, ValueError):
        return None
    return None


class _Corpus:
    """Start-up access to one sharded corpus: its index and one decoded shard."""

    def __init__(self, raw):
        self.raw = raw
        self.index = None  # False when the index cannot be kept
        self.shard = self.columns = None

    def startup_index(self):
        """The corpus's index, built on first use; None if it cannot be kept."""
        if self.index is None:
            raw = self.raw
            try:
                self.index = StartupIndex.open(
                    raw.directory, raw.shards
                ) or StartupIndex.build(raw.directory, raw.shards, raw.record_metadata)
            except OSError as error:
                logger.warning(
                    "Start-up index cannot be saved ({}); start-up reads the full "
                    "record metadata on every pass instead",
                    error,
                )
                self.index = False
        return self.index or None

    def position(self, index):
        """(shard, first record of that shard) for a record of the corpus."""
        ends = self.raw.ends
        if not 0 <= index < len(self.raw):
            raise IndexError(index)
        shard = bisect_right(ends, index)
        return shard, ends[shard - 1] if shard else 0

    def light(self, index, owner):
        startup = self.startup_index()
        if startup is None:
            return owner[index]
        shard, start = self.position(index)
        return self.record(startup, shard, start, index - start, owner)

    def record(self, startup, shard, start, local, owner):
        if self.shard != shard:
            self.columns = startup.columns(start, self.raw.ends[shard])
            self.shard = shard
        record = LightRecord()
        for name, values in self.columns:
            value = values[local]
            if value is not ABSENT:
                record[name] = value
        record.owner, record.index, record.indexed = owner, start + local, startup.known
        # As in a whole record: the saved tensor names become a batch reference.
        keys = record.pop("_shard_batch_keys", None)
        if keys is not None:
            record["batch"] = ShardBatch(self.raw, shard, local, tuple(keys))
        return record

    def scan(self, owner):
        startup = self.startup_index()
        if startup is None:
            yield from owner
            return
        start = 0
        for shard, stop in enumerate(self.raw.ends):
            for local in range(stop - start):
                yield self.record(startup, shard, start, local, owner)
            start = stop

    def fetch(self, indices):
        raw = self.raw
        if not indices:
            return []
        positions = []
        for index in indices:
            shard, start = self.position(index)
            positions.append((shard, index - start))
        startup = self.startup_index()
        metadata = path = None
        if startup is not None:
            selection = [
                startup.key,
                hashlib.sha256(array("Q", indices).tobytes()).hexdigest(),
            ]
            path = startup.directory / f"selected_{selection[1][:32]}.json"
            metadata = _saved_selection(path, selection)
        if metadata is None:
            logger.info(
                "Reading {:,} selected records whole from the record metadata of "
                "their shards{}",
                len(indices),
                "; later starts reuse them" if path is not None else "",
            )
            # In shard order, so that each shard's metadata is read only once.
            found = {
                index: raw.record_metadata(shard)[local]
                for index, (shard, local) in sorted(zip(indices, positions))
            }
            metadata = [found[index] for index in indices]
            if path is not None:
                content = json.dumps(metadata).encode("utf-8")
                header = [*selection, hashlib.sha256(content).hexdigest()]
                content = json.dumps(header).encode("utf-8") + b"\n" + content
                try:
                    atomic_write(path, lambda stream: stream.write(content))
                except OSError as error:
                    logger.warning("Selected records were not saved: {}", error)
        return [
            _whole(raw, shard, local, record)
            for (shard, local), record in zip(positions, metadata)
        ]


def _corpus(records):
    """Start-up access to `records` when they are a sharded corpus, else None."""
    raw = records.records if isinstance(records, EncodedRecords) else records
    if not isinstance(raw, ShardedRecords):
        return None
    corpus = getattr(raw, "startup_corpus", None)
    if corpus is None:
        corpus = raw.startup_corpus = _Corpus(raw)
    return corpus


def scan(records):
    """Every record in order, for a start-up pass that reads indexed fields.

    A sharded corpus yields light records, read from its index; any other
    sequence yields its own records. Light records are for reading fields, not
    for keeping: use `fetch` for the records a pass selects.
    """
    corpus = _corpus(records)
    return iter(records) if corpus is None else corpus.scan(records)


def light(record):
    """`record` for a start-up pass.

    A handle into a sharded corpus becomes a light record, read from the index
    without touching its shard; any other handle is read whole, and a plain
    record is returned as it is.
    """
    if isinstance(record, RecordRef):
        corpus = _corpus(record.owner)
        if corpus is not None:
            return corpus.light(record.index, record.owner)
    whole = getattr(record, "materialize", None)
    return record if whole is None else whole()


def fetch(records, indices):
    """Whole records at `indices`, in that order.

    Start-up needs the selected validation and test records whole. In a sharded
    corpus they are spread over every shard, so reading them is a pass over all
    record metadata; the first read is therefore saved beside the index.
    """
    indices = list(indices)
    corpus = _corpus(records)
    if corpus is None:
        return [records[index] for index in indices]
    whole = corpus.fetch(indices)
    if isinstance(records, EncodedRecords):
        return [records.factory(record, records.encoder) for record in whole]
    return whole
