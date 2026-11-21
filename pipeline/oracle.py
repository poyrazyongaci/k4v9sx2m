"""Reference pipeline for rules.md: the per-row and per-person logic of rules.py,
run out of core in three hash-partitioned stages so 60M events fit in memory.

  stage 1  per byte range of each input file: parse lines, pre-dedup, spill by hash(id)
  stage 2  per id bucket: dedup (events: smallest ingest_ts), validate, map to person,
           spill by hash(person)
  stage 3  per person bucket: sessions, user_daily part, campaign_daily partial sums

Usage: python oracle.py [DATA_DIR OUTPUT_DIR]   (env DATA_DIR, OUTPUT_DIR otherwise)
Env: SCRATCH_DIR (spill parent), ORACLE_WORKERS (default 3)."""
import json
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import tempfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules as R  # noqa: E402

CHUNK = int(os.environ.get("ORACLE_CHUNK", 48 << 20))           # bytes of CSV per stage 1 task
BUCKET_BYTES = int(os.environ.get("ORACLE_BUCKET_BYTES", 32 << 20))    # bytes of event CSV per bucket
G = {}                      # read-only state copied to every worker


def bucket(s, k):
    return zlib.crc32(s.encode("utf-8")) % k


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_all(folder):
    if not folder.is_dir():
        return
    for path in sorted(folder.iterdir()):
        with open(path, "rb") as fh:
            yield pickle.load(fh)


def ranges(path):
    """Split a file into byte ranges that each start at a line start."""
    size = path.stat().st_size
    points = [0]
    with open(path, "rb") as fh:
        target = CHUNK
        while target < size:
            fh.seek(target)
            pos = target
            while True:
                block = fh.read(1 << 16)
                if not block:
                    pos = size
                    break
                i = block.find(b"\n")
                if i >= 0:
                    pos += i + 1
                    break
                pos += len(block)
            if pos >= size:
                break
            points.append(pos)
            target = pos + CHUNK
    points.append(size)
    return [(points[i], points[i + 1]) for i in range(len(points) - 1) if points[i] < points[i + 1]]


def stage1(task):
    kind, path, start, end, tid = task
    k, scratch = G["k"], G["scratch"]
    record = R.event_record if kind == "events" else R.click_record
    best = [dict() for _ in range(k)]
    for line in R.range_lines(path, start, end):
        rec = record(line)
        if rec:
            R.keep_smallest(best[bucket(rec[0], k)], rec)  # dedup is a min, so pre-dedup is safe
    for b, d in enumerate(best):
        if d:
            dump(d, scratch / f"{kind}1" / str(b) / f"{tid:06d}.pkl")


def stage2(task):
    kind, b = task
    k, scratch, names, ws, we = G["k"], G["scratch"], G["names"], G["ws"], G["we"]
    best = {}
    for d in load_all(scratch / f"{kind}1" / str(b)):
        for rid, (key, f) in d.items():
            R.keep_smallest(best, (rid, key, f))
    out = [[] for _ in range(k)]
    for key, f in best.values():
        if kind == "events":
            v = R.valid_event(key, f, ws, we)
        else:
            v = R.valid_click(f, ws, we, G["channels"])
        if v:
            p = R.person_of(names, v[0])
            out[bucket(p, k)].append((p,) + v[1:])
    for pb, rows in enumerate(out):
        if rows:
            dump(rows, scratch / f"{kind}2" / str(pb) / f"{b:06d}.pkl")


def stage3(pb):
    scratch = G["scratch"]
    evs, clicks = {}, {}
    for rows in load_all(scratch / "events2" / str(pb)):
        for r in rows:
            evs.setdefault(r[0], []).append(r[1:])
    for rows in load_all(scratch / "clicks2" / str(pb)):
        for r in rows:
            clicks.setdefault(r[0], []).append(r[1:])
    user_rows, camp_rows = {}, {}
    for p, pevs in evs.items():
        R.process_person(p, pevs, clicks.get(p, []), G["products"], user_rows, camp_rows)
    if user_rows:
        R.write_tables(G["out"], user_rows, None, G["channels"], part=pb)
    return camp_rows


def init_worker(state):
    G.update(state)


def run_pool(fn, tasks, workers):
    """spawn, not fork: forked workers hang on macOS. Each worker gets a copy of G."""
    if workers <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    # A dead worker raises BrokenProcessPool instead of hanging.
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=init_worker,
                             initargs=(dict(G),)) as pool:
        return list(pool.map(fn, tasks))


def main():
    t0 = time.time()
    data_dir = Path(sys.argv[-2] if len(sys.argv) >= 3 else os.environ["DATA_DIR"])
    out_dir = Path(sys.argv[-1] if len(sys.argv) >= 3 else os.environ["OUTPUT_DIR"])
    workers = int(os.environ.get("ORACLE_WORKERS", "3"))
    scratch_parent = os.environ.get("SCRATCH_DIR") or None
    if scratch_parent:
        Path(scratch_parent).mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="oracle-", dir=scratch_parent))

    def log(msg):
        print(f"[oracle {time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)

    try:
        manifest = json.loads((data_dir / "manifest.json").read_text())
        G["ws"], G["we"] = R.parse_f1(manifest["window_start"]), R.parse_f1(manifest["window_end"])
        G["products"], G["channels"] = R.load_dims(data_dir)
        event_files = sorted((data_dir / "events").glob("*.csv"))
        click_files = sorted((data_dir / "clicks").glob("*.csv"))
        event_bytes = sum(p.stat().st_size for p in event_files)
        G["k"] = max(1, event_bytes // BUCKET_BYTES)
        G["scratch"], G["out"] = scratch, out_dir
        (out_dir / "user_daily").mkdir(parents=True, exist_ok=True)
        (out_dir / "campaign_daily").mkdir(parents=True, exist_ok=True)

        tasks = []
        for kind, files in (("events", event_files), ("clicks", click_files)):
            for path in files:
                for start, end in ranges(path):
                    tasks.append((kind, path, start, end, len(tasks)))
        log(f"stage 1: {len(tasks)} ranges, {G['k']} buckets, {workers} workers")
        run_pool(stage1, tasks, workers)

        persons = R.Persons()
        for path in sorted((data_dir / "links").glob("*.csv")):
            persons.add_lines(R.data_lines(path))
        G["names"] = persons.names()
        del persons
        log(f"links: {len(G['names'])} linked ids")

        log("stage 2")
        run_pool(stage2, [(kind, b) for kind in ("events", "clicks") for b in range(G["k"])], workers)
        G["names"] = None

        log("stage 3")
        camp_rows = {}
        for part in run_pool(stage3, list(range(G["k"])), workers):
            for key, vals in part.items():
                acc = camp_rows.setdefault(key, [0, 0, 0, 0])
                for i in range(4):
                    acc[i] += vals[i]
        R.write_tables(out_dir, None, camp_rows, G["channels"], part=0)
        if not any((out_dir / "user_daily").glob("*.parquet")):
            R.write_tables(out_dir, {}, None, G["channels"], part=0)  # empty table, right schema
        log("done")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
