"""Literal helpers for rules.md, shared by naive.py and oracle.py.
Every function here handles one row or one person. No vectorization."""
import bisect
import re
from pathlib import Path

DAY = 86_400_000
GAP = 1_800_000          # a gap over this starts a new session
REACH = 7 * DAY          # click reach before a session start
LATE = DAY               # ingest_ts - ts over this is late
MAX_PRICE = 10_000_000
DIGITS = re.compile(r"[0-9]+")   # [0-9] is a literal ASCII range, never Unicode digits
F1 = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})Z")
F2 = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})"
                r"([+-])([0-9]{2}):([0-9]{2})")
EVENT_TYPES = ("view", "cart", "purchase")


# ---------- CSV dialect ----------

def range_lines(path, start, end):
    """Data lines whose first byte is in [start, end). start is 0 or right after a \\n.
    Only \\n ends a line. The first line of the file (start 0) is the header.
    A final line without \\n is still a line. A zero-byte file has no lines."""
    with open(path, "rb") as fh:
        fh.seek(start)
        data = fh.read(end - start)
    if not data:
        return []
    lines = data.split(b"\n")
    if lines[-1] == b"":  # nothing after a final \n: not a line
        lines.pop()
    if start == 0:
        lines = lines[1:]
    return [line.decode("utf-8") for line in lines]


def data_lines(path):
    return range_lines(path, 0, Path(path).stat().st_size)


def fields(line, n):
    """Split on every comma. Strip quotes that wrap a whole field. None if the line
    has the wrong field count or a quote anywhere else. Empty values become None."""
    parts = line.split(",")
    if len(parts) != n:
        return None
    out = []
    for p in parts:
        if len(p) >= 2 and p[0] == '"' and p[-1] == '"':
            p = p[1:-1]
        if '"' in p:
            return None
        out.append(p if p != "" else None)
    return out


# ---------- timestamps ----------

def leap(y):
    return (y % 4 == 0 and y % 100 != 0) or y % 400 == 0


def days_in_month(y, m):
    return [31, 29 if leap(y) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]


def days_from_civil(y, m, d):
    """Days since 1970-01-01 in the proleptic Gregorian calendar (y >= 1)."""
    y -= m <= 2
    era = y // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def civil_ms(y, mo, d, h, mi, s):
    """None for an impossible calendar value. Year 0000 is treated as impossible;
    it is outside every window anyway."""
    if y < 1 or not 1 <= mo <= 12 or not 1 <= d <= days_in_month(y, mo):
        return None
    if h > 23 or mi > 59 or s > 59:
        return None
    return (days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + s) * 1000


def parse_ts(s):
    """Epoch ms for F1, F2 or F3, else None. The window check is separate."""
    if s is None:
        return None
    m = F1.fullmatch(s)
    if m:
        return civil_ms(*map(int, m.groups()))
    m = F2.fullmatch(s)
    if m:
        g = m.groups()
        local = civil_ms(*map(int, g[:6]))
        oh, om = int(g[7]), int(g[8])
        if local is None or oh > 23 or om > 59:
            return None
        offset = (oh * 3600 + om * 60) * 1000
        return local - offset if g[6] == "+" else local + offset
    if DIGITS.fullmatch(s):
        return int(s)  # Python ints are unbounded: a huge value is just outside the window
    return None


def f1_or_none(s):
    m = F1.fullmatch(s or "")
    return civil_ms(*map(int, m.groups())) if m else None


def parse_f1(s):
    m = F1.fullmatch(s or "")
    v = civil_ms(*map(int, m.groups())) if m else None
    if v is None:
        raise SystemExit(f"ingest_ts or manifest time is not valid F1: {s!r}")
    return v


# ---------- dimension files ----------

def load_dims(data_dir):
    products, channels = {}, {}
    for target, name in ((products, "products.csv"), (channels, "campaigns.csv")):
        for line in data_lines(Path(data_dir) / name):
            f = fields(line, 2)
            if f and f[0] is not None:
                target[f[0]] = f[1]
    return products, channels


# ---------- linked IDs ----------

class Persons:
    """Union-find over IDs. The root of each set is its byte-wise smallest ID
    (Python str order is code point order, which equals UTF-8 byte order)."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression, iterative
            self.parent[x], x = root, self.parent[x]
        return root

    def link(self, a, b):
        for x in (a, b):
            self.parent.setdefault(x, x)
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def add_lines(self, lines):
        for line in lines:
            f = fields(line, 2)
            if f is None or f[0] is None or f[1] is None:
                continue
            self.link(f[0], f[1])

    def names(self):
        """Flat dict: every linked ID -> its person's name."""
        return {x: self.find(x) for x in self.parent}


def person_of(names, uid):
    return names.get(uid, uid)


# ---------- events and clicks ----------

def event_record(line):
    """(event_id, dedup_key, fields) for a parsed event line, or None."""
    f = fields(line, 7)
    if f is None or f[0] is None:
        return None
    ingest = f1_or_none(f[3])
    if ingest is None:
        # rules.md: "ingest_ts is always valid F1". A line that breaks this (a data
        # row equal to the header, say) is treated as malformed. Open question.
        return None
    # Copies with equal ingest_ts are identical per the spec. The rest of the key
    # only makes the choice deterministic if they are not.
    return f[0], (ingest, tuple(v or "" for v in f[1:])), f


def click_record(line):
    f = fields(line, 4)
    if f is None or f[0] is None:
        return None
    return f[0], tuple(v or "" for v in f[1:]), f


def keep_smallest(best, rec):
    rid, key, f = rec
    old = best.get(rid)
    if old is None or key < old[0]:
        best[rid] = (key, f)


def valid_event(key, f, ws, we):
    """(user_id, ts, event_type, product_id, price) for a kept copy that passes
    validation and is not late, else None. price is 0 for view and cart."""
    _, user, ts_raw, _, etype, product, price_raw = f
    ts = parse_ts(ts_raw)
    if user is None or ts is None or not ws <= ts < we:
        return None
    if etype not in EVENT_TYPES or product is None:
        return None
    price = 0
    if etype == "purchase":
        if price_raw is None or not DIGITS.fullmatch(price_raw):
            return None
        price = int(price_raw)
        if not 1 <= price <= MAX_PRICE:
            return None
    if key[0] - ts > LATE:
        return None
    return user, ts, etype, product, price


def valid_click(f, ws, we, channels):
    """(user_id, ts, click_id, campaign_id) or None."""
    cid, user, camp, ts_raw = f
    ts = parse_ts(ts_raw)
    if user is None or ts is None or not ws - REACH <= ts < we or camp not in channels:
        return None
    return user, ts, cid, camp


# ---------- one person ----------

def process_person(person, evs, pclicks, products, user_rows, camp_rows):
    """evs: [(ts, event_type, product_id, price)], pclicks: [(ts, click_id, campaign)].
    Adds to user_rows[(person, day)] = [sessions, events, revenue, {category: cents}]
    and camp_rows[(campaign, day)] = [sessions, purchasing, revenue, assisted]."""
    evs = sorted(evs, key=lambda e: e[0])
    pclicks = sorted(pclicks)
    sessions = []
    prev = None
    for e in evs:
        if prev is None or e[0] - prev > GAP:  # equal ts: gap 0, same session
            sessions.append([])
        sessions[-1].append(e)
        prev = e[0]
    times = [c[0] for c in pclicks]
    for sess in sessions:
        start = sess[0][0]
        day = start // DAY
        purchases = [e for e in sess if e[1] == "purchase"]
        revenue = sum(e[3] for e in purchases)
        u = user_rows.setdefault((person, day), [0, 0, 0, {}])
        u[0] += 1
        u[1] += len(sess)
        u[2] += revenue
        for e in purchases:
            cat = products.get(e[2], "unknown")
            u[3][cat] = u[3].get(cat, 0) + e[3]
        # Clicks with start - 7 days <= ts <= start.
        cands = pclicks[bisect.bisect_left(times, start - REACH):bisect.bisect_right(times, start)]
        if cands:
            credit = min(cands, key=lambda c: (-c[0], c[1]))[2]  # latest ts, then smallest click_id
        else:
            credit = "organic"
        r = camp_rows.setdefault((credit, day), [0, 0, 0, 0])
        r[0] += 1
        r[1] += 1 if purchases else 0
        r[2] += revenue
        for camp in {c[2] for c in cands}:
            camp_rows.setdefault((camp, day), [0, 0, 0, 0])[3] += 1


def top_category(cats):
    if not cats:
        return None
    return min(cats.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def write_tables(out_dir, user_rows, camp_rows, channels, part=0):
    """Writes user_daily unless user_rows is None, campaign_daily unless camp_rows is None."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    out_dir = Path(out_dir)
    tables = {}
    if user_rows is not None:
        ud = {"user_id": [], "day": [], "sessions": [], "events": [], "revenue_cents": [], "top_category": []}
        for (p, day), (s, n, rev, cats) in user_rows.items():
            for k, v in zip(ud, (p, day, s, n, rev, top_category(cats))):
                ud[k].append(v)
        tables["user_daily"] = ud
    if camp_rows is not None:
        cd = {k: [] for k in ("campaign_id", "channel", "day", "sessions", "purchasing_sessions",
                              "revenue_cents", "assisted_sessions")}
        for (camp, day), (s, ps, rev, a) in camp_rows.items():
            ch = "organic" if camp == "organic" else channels[camp]
            for k, v in zip(cd, (camp, ch, day, s, ps, rev, a)):
                cd[k].append(v)
        tables["campaign_daily"] = cd
    strings = ("user_id", "top_category", "campaign_id", "channel")
    for name, cols in tables.items():
        (out_dir / name).mkdir(parents=True, exist_ok=True)
        arrays = {k: pa.array(v, type=pa.date32() if k == "day" else pa.string() if k in strings
                              else pa.int64()) for k, v in cols.items()}
        pq.write_table(pa.table(arrays), out_dir / name / f"part-{part:05d}.parquet")
