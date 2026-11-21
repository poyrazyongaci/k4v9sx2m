//! Differential tests: the kernel (asm on x86, the twin elsewhere) and the
//! twin against the old loop, entry for entry.

use super::*;
use std::collections::HashMap;

fn rng(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    mix(*state)
}

/// 2026-10-01 to 2026-10-15, as the rules' example.
const WS: i64 = 1_790_812_800_000;
const WE: i64 = WS + 14 * DAY_MS;

fn lookups() -> Lookups {
    let cats: Vec<Vec<u8>> = (0..20).map(|i| format!("cat{i:02}").into_bytes()).chain([b"unknown".to_vec()]).collect();
    let mut product_dense = vec![u32::MAX; 2000];
    for (n, slot) in product_dense.iter_mut().enumerate().skip(1) {
        if n % 7 != 3 {
            *slot = (n % 20) as u32;
        }
    }
    let mut product_odd = HashMap::new();
    product_odd.insert(b"P12".to_vec(), 4);
    product_odd.insert(b"p012".to_vec(), 5);
    product_odd.insert(b"x".to_vec(), 6);
    Lookups {
        window_start: WS,
        window_end: WE,
        cats,
        unknown_cat: 20,
        product_dense,
        product_odd,
        camp_ids: vec![b"c_1".to_vec()],
        camp_channels: vec![b"email".to_vec()],
        campaigns: {
            let mut h = HashMap::with_hasher(IdHash);
            h.insert(b"c_1".to_vec(), 0);
            h
        },
    }
}

type Out = Vec<Vec<(u64, u64, i64, u64)>>;

/// Every entry each loop writes for the files, one after the other, by
/// partition, each file driven as `run_chunk` drives `consume_fused`.
/// One parser for all of them: a scatter holds its memory to the end.
fn run(files: &[&[u8]], which: u8, lk: &Lookups) -> Out {
    let fast = Fast::new(lk, 0).unwrap();
    let mut st = ParseState {
        fast: &fast,
        cache: DateCache::new(),
        entries: Scatter::new(4, CHUNK_BYTES),
        shift1: 62,
        clicks: Vec::new(),
        pairs: Vec::new(),
        emit: None,
        dense: None,
    };
    let mut scan = Scan::new(WINDOW);
    for text in files {
      let mut buf = text.to_vec();
      buf.extend_from_slice(&[0u8; 4096]);
      let len = text.len();
      let m = Mapped { ptr: buf.as_ptr(), len };
      let mut pos = 0;
      unsafe {
        while pos < len {
            pos = match which {
                1 => st.events_hot(m, pos, len, &mut scan),
                2 => st.events_twin(m, pos, len, &mut scan),
                _ => st.events::<0, _>(Fused { m, pos, end: len, scan: &mut scan }),
            };
            if pos < len {
                let end = text[pos..].iter().position(|&c| c == b'\n').map_or(len, |i| pos + i);
                st.long_line(&text[pos..end], Kind::Events);
                pos = end + 1;
            }
        }
      }
    }
    let lists = st.entries.finish();
    lists
        .iter()
        .map(|l| {
            let mut v = Vec::new();
            unsafe { l.for_each_chunk(|c| v.extend(c.iter().map(|e| (e.ekey, e.ukey, e.ingest, e.w3)))) };
            v
        })
        .collect()
}

/// How many lines of `text` the kernel takes without leaving the loop
/// for the old one (purchases and full buffers count as taken).
fn accepted(text: &[u8]) -> (usize, usize) {
    let lk = lookups();
    let fast = Fast::new(&lk, 0).unwrap();
    let mut entries: Scatter<Entry> = Scatter::new(4, CHUNK_BYTES);
    let hot = entries.hot();
    let mut buf = text.to_vec();
    buf.extend_from_slice(&[0u8; 4096]);
    let mut c = Ctx::new(&fast, &hot, 62, unsafe { buf.as_ptr().add(text.len()) });
    c.load_cache(&DateCache::new());
    // Months of the test window in the cache copy, as the old loop would
    // have them after its first lines.
    let mut cache = DateCache::new();
    unsafe {
        crate::ts::parse_ts(b"2026-10-03T00:00:00Z".as_ptr(), 20, &mut cache);
        let mut pad = b"2026-10-15T00:00:00Z".to_vec();
        pad.extend_from_slice(&[0; 32]);
        crate::ts::parse_ts(pad.as_ptr(), 20, &mut cache);
    }
    c.load_cache(&cache);
    let (mut pos, mut yes, mut all) = (0, 0, 0);
    while pos < text.len() {
        let code = unsafe { line_twin(&mut c, buf.as_ptr().add(pos)) };
        all += 1;
        yes += usize::from(code != SLOW);
        pos = text[pos..].iter().position(|&c| c == b'\n').map_or(text.len(), |i| pos + i + 1);
    }
    let _ = entries.finish();
    (yes, all)
}

fn same(text: &[u8], lk: &Lookups, what: &str) -> usize {
    same_all(&[text], lk, what)
}

/// Both loops over the files give the same entries; returns how many.
fn same_all(files: &[&[u8]], lk: &Lookups, what: &str) -> usize {
    let a = run(files, 0, lk);
    let n: usize = a.iter().map(Vec::len).sum();
    for which in [1, 2] {
    let b = run(files, which, lk);
    if a != b {
        let what = format!("{what} (loop {which})");
        // Find the first file and line that differ, for the message.
        for text in files {
            if run(&[text], 0, lk) == run(&[text], which, lk) {
                continue;
            }
            let mut at = 0;
            while at < text.len() {
                let end = text[at..].iter().position(|&c| c == b'\n').map_or(text.len(), |i| at + i + 1);
                let line = &text[at..end];
                if run(&[line], 0, lk) != run(&[line], which, lk) {
                    panic!("{what}: differs on {:?}", String::from_utf8_lossy(line));
                }
                at = end;
            }
            panic!("{what}: differs on file {:?}", String::from_utf8_lossy(text));
        }
        panic!("{what}: differs, no single file shows it");
    }
    }
    n
}

fn pick<'x>(state: &mut u64, xs: &[&'x str]) -> &'x str {
    xs[(rng(state) % xs.len() as u64) as usize]
}

fn id(state: &mut u64, first: u8) -> Vec<u8> {
    let len = match rng(state) % 20 {
        0 => 0,
        1 => 10 + rng(state) % 3,
        2 => 1 + rng(state) % 4,
        _ => 5 + rng(state) % 5,
    };
    let mut v = vec![first];
    for _ in 1..len {
        v.push(b"abcdefghijklmnopqrstuvwxyz0123456789"[(rng(state) % 36) as usize]);
    }
    if len == 0 {
        v.clear();
    }
    if rng(state) % 500 == 0 {
        v.push(*pick(state, &["\u{e9}", "\0", "\x7f", " "]).as_bytes().last().unwrap());
    }
    v
}

fn ts(state: &mut u64, at: i64) -> String {
    let (y, mo, d) = civil_from_days(at.div_euclid(DAY_MS));
    let sec = at.rem_euclid(DAY_MS) / 1000;
    let (h, mi, s) = (sec / 3600, sec / 60 % 60, sec % 60);
    match rng(state) % 100 {
        0..=57 => format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z"),
        58..=67 => {
            let off = (rng(state) % 1381) as i64 - 600;
            let local = at / 1000 * 1000 + off * 60_000;
            let (y, mo, d) = civil_from_days(local.div_euclid(DAY_MS));
            let sec = local.rem_euclid(DAY_MS) / 1000;
            let sign = if off < 0 { '-' } else { '+' };
            let off = off.abs();
            format!("{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}{sign}{:02}:{:02}", sec / 3600, sec / 60 % 60, sec % 60, off / 60, off % 60)
        }
        68..=96 => format!("{at}"),
        97 => format!("{:015}", at),
        98 => pick(state, &["", "2026-10-03T23:58:10.000Z", "2026-02-30T00:00:00Z", "2026-10-03 23:58:10Z", "1.79e12", "-5", "2026-10-03T23:58:10+24:00", "2026-10-03t23:58:10Z", "86400000", "2026-10-03T23:58:10+0200"]).to_string(),
        _ => {
            let mut v = format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z").into_bytes();
            let i = (rng(state) % v.len() as u64) as usize;
            v[i] = b"0123456789:-TZ+ x"[(rng(state) % 17) as usize];
            String::from_utf8(v).unwrap()
        }
    }
}

/// A line shaped like the generated ones, with their dirt.
fn line(state: &mut u64) -> Vec<u8> {
    let span = WE - WS + 4 * DAY_MS;
    let at = WS - 2 * DAY_MS + (rng(state) % span as u64) as i64;
    let ingest = at + match rng(state) % 50 {
        0 => LATE_MS,
        1 => LATE_MS + 1000,
        2 => LATE_MS + (rng(state) % (5 * DAY_MS as u64)) as i64,
        3 => -((rng(state) % 100_000) as i64),
        _ => (rng(state) % 600_000) as i64,
    };
    let (y, mo, d) = civil_from_days(ingest.div_euclid(DAY_MS));
    let sec = ingest.rem_euclid(DAY_MS) / 1000;
    let ing = format!("{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}Z", sec / 3600, sec / 60 % 60, sec % 60);
    let ty = pick(state, &["view", "view", "view", "cart", "cart", "purchase", "Purchase", "cart ", "VIEW", "", "buy", "purchas", "purchasex"]);
    let prod = match rng(state) % 40 {
        0 => String::new(),
        1 => pick(state, &["P12", "p012", "x", "p0", "p99999999", "p1 "]).to_string(),
        _ => format!("p{}", rng(state) % 2100),
    };
    let price = match rng(state) % 12 {
        0 => String::new(),
        1 => pick(state, &["0", "0250", "10000000", "10000001", "19.99", "-5", "+5", "1e3", "1000\r", "00000000000000000250"]).to_string(),
        _ => format!("{}", 100 + rng(state) % 50_000),
    };
    let fields: [Vec<u8>; 7] = [
        id(state, b'e'),
        id(state, b'u'),
        ts(state, at).into_bytes(),
        ing.into_bytes(),
        ty.as_bytes().to_vec(),
        prod.into_bytes(),
        price.into_bytes(),
    ];
    let mut out = Vec::new();
    for (k, f) in fields.iter().enumerate() {
        if k > 0 {
            out.push(b',');
        }
        let q = rng(state) % 5 == 0;
        if q {
            out.push(b'"');
        }
        out.extend_from_slice(f);
        if q {
            out.push(b'"');
        }
    }
    out
}

/// Damage of every kind the rules name, on a line.
fn damage(state: &mut u64, l: &mut Vec<u8>) {
    let subs: &[u8] = b",\"\n\r\0\x80\xff :-TZ0123456789ep";
    for _ in 0..1 + rng(state) % 3 {
        if l.is_empty() {
            l.push(b'"');
            continue;
        }
        let at = (rng(state) % l.len() as u64) as usize;
        let c = subs[(rng(state) % subs.len() as u64) as usize];
        match rng(state) % 3 {
            0 => {
                l.remove(at);
            }
            1 => l.insert(at, c),
            _ => l[at] = c,
        }
    }
}

#[test]
fn random_files_match_old_loop() {
    let lk = lookups();
    let mut state = 1;
    let (mut lines, mut entries, mut fast, mut seen) = (0, 0, 0, 0);
    let mut batch = Vec::new();
    for round in 0..6000 {
        let mut text = Vec::new();
        let n = 1 + rng(&mut state) % 1000;
        let dirt = rng(&mut state) % 4;
        for _ in 0..n {
            let mut l = line(&mut state);
            if dirt > 0 && rng(&mut state) % (8 * dirt) == 0 {
                damage(&mut state, &mut l);
            }
            text.extend_from_slice(&l);
            if rng(&mut state) % 300 == 0 {
                text.push(b'\r');
            }
            text.push(b'\n');
        }
        if round % 3 == 0 {
            text.pop();
        }
        if round % 50 == 0 {
            let (y, a) = accepted(&text);
            fast += y;
            seen += a;
        }
        lines += n as usize;
        batch.push(text);
        if batch.len() == 100 {
            let files: Vec<&[u8]> = batch.iter().map(Vec::as_slice).collect();
            entries += same_all(&files, &lk, &format!("rounds to {round}"));
            batch.clear();
        }
    }
    eprintln!("{lines} lines, {entries} entries; fast path took {fast} of {seen} sampled");
    assert!(fast * 2 > seen);
}

/// Every quote placement on every field of a plain line, one and two
/// fields at a time, and every field empty.
#[test]
fn quote_placements_and_empty_fields() {
    let lk = lookups();
    let base: [&[u8]; 7] = [b"e1234567a", b"u12345678", b"2026-10-03T23:58:10Z", b"2026-10-03T23:59:10Z", b"purchase", b"p123", b"100"];
    let variants = |x: &[u8]| -> Vec<Vec<u8>> {
        let mut v = vec![x.to_vec(), Vec::new()];
        let q = |a: &[u8], b: &[u8], c: &[u8]| [a, b, c].concat();
        v.push(q(b"\"", x, b"\""));
        v.push(q(b"\"", b"", b"\""));
        v.push(q(b"\"", b"", b""));
        v.push(q(b"\"", x, b""));
        v.push(q(b"", x, b"\""));
        v.push(q(b"\"\"", x, b""));
        v.push(q(b"", x, b"\"\""));
        v.push(q(b"\"\"", x, b"\"\""));
        v.push(q(b"\"", b"\"", b"\""));
        v.push(q(b"\"\"\"", b"", b""));
        v.push(q(b"\"\"\"\"", b"", b""));
        v.push(q(&x[..1], b"\"", &x[1..]));
        v.push(q(b"\"", &x[..1], &[b"\"", &x[1..], b"\""].concat()));
        v
    };
    let mut text = Vec::new();
    for k in 0..7 {
        for v in variants(base[k]) {
            for j in 0..7 {
                for u in variants(base[j]) {
                    let mut f: Vec<Vec<u8>> = base.iter().map(|x| x.to_vec()).collect();
                    f[j] = u.clone();
                    f[k] = v.clone();
                    text.extend_from_slice(&f.join(&b","[..]));
                    text.push(b'\n');
                }
            }
        }
    }
    // Every subset of fields empty, then quoted empty.
    for mask in 0..128u32 {
        for quoted in [false, true] {
            let f: Vec<Vec<u8>> = (0..7)
                .map(|k| if mask >> k & 1 == 1 { if quoted { b"\"\"".to_vec() } else { Vec::new() } } else { base[k].to_vec() })
                .collect();
            text.extend_from_slice(&f.join(&b","[..]));
            text.push(b'\n');
        }
    }
    same(&text, &lk, "quotes");
}

/// Lines of 20 to 300 bytes, grown or cut in each field in turn, plain
/// and quoted: every comma and line break on each side of the 32, 64
/// and 96 byte edges.
#[test]
fn line_lengths() {
    let lk = lookups();
    let base: [&[u8]; 7] = [b"e1234567a", b"u12345678", b"1791071890000", b"2026-10-03T23:59:10Z", b"view", b"p123", b"100"];
    let mut text = Vec::new();
    for k in 0..7 {
        for quoted in [false, true] {
            for total in 20..=300usize {
                let mut f: Vec<Vec<u8>> = base.iter().map(|x| x.to_vec()).collect();
                let now = f.join(&b","[..]).len() + 2 * usize::from(quoted);
                if total < now {
                    let cut = (now - total).min(f[k].len());
                    f[k].truncate(base[k].len() - cut);
                } else {
                    let filler = if k == 2 { b'0' } else { b'7' };
                    f[k].splice(0..0, std::iter::repeat_n(filler, total - now));
                }
                if quoted {
                    f[k] = [&b"\""[..], &f[k], b"\""].concat();
                }
                text.extend_from_slice(&f.join(&b","[..]));
                text.push(b'\n');
            }
        }
    }
    let (y, a) = accepted(&text);
    eprintln!("lengths: fast path took {y} of {a}");
    same(&text, &lk, "lengths");
}

/// Timestamps of each form at and around the edges that matter: the
/// window ends, the late limit, month and year ends, leap days.
#[test]
fn timestamp_edges() {
    let lk = lookups();
    let mut text = Vec::new();
    let mut state = 9;
    let points = [WS - 1, WS, WS + 1, WE - 1, WE, WE + 1, WS + 30 * DAY_MS, 0, 86_400_000, days_from_civil(2028, 2, 29) * DAY_MS, days_from_civil(2026, 12, 31) * DAY_MS + DAY_MS - 1000];
    for &at in &points {
        for late in [0, LATE_MS, LATE_MS + 1, LATE_MS + 1000, -1000, -LATE_MS] {
            for _ in 0..30 {
                let t = ts(&mut state, at);
                let ing = at + late;
                let (y, mo, d) = civil_from_days(ing.div_euclid(DAY_MS));
                let sec = ing.rem_euclid(DAY_MS) / 1000;
                let ing = format!("{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}Z", sec / 3600, sec / 60 % 60, sec % 60);
                for ty in ["view", "purchase"] {
                    text.extend_from_slice(format!("e1,u1,{t},{ing},{ty},p5,250\n").as_bytes());
                    text.extend_from_slice(format!("\"e2\",\"u2\",\"{t}\",\"{ing}\",\"{ty}\",\"p5\",\"250\"\n").as_bytes());
                }
            }
        }
    }
    for t in ["0001791071890000", "1791071890000", "179107189000", "17910718900000", "2026-10-03T23:58:10-00:00", "2026-10-03T23:58:10+23:59", "2026-10-03T23:58:10-23:59", "2026-10-03T23:58:10+24:00", "2026-10-03T23:58:10+05:60", "2026-10-03T23:58:10\r", "2026-10-32T00:00:00Z", "2026-11-31T00:00:00Z", "2026-10-03T24:00:00Z", "2026-10-03T23:60:00Z", "2026-10-03T23:59:60Z"] {
        for ing in ["2026-10-04T00:00:00Z", "2026-10-03T23:58:10Z", "x", "", "2026-10-03T23:58:10+00:00"] {
            text.extend_from_slice(format!("e3,u3,{t},{ing},cart,p5,\n").as_bytes());
        }
    }
    same(&text, &lk, "timestamps");
}

/// Byte noise, files with no line break at the end, CRLF files, empty
/// lines, and lines cut at the end of the mapping.
#[test]
fn noise_and_file_ends() {
    let lk = lookups();
    let mut state = 4;
    let alphabet: &[u8] = b",,,,\"\"\n\r\0\x80\xffeup0123456789-:TZ viewcart";
    let mut batch = Vec::new();
    for round in 0..20_000 {
        let len = (rng(&mut state) % 400) as usize;
        let mut text: Vec<u8> = (0..len).map(|_| alphabet[(rng(&mut state) % alphabet.len() as u64) as usize]).collect();
        if round % 2 == 0 {
            // Real lines with noise between them.
            let mut good = line(&mut state);
            good.push(b'\n');
            let at = (rng(&mut state) % (text.len() as u64 + 1)) as usize;
            text.splice(at..at, good);
        }
        batch.push(text);
        if batch.len() == 1000 {
            let files: Vec<&[u8]> = batch.iter().map(Vec::as_slice).collect();
            same_all(&files, &lk, &format!("noise to {round}"));
            batch.clear();
        }
    }
    let mut good = Vec::new();
    for _ in 0..50 {
        good.extend_from_slice(&line(&mut state));
        good.extend_from_slice(b"\r\n");
    }
    same(&good, &lk, "crlf");
    let cuts: Vec<&[u8]> = (0..good.len().min(400)).map(|cut| &good[..good.len() - cut]).collect();
    same_all(&cuts, &lk, "cut");
    let edges: [&[u8]; 7] = [b"", b"\n", b"\n\n", b",,,,,,", b",,,,,,\n", b"\"\",\"\",\"\",\"\",\"\",\"\",\"\"", b"e1,u1,1791071890000,2026-10-03T23:58:10Z,view,p1,"];
    same_all(&edges, &lk, "edge");
}

/// The share of lines the line splitter takes, over the event files of
/// `HOT_DATA` (a dataset folder). Run by hand with `--ignored`.
#[test]
#[ignore]
fn fast_path_share() {
    let Some(dir) = std::env::var_os("HOT_DATA") else { return };
    let (mut yes, mut all) = (0, 0);
    for e in std::fs::read_dir(std::path::Path::new(&dir).join("events")).unwrap() {
        let text = std::fs::read(e.unwrap().path()).unwrap();
        let start = text.iter().position(|&c| c == b'\n').map_or(text.len(), |i| i + 1);
        let (y, a) = accepted(&text[start..]);
        yes += y;
        all += a;
    }
    eprintln!("fast path share: {yes} of {all} lines ({:.3}%)", 100.0 * yes as f64 / all.max(1) as f64);
}

/// The second window at its limits: the third comma at bytes 20 to 63 and
/// the line break at bytes 80 to 97 (95 is the last the kernel can take,
/// and `a2 + 63` the last of the window), fields 3 to 6 quoted in turn, a
/// lone or stray quote just before the line break, all three `ts` forms,
/// and each line followed by commas and quotes so the masks see past it.
#[test]
fn second_window_edges() {
    let lk = lookups();
    let mut text = Vec::new();
    let mut at95 = Vec::new();
    let tails: [&[u8]; 3] = [b"", b"\"", b"x\""];
    for a2 in 20..=63usize {
        for nl in 80..=97usize {
            for quoted in 0..7usize {
                for tail in tails {
                    for ts in [&b"1791071890000"[..], b"2026-10-03T23:58:10Z", b"2026-10-03T19:58:10-04:00"] {
                        // Fields 0 and 1 sized so that field 2 ends at a2.
                        let rest = a2 as isize - ts.len() as isize - 2;
                        if !(2..=18).contains(&rest) {
                            continue;
                        }
                        let id_len = (rest as usize).div_ceil(2);
                        let user_len = rest as usize - id_len;
                        let mut f: Vec<Vec<u8>> = vec![
                            std::iter::once(b'e').chain(std::iter::repeat_n(b'1', id_len - 1)).collect(),
                            std::iter::once(b'u').chain(std::iter::repeat_n(b'2', user_len - 1)).collect(),
                            ts.to_vec(),
                            b"2026-10-04T00:00:00Z".to_vec(),
                            if quoted % 2 == 0 { b"view".to_vec() } else { b"purchase".to_vec() },
                            b"p5".to_vec(),
                            b"1".to_vec(),
                        ];
                        if quoted >= 3 {
                            f[quoted] = [&b"\""[..], &f[quoted], b"\""].concat();
                        }
                        let mut line = f.join(&b","[..]);
                        // Field 6 padded with digits so the line break is at
                        // `nl`, then the tail.
                        let want = nl.saturating_sub(tail.len());
                        if line.len() > want {
                            continue;
                        }
                        let pad = want - line.len();
                        let at = if quoted == 6 { line.len() - 1 } else { line.len() };
                        line.splice(at..at, std::iter::repeat_n(b'1', pad));
                        line.extend_from_slice(tail);
                        if nl == 95 {
                            at95.extend_from_slice(&line);
                            at95.push(b'\n');
                        }
                        text.extend_from_slice(&line);
                        text.push(b'\n');
                        text.extend_from_slice(b",\",\",\",,,\"\n");
                    }
                }
            }
        }
    }
    let (y, a) = accepted(&text);
    let (y95, a95) = accepted(&at95);
    eprintln!("second window: kernel took {y} of {a}, {y95} of {a95} with the line break at byte 95");
    assert!(y95 > 0);
    same(&text, &lk, "second window");
}
