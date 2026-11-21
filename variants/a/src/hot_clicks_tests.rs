//! Differential tests: the click kernel (asm on x86, the twin elsewhere)
//! and the twin against the old click loop, click for click, with the
//! block layout.

use super::*;
use std::collections::HashMap;

fn rng(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    mix(*state)
}

/// 2026-10-01 to 2026-10-15; the click window starts 7 days earlier.
const WS: i64 = 1_790_812_800_000;
const WE: i64 = WS + 14 * DAY_MS;

/// Campaigns `c_1` to `c_1999` except every 13th, and odd ids that only
/// the exact map knows.
fn lookups() -> Lookups {
    let mut camp_ids: Vec<Vec<u8>> = (1..2000).filter(|n| n % 13 != 0).map(|n| format!("c_{n}").into_bytes()).collect();
    for odd in ["C_7", "c_0012", "camp", "c_", "c_12345678", "c_1 ", " c_1", "\u{e9}c_1"] {
        camp_ids.push(odd.as_bytes().to_vec());
    }
    let mut campaigns = HashMap::with_hasher(IdHash);
    for (i, id) in camp_ids.iter().enumerate() {
        campaigns.insert(id.clone(), i as u32);
    }
    Lookups {
        window_start: WS,
        window_end: WE,
        cats: vec![b"a".to_vec(), b"unknown".to_vec()],
        unknown_cat: 1,
        product_dense: vec![u32::MAX; 10],
        product_odd: HashMap::new(),
        camp_channels: camp_ids.iter().map(|_| b"email".to_vec()).collect(),
        camp_ids,
        campaigns,
    }
}

type Out = (Vec<usize>, Vec<(u64, u64, u64)>);

/// Every click each loop appends for the files, one after the other, and
/// the block lengths, each file driven as `run_chunk` drives
/// `consume_fused`, with `long_line` for what no loop takes.
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
                    1 => st.clicks_hot(m, pos, len, &mut scan),
                    2 => st.clicks_twin(m, pos, len, &mut scan),
                    _ => st.clicks::<0, _>(Fused { m, pos, end: len, scan: &mut scan }),
                };
                if pos < len {
                    let end = text[pos..].iter().position(|&c| c == b'\n').map_or(len, |i| pos + i);
                    st.long_line(&text[pos..end], Kind::Clicks);
                    pos = end + 1;
                }
            }
        }
    }
    let _ = st.entries.finish();
    let lens = st.clicks.iter().map(Vec::len).collect();
    let all = st.clicks.iter().flatten().map(|c| (c.key, c.ckey, c.tc)).collect();
    (lens, all)
}

/// How many lines of `text` the kernel decides without the old loop.
fn accepted(text: &[u8], lk: &Lookups) -> (usize, usize) {
    let fast = Fast::new(lk, 0).unwrap();
    let mut entries: Scatter<Entry> = Scatter::new(4, CHUNK_BYTES);
    let hot = entries.hot();
    let mut buf = text.to_vec();
    buf.extend_from_slice(&[0u8; 4096]);
    let mut c = Ctx::new(&fast, &hot, 62, unsafe { buf.as_ptr().add(text.len()) });
    let mut cache = DateCache::new();
    for month in [&b"2026-09-24T00:00:00Z"[..], b"2026-10-15T00:00:00Z"] {
        let mut pad = month.to_vec();
        pad.extend_from_slice(&[0; 32]);
        unsafe { crate::ts::parse_ts(pad.as_ptr(), 20, &mut cache) };
    }
    c.load_cache(&cache);
    let mut sink = vec![Click { key: 0, ckey: 0, tc: 0 }; 1];
    let (mut pos, mut yes, mut all) = (0, 0, 0);
    while pos < text.len() {
        (c.at, c.astop) = (sink.as_mut_ptr(), sink.as_mut_ptr());
        let code = unsafe { cline_twin(&mut c, buf.as_ptr().add(pos)) };
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

/// The three loops over the files give the same clicks and blocks;
/// returns how many clicks.
fn same_all(files: &[&[u8]], lk: &Lookups, what: &str) -> usize {
    let a = run(files, 0, lk);
    for which in [1, 2] {
        if run(files, which, lk) == a {
            continue;
        }
        for text in files {
            if run(&[text], 0, lk) == run(&[text], which, lk) {
                continue;
            }
            let mut at = 0;
            while at < text.len() {
                let end = text[at..].iter().position(|&c| c == b'\n').map_or(text.len(), |i| at + i + 1);
                let line = &text[at..end];
                if run(&[line], 0, lk) != run(&[line], which, lk) {
                    panic!("{what} (loop {which}): differs on {:?}", String::from_utf8_lossy(line));
                }
                at = end;
            }
            panic!("{what} (loop {which}): differs on file {:?}", String::from_utf8_lossy(text));
        }
        panic!("{what} (loop {which}): differs, no single file shows it");
    }
    a.1.len()
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
        v.push(*pick(state, &["\u{e9}", "\0", "\x7f", " ", "\t"]).as_bytes().last().unwrap());
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

fn campaign(state: &mut u64) -> String {
    match rng(state) % 30 {
        0 => pick(state, &["", "c_", "c", "C_7", "c_0", "c_0012", "c_012", "camp", "c_12345678", "c_1 ", " c_1", "c_1x", "c_-1", "c__1", "x_1", "c_9999999", "c_2000", "c_2600"]).to_string(),
        1 => format!("c_{}", 13 * (1 + rng(state) % 150)),
        2 => format!("c_{}", 2000 + rng(state) % 100_000),
        _ => format!("c_{}", 1 + rng(state) % 1999),
    }
}

/// A click line shaped like the generated ones, with their dirt.
fn line(state: &mut u64) -> Vec<u8> {
    let span = WE - WS + 10 * DAY_MS;
    let at = WS - 8 * DAY_MS + (rng(state) % span as u64) as i64;
    let fields: [Vec<u8>; 4] = [id(state, b'k'), id(state, b'u'), campaign(state).into_bytes(), ts(state, at).into_bytes()];
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

fn damage(state: &mut u64, l: &mut Vec<u8>) {
    let subs: &[u8] = b",\"\n\r\0\x80\xff :-TZ0123456789c_";
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
fn random_click_files_match_old_loop() {
    let lk = lookups();
    let mut state = 21;
    let (mut lines, mut clicks, mut fast, mut seen) = (0, 0, 0, 0);
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
            let (y, a) = accepted(&text, &lk);
            fast += y;
            seen += a;
        }
        lines += n as usize;
        batch.push(text);
        if batch.len() == 100 {
            let files: Vec<&[u8]> = batch.iter().map(Vec::as_slice).collect();
            clicks += same_all(&files, &lk, &format!("rounds to {round}"));
            batch.clear();
        }
    }
    eprintln!("{lines} click lines, {clicks} clicks; kernel took {fast} of {seen} sampled");
    assert!(fast * 2 > seen);
}

/// Every quote placement on every field, one and two fields at a time,
/// every field empty, plain and quoted.
#[test]
fn click_quotes_and_empty_fields() {
    let lk = lookups();
    let base: [&[u8]; 4] = [b"k1234567a", b"u12345678", b"c_123", b"2026-10-03T23:58:10Z"];
    let variants = |x: &[u8]| -> Vec<Vec<u8>> {
        let q = |a: &[u8], b: &[u8], c: &[u8]| [a, b, c].concat();
        vec![
            x.to_vec(),
            Vec::new(),
            q(b"\"", x, b"\""),
            q(b"\"", b"", b"\""),
            q(b"\"", b"", b""),
            q(b"\"", x, b""),
            q(b"", x, b"\""),
            q(b"\"\"", x, b""),
            q(b"", x, b"\"\""),
            q(b"\"\"", x, b"\"\""),
            q(b"\"", b"\"", b"\""),
            q(b"\"\"\"", b"", b""),
            q(&x[..1], b"\"", &x[1..]),
            q(b"\"", &x[..1], &[b"\"", &x[1..], b"\""].concat()),
        ]
    };
    let mut text = Vec::new();
    for k in 0..4 {
        for v in variants(base[k]) {
            for j in 0..4 {
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
    for mask in 0..16u32 {
        for quoted in [false, true] {
            let f: Vec<Vec<u8>> = (0..4)
                .map(|k| if mask >> k & 1 == 1 { if quoted { b"\"\"".to_vec() } else { Vec::new() } } else { base[k].to_vec() })
                .collect();
            text.extend_from_slice(&f.join(&b","[..]));
            text.push(b'\n');
        }
    }
    same(&text, &lk, "click quotes");
}

/// Lines of 10 to 200 bytes, each field grown or cut in turn, plain and
/// quoted: the line break on each side of bytes 32 and 64.
#[test]
fn click_line_lengths() {
    let lk = lookups();
    let base: [&[u8]; 4] = [b"k1234567a", b"u12345678", b"c_123", b"1791071890000"];
    let mut text = Vec::new();
    for k in 0..4 {
        for quoted in [false, true] {
            for total in 10..=200usize {
                let mut f: Vec<Vec<u8>> = base.iter().map(|x| x.to_vec()).collect();
                let now = f.join(&b","[..]).len() + 2 * usize::from(quoted);
                if total < now {
                    let cut = (now - total).min(f[k].len());
                    f[k].truncate(base[k].len() - cut);
                } else {
                    let filler = if k == 3 { b'0' } else { b'7' };
                    let at = if k == 2 { 2 } else { 0 };
                    f[k].splice(at..at, std::iter::repeat_n(filler, total - now));
                }
                if quoted {
                    f[k] = [&b"\""[..], &f[k], b"\""].concat();
                }
                text.extend_from_slice(&f.join(&b","[..]));
                text.push(b'\n');
            }
        }
    }
    let (y, a) = accepted(&text, &lk);
    eprintln!("click lengths: kernel took {y} of {a}");
    same(&text, &lk, "click lengths");
}

/// Timestamps of each form at the click window's ends, and every campaign
/// form, crossed.
#[test]
fn click_timestamps_and_campaigns() {
    let lk = lookups();
    let cws = WS - 7 * DAY_MS;
    let mut state = 5;
    let mut text = Vec::new();
    let camps = ["c_1", "c_13", "c_1999", "c_2000", "c_9999999", "c_12345678", "c_0", "c_01", "C_7", "c_0012", "camp", "c_", "c", "", "c_1 ", " c_1", "c_1x", "\u{e9}c_1"];
    for at in [cws - 1, cws, cws + 1, WS, WE - 1, WE, WE + 1, 0, days_from_civil(2026, 9, 30) * DAY_MS + DAY_MS - 1000] {
        for camp in camps {
            for _ in 0..12 {
                let t = ts(&mut state, at);
                text.extend_from_slice(format!("k1,u1,{camp},{t}\n").as_bytes());
                text.extend_from_slice(format!("\"k2\",\"u2\",\"{camp}\",\"{t}\"\n").as_bytes());
            }
        }
    }
    for t in ["0001791071890000", "1791071890000", "179107189000", "2026-10-03T23:58:10-00:00", "2026-10-03T23:58:10+23:59", "2026-10-03T23:58:10+24:00", "2026-10-03T23:58:10+05:60", "2026-10-03T23:58:10\r", "2026-10-32T00:00:00Z", "2026-09-31T00:00:00Z"] {
        text.extend_from_slice(format!("k3,u3,c_5,{t}\n").as_bytes());
    }
    same(&text, &lk, "click timestamps");
}

/// Byte noise, files with no line break at the end, CRLF, empty lines,
/// and files cut at every length near the end.
#[test]
fn click_noise_and_file_ends() {
    let lk = lookups();
    let mut state = 8;
    let alphabet: &[u8] = b",,,\"\"\n\r\0\x80\xffkuc_0123456789-:TZ ";
    let mut batch = Vec::new();
    for round in 0..20_000 {
        let len = (rng(&mut state) % 300) as usize;
        let mut text: Vec<u8> = (0..len).map(|_| alphabet[(rng(&mut state) % alphabet.len() as u64) as usize]).collect();
        if round % 2 == 0 {
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
    let mut plain = Vec::new();
    for _ in 0..30 {
        plain.extend_from_slice(&line(&mut state));
        plain.push(b'\n');
    }
    let cuts: Vec<&[u8]> = (0..plain.len().min(300)).map(|cut| &plain[..plain.len() - cut]).collect();
    same_all(&cuts, &lk, "cut");
    let edges: [&[u8]; 6] = [b"", b"\n", b"\n\n", b",,,", b",,,\n", b"k1,u1,c_1,1791071890000"];
    same_all(&edges, &lk, "edge");
}

/// More clicks than a block holds, so the kernel fills blocks and starts
/// new ones, with refused lines in between.
#[test]
fn click_blocks() {
    let lk = lookups();
    let mut text = Vec::new();
    for i in 0..(3 * CLICK_BLOCK) {
        if i % 997 == 0 {
            text.extend_from_slice(b"k12345678901,u1,c_5,1791071890000\n");
        }
        text.extend_from_slice(format!("k{i},u{},c_{},1791071890000\n", i % 50, 1 + i % 1000).as_bytes());
    }
    let n = same(&text, &lk, "blocks");
    assert!(n > 2 * CLICK_BLOCK);
}
