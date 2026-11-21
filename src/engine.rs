//! The fast engine. Three parallel phases, each a pool of threads pulling
//! jobs off an atomic counter:
//!
//! 1. parse: event and click chunks are scanned and parsed; every
//!    well-formed event copy becomes a 32-byte entry scattered by a hash of
//!    its event key. One job meanwhile reads all links and builds the
//!    user -> person map (union-find).
//! 2. resolve: per event partition, dedup on the exact event key keeping the
//!    smallest ingest, then scatter the survivors as 16-byte records by a
//!    hash of their RAW user. Clicks get the same scatter. No link lookup.
//! 3. person, in two passes over the same threads. The raw pass, per user
//!    partition: group by user, ONE link lookup per user. A user in no link
//!    is a whole person: sessionize, attribute clicks, aggregate, write
//!    user_daily rows. Every linked user is forwarded, as one run of words,
//!    to the partition of its person's name. The person pass, per partition
//!    of forwarded runs: group the runs by person and do the same.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::common::*;
use crate::ids::{self, cmp_ids};
use crate::output::write_camp_rows;
use crate::usercols::UserOut;
use crate::scan::{for_each_line, Scan};
use crate::sys::{self, prefetch, prof, List, Mapped, Scatter};
use crate::ts::{canonical_number, load32, load64, parse_price, parse_ts, parse_ts_ingest, DateCache, BAD};

pub struct Config {
    pub threads: usize,
    pub timing: bool,
    /// Collect user rows in memory instead of writing files (tests).
    pub collect: bool,
    pub chunk_bytes: usize,
    pub p1_bits: Option<u32>,
    pub p2_bits: Option<u32>,
}

impl Config {
    pub fn from_env() -> Self {
        let num = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u32>().ok());
        Config {
            threads: sys::threads(),
            timing: std::env::var_os("ENGINE_TIMING").is_some(),
            collect: false,
            chunk_bytes: num("ENGINE_CHUNK_KB").map_or(4 << 20, |kb| (kb as usize) << 10),
            p1_bits: num("ENGINE_P1_BITS"),
            p2_bits: num("ENGINE_P2_BITS"),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Entry {
    ekey: u64,
    ukey: u64,
    ingest: i64,
    /// `ts_rel << 32 | price << 8 | category`, or `DEAD` for a copy that is
    /// invalid or lost the dedup.
    w3: u64,
}

const DEAD: u64 = u64::MAX;

/// A valid click. `key` is the user key. `tc` is `ts_rel << 32 | campaign`, ts relative to the
/// start of the click window.
#[repr(C)]
#[derive(Clone, Copy)]
struct Click {
    key: u64,
    ckey: u64,
    tc: u64,
}

/// A surviving event on its way to the raw pass: `pkey` is the key of its
/// raw user, `v` the entry's `w3`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Ev {
    pkey: u64,
    v: u64,
}

const CHUNK_BYTES: usize = 16384;
const CLICK_BLOCK: usize = 16384;
const WINDOW: usize = 1 << 16;
const WINDOW_SLACK: usize = 4096;

#[inline(always)]
fn mix(k: u64) -> u64 {
    let x = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let x = (x ^ (x >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 32)
}

/// Lookup tables derived from the catalogs for the fast paths.
struct Fast<'a> {
    lk: &'a Lookups,
    ws: i64,
    wlen: u64,
    cws: i64,
    cwlen: u64,
    /// Category of product `p<n>`, by `n`; `unknown` where the file has none.
    prod_cat: Vec<u8>,
    unknown: u8,
    /// Campaign index of `c_<n>`, by `n`; `u32::MAX` where the file has none.
    camp_idx: Vec<u32>,
    ncamp: usize,
    ndays: usize,
    ws_day: i64,
    ws_in_day: u64,
}

impl<'a> Fast<'a> {
    /// `None` when the dataset is outside what the compact records can
    /// hold; the caller then runs the reference pipeline instead.
    fn new(lk: &'a Lookups, event_bytes: u64) -> Option<Self> {
        let wlen = lk.window_end.checked_sub(lk.window_start)?;
        if wlen <= 0 || wlen + CLICK_REACH_MS >= (u32::MAX as i64) {
            return None;
        }
        if lk.cats.len() > 255 || lk.camp_ids.len() >= (1 << 24) {
            return None;
        }
        if event_bytes >= (1 << 34) {
            return None;
        }
        // The late check adds 24 hours to a `ts` inside the window.
        lk.window_end.checked_add(LATE_MS)?;
        let ws_day = lk.window_start.div_euclid(DAY_MS);
        let ndays = ((lk.window_end - 1).div_euclid(DAY_MS) - ws_day + 1) as usize;
        if (lk.camp_ids.len() + 1) * ndays > (1 << 19) {
            return None;
        }
        let unknown = lk.unknown_cat as u8;
        let prod_cat: Vec<u8> = lk.product_dense.iter().map(|&c| if c == u32::MAX { unknown } else { c as u8 }).collect();
        let mut camp_idx = Vec::new();
        for (id, &idx) in &lk.campaigns {
            if let Some(n) = canonical_suffix(id, b"c_") {
                let n = n as usize;
                if camp_idx.len() <= n {
                    camp_idx.resize(n + 1, u32::MAX);
                }
                camp_idx[n] = idx;
            }
        }
        Some(Fast {
            lk,
            ws: lk.window_start,
            wlen: wlen as u64,
            cws: lk.window_start - CLICK_REACH_MS,
            cwlen: (wlen + CLICK_REACH_MS) as u64,
            prod_cat,
            unknown: lk.unknown_cat as u8,
            camp_idx,
            ncamp: lk.camp_ids.len(),
            ndays,
            ws_day,
            ws_in_day: lk.window_start.rem_euclid(DAY_MS) as u64,
        })
    }

    #[inline(always)]
    unsafe fn category(&self, p: *const u8, len: usize) -> u8 {
        if len >= 2 && *p == b'p' {
            if let Some(n) = canonical_number(p.add(1), len - 1) {
                return match self.prod_cat.get(n as usize) {
                    Some(&c) => c,
                    None => self.unknown,
                };
            }
        }
        self.category_slow(std::slice::from_raw_parts(p, len))
    }

    #[cold]
    #[inline(never)]
    fn category_slow(&self, id: &[u8]) -> u8 {
        self.lk.product_cat(id) as u8
    }

    #[inline(always)]
    unsafe fn campaign(&self, p: *const u8, len: usize) -> u32 {
        if len >= 3 && *p == b'c' && *p.add(1) == b'_' {
            if let Some(n) = canonical_number(p.add(2), len - 2) {
                return match self.camp_idx.get(n as usize) {
                    Some(&c) => c,
                    None => u32::MAX,
                };
            }
        }
        self.campaign_slow(std::slice::from_raw_parts(p, len))
    }

    #[cold]
    #[inline(never)]
    fn campaign_slow(&self, id: &[u8]) -> u32 {
        self.lk.campaigns.get(id).copied().unwrap_or(u32::MAX)
    }
}

// ---------------------------------------------------------------- links

/// user key -> person key (the byte-wise smallest ID of the person), for
/// every linked ID whose person is not named by that ID itself. A miss
/// means the ID is its own person. Open addressing, 16-byte slots, on huge
/// pages where the kernel gives them.
///
/// A small blocked Bloom filter (two bits in one word per key) holds every
/// linked ID: the ones in the table and the names of linked people, which
/// are not in it. A filter miss is exact and means "in no link at all". A
/// filter hit is always confirmed by the table; if the table has no entry,
/// the ID is taken for the name of a linked person, which is harmless when
/// the filter was wrong: such an ID only takes the longer road through the
/// person pass.
#[derive(Clone, Copy)]
pub struct LinkMap {
    slots: *mut [u64; 2],
    mask: usize,
    /// Index of a slot past the table that is always empty: where lookups
    /// the filter rejected are sent, so the probe needs no extra branch.
    nowhere: usize,
    filter: *mut u64,
    fmask: usize,
}
unsafe impl Send for LinkMap {}
unsafe impl Sync for LinkMap {}

impl LinkMap {
    /// A table of `cap` slots and a filter sized for `keys` keys.
    fn with_slots(cap: usize, keys: usize) -> LinkMap {
        // Fresh anonymous memory is zero, and 0 is the empty key. One extra
        // slot at the end stays empty for ever (`nowhere`).
        let words = (keys / 8).next_power_of_two().clamp(64, 1 << 21);
        // The filter is small; put it on a 2 MiB boundary so one huge page
        // can back it.
        const HUGE: usize = 2 << 20;
        let raw = sys::anon(words * 8 + HUGE) as usize;
        let filter = ((raw + HUGE - 1) & !(HUGE - 1)) as *mut u64;
        LinkMap { slots: sys::anon((cap + 1) * 16) as *mut [u64; 2], mask: cap - 1, nowhere: cap, filter, fmask: words - 1 }
    }
    #[inline(always)]
    fn home_of(&self, h: u64) -> usize {
        (h >> 17) as usize & self.mask
    }
    #[inline(always)]
    fn home(&self, key: u64) -> usize {
        self.home_of(mix(key))
    }
    #[inline(always)]
    fn filter_word(&self, h: u64) -> *mut u64 {
        unsafe { self.filter.add((h >> 43) as usize & self.fmask) }
    }
    #[inline(always)]
    fn filter_bits(h: u64) -> u64 {
        (1u64 << (h & 63)) | (1u64 << ((h >> 6) & 63))
    }
    /// Where the lookup of a key with hash `h` starts: its home slot if
    /// the filter says the key may be present, else the empty slot.
    #[inline(always)]
    fn start(&self, h: u64) -> usize {
        let bits = Self::filter_bits(h);
        let maybe = unsafe { *self.filter_word(h) } & bits == bits;
        // A coin toss: this must be a conditional move, never a branch.
        std::hint::select_unpredictable(maybe, self.home_of(h), self.nowhere)
    }
    /// Prefetch for the build, which probes before the filter exists.
    #[inline(always)]
    fn prefetch_home(&self, key: u64) {
        prefetch(unsafe { self.slots.add(self.home(key)) });
    }
    /// A lookup in three stages, so that a block of them can overlap their
    /// cache misses: `stage1` hashes the key and asks for its filter word,
    /// `stage2` reads the filter and asks for the table slot (or for the
    /// empty slot), `stage3` reads the slot and returns the person.
    #[inline(always)]
    fn stage1(&self, key: u64) -> u64 {
        let h = mix(key);
        prefetch(self.filter_word(h));
        h
    }
    #[inline(always)]
    fn stage2(&self, h: u64) -> usize {
        let i = self.start(h);
        prefetch(unsafe { self.slots.add(i) });
        i
    }
    #[inline(always)]
    fn stage3(&self, i: usize, key: u64) -> u64 {
        let s = unsafe { *self.slots.add(i) };
        // Hit or empty slot is another coin toss; a slot holding some other
        // key is rare and goes out of line.
        let found = std::hint::select_unpredictable(s[0] == key, s[1], key);
        if (s[0] != key) & (s[0] != 0) {
            return self.person_far(key, i);
        }
        found
    }
    /// The rest of the probe sequence after a first slot held another key.
    #[inline(never)]
    fn person_far(&self, key: u64, first: usize) -> u64 {
        let mut i = (first + 1) & self.mask;
        loop {
            let s = unsafe { *self.slots.add(i) };
            if s[0] == key {
                return s[1];
            }
            if s[0] == 0 {
                return key;
            }
            i = (i + 1) & self.mask;
        }
    }

    fn build(pairs: &[(u64, u64)]) -> LinkMap {
        // Pass 1: a dense node id per distinct ID, and union-find over them.
        let cap = (pairs.len() * 4).next_power_of_two().max(16);
        let ids = LinkMap::with_slots(cap, 0);
        let mut keys: Vec<u64> = Vec::with_capacity(pairs.len() * 2 + 16);
        let mut parent: Vec<u32> = Vec::with_capacity(pairs.len() * 2 + 16);
        fn find(parent: &mut [u32], mut x: u32) -> u32 {
            loop {
                let p = parent[x as usize];
                if p == x {
                    return x;
                }
                let g = parent[p as usize];
                parent[x as usize] = g;
                x = g;
            }
        }
        for block in pairs.chunks(16) {
            for &(a, b) in block {
                ids.prefetch_home(a);
                ids.prefetch_home(b);
            }
            for &(a, b) in block {
                let mut node = [0u32; 2];
                for (side, key) in [a, b].into_iter().enumerate() {
                    let mut i = ids.home(key);
                    node[side] = loop {
                        let s = unsafe { &mut *ids.slots.add(i) };
                        if s[0] == key {
                            break s[1] as u32;
                        }
                        if s[0] == 0 {
                            let id = keys.len() as u32;
                            *s = [key, u64::from(id)];
                            keys.push(key);
                            parent.push(id);
                            break id;
                        }
                        i = (i + 1) & ids.mask;
                    };
                }
                let ra = find(&mut parent, node[0]);
                let rb = find(&mut parent, node[1]);
                if ra != rb {
                    // The root is always the smallest ID of its set, so the
                    // root's key is the person's name.
                    if cmp_ids(keys[ra as usize], keys[rb as usize]).is_lt() {
                        parent[rb as usize] = ra;
                    } else {
                        parent[ra as usize] = rb;
                    }
                }
            }
        }
        // Pass 2: the final map holds only IDs that are not their own name.
        let mut renamed: Vec<[u64; 2]> = Vec::with_capacity(keys.len());
        let mut names: Vec<u64> = Vec::new();
        for id in 0..keys.len() as u32 {
            let root = find(&mut parent, id);
            if root != id {
                renamed.push([keys[id as usize], keys[root as usize]]);
            } else {
                names.push(keys[id as usize]);
            }
        }
        let map = LinkMap::with_slots((renamed.len() * 2).next_power_of_two().max(16), keys.len() * 2);
        // The names of linked people are in the filter but not in the
        // table, so a filter miss means "in no link at all": that ID is a
        // whole person by itself.
        for &key in &names {
            let h = mix(key);
            unsafe { *map.filter_word(h) |= Self::filter_bits(h) };
        }
        for block in renamed.chunks(16) {
            for pair in block {
                map.prefetch_home(pair[0]);
            }
            for pair in block {
                let h = mix(pair[0]);
                let mut i = map.home_of(h);
                unsafe {
                    while (*map.slots.add(i))[0] != 0 {
                        i = (i + 1) & map.mask;
                    }
                    *map.slots.add(i) = *pair;
                    *map.filter_word(h) |= Self::filter_bits(h);
                }
            }
        }
        map
    }
}

// ---------------------------------------------------------------- phase 1

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Events,
    Clicks,
    Links,
}

struct Job {
    kind: Kind,
    file: usize,
    start: usize,
    end: usize,
}

struct ParseState<'a> {
    fast: &'a Fast<'a>,
    cache: DateCache,
    entries: Scatter<Entry>,
    shift1: u32,
    clicks: Vec<Vec<Click>>,
    pairs: Vec<(u64, u64)>,
}

const VIEW: u32 = u32::from_le_bytes(*b"view");
const CART: u32 = u32::from_le_bytes(*b"cart");
const PURCHASE: u64 = u64::from_le_bytes(*b"purchase");

impl<'a> ParseState<'a> {
    #[inline(never)]
    unsafe fn events(&mut self, base: *const u8, scan: &Scan, limit: u32) -> u32 {
        let fast = self.fast;
        let (ws, wlen) = (fast.ws, fast.wlen);
        let shift1 = self.shift1;
        let cache = &mut self.cache;
        let entries = &mut self.entries;
        let known = scan.clean;
        for_each_line::<7>(scan, limit, |s, e| {
            let id_len = (e[0] - s[0]) as usize;
            if id_len == 0 {
                return;
            }
            let ekey = ids::pack_in(base.add(s[0] as usize), id_len, known);
            let user_len = (e[1] - s[1]) as usize;
            let ukey = ids::pack_in(base.add(s[1] as usize), user_len, known);
            let (ts, ingest) = parse_ts_ingest(
                base.add(s[2] as usize),
                (e[2] - s[2]) as usize,
                base.add(s[3] as usize),
                (e[3] - s[3]) as usize,
                cache,
            );
            let type_len = e[4] - s[4];
            let tp = base.add(s[4] as usize);
            let word = load32(tp);
            let ts_rel = ts.wrapping_sub(ws) as u64;
            // Late is `ingest - ts > 24 h`, which is `ingest > ts + 24 h`
            // whenever that sum does not overflow; with `ts` in the window
            // (the only case where the answer is used) it cannot, see
            // `Fast::new`. An ingest_ts that is no timestamp is `i64::MAX`
            // here, later than any `ts + 24 h`: the copy sorts last and is
            // never valid, as in the reference.
            let fresh = ingest <= ts.wrapping_add(LATE_MS);
            // Nearly every row passes these four rules and is a view or a
            // cart: one well-predicted branch for the rules, one for the
            // length of the type.
            let sound = (user_len != 0) & (e[5] != s[5]) & (ts_rel < wlen) & fresh;
            let mut w3 = DEAD;
            if !sound {
                std::hint::cold_path();
            } else if type_len == 4 {
                // View or cart is a coin toss, so it selects, it does not
                // branch.
                let plain = (word == VIEW) | (word == CART);
                w3 = std::hint::select_unpredictable(plain, ts_rel << 32, DEAD);
            } else if type_len == 8 && load64(tp) == PURCHASE {
                let price = parse_price(base.add(s[6] as usize), (e[6] - s[6]) as usize);
                if price != 0 {
                    let cat = fast.category(base.add(s[5] as usize), (e[5] - s[5]) as usize);
                    w3 = (ts_rel << 32) | (price << 8) | u64::from(cat);
                }
            }
            let part = (mix(ekey) >> shift1) as usize;
            entries.push(part, Entry { ekey, ukey, ingest, w3 });
        })
    }

    #[inline(never)]
    unsafe fn clicks(&mut self, base: *const u8, scan: &Scan, limit: u32) -> u32 {
        let fast = self.fast;
        let (cws, cwlen) = (fast.cws, fast.cwlen);
        let cache = &mut self.cache;
        let clicks = &mut self.clicks;
        let known = scan.clean;
        for_each_line::<4>(scan, limit, |s, e| {
            let id_len = (e[0] - s[0]) as usize;
            let user_len = (e[1] - s[1]) as usize;
            if id_len == 0 || user_len == 0 {
                return;
            }
            let ts = parse_ts(base.add(s[3] as usize), (e[3] - s[3]) as usize, cache);
            let rel = ts.wrapping_sub(cws) as u64;
            if rel >= cwlen {
                return;
            }
            let camp = fast.campaign(base.add(s[2] as usize), (e[2] - s[2]) as usize);
            if camp == u32::MAX {
                return;
            }
            let ckey = ids::pack_in(base.add(s[0] as usize), id_len, known);
            let ukey = ids::pack_in(base.add(s[1] as usize), user_len, known);
            if clicks.last().is_none_or(|b| b.len() == CLICK_BLOCK) {
                clicks.push(Vec::with_capacity(CLICK_BLOCK));
            }
            clicks.last_mut().unwrap_unchecked().push(Click { key: ukey, ckey, tc: (rel << 32) | u64::from(camp) });
        })
    }

    #[inline(never)]
    unsafe fn links(&mut self, base: *const u8, scan: &Scan, limit: u32) -> u32 {
        let pairs = &mut self.pairs;
        let known = scan.clean;
        for_each_line::<2>(scan, limit, |s, e| {
            let (la, lb) = ((e[0] - s[0]) as usize, (e[1] - s[1]) as usize);
            if la == 0 || lb == 0 {
                return;
            }
            pairs.push((ids::pack_in(base.add(s[0] as usize), la, known), ids::pack_in(base.add(s[1] as usize), lb, known)));
        })
    }

    /// One line that no scan window holds whole: longer than a window, or
    /// the last line of a file with no line break after it. Split and
    /// parsed with the reference readers, on slices, so neither the line
    /// nor a field has a length limit. It must decide exactly what the
    /// three loops above decide.
    #[cold]
    #[inline(never)]
    unsafe fn long_line(&mut self, line: &[u8], kind: Kind) {
        let fast = self.fast;
        match kind {
            Kind::Events => {
                let Some(f) = split_long::<7>(line) else {
                    return;
                };
                if f[0].is_empty() {
                    return;
                }
                let ekey = ids::pack_slice(f[0]);
                let ukey = ids::pack_slice(f[1]);
                let ingest = parse_ts_naive(f[3]);
                let ts = parse_ts_naive(f[2]).unwrap_or(BAD);
                let purchase = f[4] == b"purchase";
                let typed = purchase || f[4] == b"view" || f[4] == b"cart";
                let ts_rel = ts.wrapping_sub(fast.ws) as u64;
                let mut valid = typed
                    && !f[1].is_empty()
                    && !f[5].is_empty()
                    && ts_rel < fast.wlen
                    && ingest.is_some_and(|at| at.saturating_sub(ts) <= LATE_MS);
                let mut pc = 0u64;
                if purchase && valid {
                    match parse_price_naive(f[6]) {
                        Some(price) => pc = (price << 8) | u64::from(fast.lk.product_cat(f[5]) as u8),
                        None => valid = false,
                    }
                }
                let w3 = if valid { (ts_rel << 32) | pc } else { DEAD };
                let part = (mix(ekey) >> self.shift1) as usize;
                self.entries.push(part, Entry { ekey, ukey, ingest: ingest.unwrap_or(i64::MAX), w3 });
            }
            Kind::Clicks => {
                let Some(f) = split_long::<4>(line) else {
                    return;
                };
                if f[0].is_empty() || f[1].is_empty() {
                    return;
                }
                let ts = parse_ts_naive(f[3]).unwrap_or(BAD);
                let rel = ts.wrapping_sub(fast.cws) as u64;
                if rel >= fast.cwlen {
                    return;
                }
                let camp = fast.campaign_slow(f[2]);
                if camp == u32::MAX {
                    return;
                }
                let click = Click { key: ids::pack_slice(f[1]), ckey: ids::pack_slice(f[0]), tc: (rel << 32) | u64::from(camp) };
                if self.clicks.last().is_none_or(|b| b.len() == CLICK_BLOCK) {
                    self.clicks.push(Vec::with_capacity(CLICK_BLOCK));
                }
                self.clicks.last_mut().unwrap_unchecked().push(click);
            }
            Kind::Links => {
                let Some(f) = split_long::<2>(line) else {
                    return;
                };
                if f[0].is_empty() || f[1].is_empty() {
                    return;
                }
                self.pairs.push((ids::pack_slice(f[0]), ids::pack_slice(f[1])));
            }
        }
    }

    #[inline(always)]
    unsafe fn consume(&mut self, kind: Kind, base: *const u8, scan: &Scan, limit: u32) -> u32 {
        match kind {
            Kind::Events => self.events(base, scan, limit),
            Kind::Clicks => self.clicks(base, scan, limit),
            Kind::Links => self.links(base, scan, limit),
        }
    }
}

struct Parser<'a> {
    st: ParseState<'a>,
    scan: Scan,
}

unsafe fn find_newline(p: *const u8, len: usize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let hit = libc::memchr(p as *const libc::c_void, i32::from(b'\n'), len);
    if hit.is_null() {
        None
    } else {
        Some(hit as usize - p as usize)
    }
}

/// Stage 1 as a function of its own. With one caller the compiler would
/// inline the scan loop into `run_chunk`, where it measures 3% slower.
#[inline(never)]
unsafe fn scan_window(scan: &mut Scan, base: *const u8, len: usize) {
    scan.scan(base, len);
}

fn find_byte(hay: &[u8], byte: u8) -> Option<usize> {
    if hay.is_empty() {
        return None;
    }
    let hit = unsafe { libc::memchr(hay.as_ptr() as *const libc::c_void, i32::from(byte), hay.len()) };
    if hit.is_null() {
        None
    } else {
        Some(hit as usize - hay.as_ptr() as usize)
    }
}

/// The `N` fields of a line, quotes stripped, or `None` when the line is
/// malformed: what `split_line` answers, for a line of any length. It gives
/// up at the first comma too many, so a long run of commas costs nothing.
fn split_long<const N: usize>(line: &[u8]) -> Option<[&[u8]; N]> {
    let mut out: [&[u8]; N] = [&[]; N];
    let mut rest = line;
    for (i, slot) in out.iter_mut().enumerate() {
        let field = match find_byte(rest, b',') {
            Some(at) if i + 1 < N => {
                let field = &rest[..at];
                rest = &rest[at + 1..];
                field
            }
            None if i + 1 == N => rest,
            _ => return None,
        };
        *slot = match find_byte(field, b'"') {
            None => field,
            // The only legal quotes are one pair around the whole field.
            Some(0) if field.len() >= 2 && field[field.len() - 1] == b'"' => {
                let inner = &field[1..field.len() - 1];
                if find_byte(inner, b'"').is_some() {
                    return None;
                }
                inner
            }
            Some(_) => return None,
        };
    }
    Some(out)
}

impl<'a> Parser<'a> {
    /// Parse the lines of `m` that start in `[a, b)`. The header line
    /// (the one starting at 0) is skipped.
    unsafe fn run_chunk(&mut self, m: Mapped, a: usize, b: usize, kind: Kind) {
        let len = m.len;
        let from = a.saturating_sub(1);
        // A line starts in `[a, b)` exactly when a line break sits in
        // `[a - 1, b - 1)`, so the search stops at `b`: a chunk in the
        // middle of a very long line must not read on to that line's end.
        let Some(first) = find_newline(m.ptr.add(from), b.min(len) - from) else {
            return;
        };
        let mut pos = from + first + 1;
        while pos < b {
            let wlen = WINDOW.min(len - pos).min(b - pos + WINDOW_SLACK);
            let base = m.ptr.add(pos);
            let t = prof::start();
            scan_window(&mut self.scan, base, wlen);
            let t = prof::add(0, t);
            let limit = (b - pos).min(u32::MAX as usize) as u32;
            let used = self.st.consume(kind, base, &self.scan, limit);
            prof::add(1, t);
            if used > 0 {
                pos += used as usize;
                continue;
            }
            // No complete line in the window: a very long line, or the last
            // line of a file that does not end in a newline.
            let after = pos + wlen;
            let end = find_newline(m.ptr.add(after), len - after).map_or(len, |i| after + i);
            self.st.long_line(std::slice::from_raw_parts(base, end - pos), kind);
            pos = end + 1;
        }
    }
}

// ---------------------------------------------------------------- phase 2

struct Resolver {
    table: Vec<u64>,
    ev: Scatter<Ev>,
    cl: Scatter<Click>,
    shift2: u32,
    entry_chunk_bytes: usize,
}

const PTR_MASK: u64 = (1 << 48) - 1;
/// Link lookups run in blocks of this many records, one stage at a time.
const BLOCK: usize = 128;

impl Resolver {
    /// Dedup one event partition and scatter its surviving valid events by
    /// raw user.
    unsafe fn events(&mut self, lists: &[List<Entry>]) {
        let t = prof::start();
        let n: usize = lists.iter().map(|l| l.count()).sum();
        if n == 0 {
            return;
        }
        let cap = (n * 2).next_power_of_two().max(64);
        self.table.clear();
        self.table.resize(cap, 0);
        let mask = cap - 1;
        let table = self.table.as_mut_ptr();
        for list in lists {
            list.for_each_chunk(|chunk| {
                for entry in chunk.iter_mut() {
                    let h = mix(entry.ekey);
                    let tag = h << 48;
                    let me = entry as *mut Entry;
                    let mut i = (h >> 20) as usize & mask;
                    loop {
                        let slot = *table.add(i);
                        if slot == 0 {
                            *table.add(i) = tag | me as u64;
                            break;
                        }
                        if slot & !PTR_MASK == tag {
                            let other = (slot & PTR_MASK) as *mut Entry;
                            if (*other).ekey == (*me).ekey {
                                // Same event: the smaller ingest stays. On
                                // a tie the copies are identical.
                                if (*me).ingest < (*other).ingest {
                                    (*other).w3 = DEAD;
                                    *table.add(i) = tag | me as u64;
                                } else {
                                    (*me).w3 = DEAD;
                                }
                                break;
                            }
                        }
                        i = (i + 1) & mask;
                    }
                }
            });
        }
        let t = prof::add(4, t);
        let shift2 = self.shift2;
        let ev = &mut self.ev;
        for list in lists {
            list.for_each_chunk(|chunk| {
                for entry in chunk.iter() {
                    if entry.w3 != DEAD {
                        ev.push((mix(entry.ukey) >> shift2) as usize, Ev { pkey: entry.ukey, v: entry.w3 });
                    }
                }
            });
        }
        // The entries of this partition are dead: their chunks become the
        // next output chunks, so resolve needs almost no fresh memory.
        let bytes = self.entry_chunk_bytes;
        for list in lists {
            list.release(|chunk| {
                if self.cl.spare() < 16 {
                    self.cl.recycle(chunk, bytes);
                } else {
                    self.ev.recycle(chunk, bytes);
                }
            });
        }
        prof::add(5, t);
    }

    unsafe fn clicks(&mut self, block: &[Click]) {
        let t = prof::start();
        let shift2 = self.shift2;
        let cl = &mut self.cl;
        for click in block {
            cl.push((mix(click.key) >> shift2) as usize, *click);
        }
        prof::add(6, t);
    }
}

// ---------------------------------------------------------------- phase 3

struct PersonPhase<'a> {
    fast: &'a Fast<'a>,
    // person key -> local id
    tab_keys: Vec<u64>,
    tab_ids: Vec<u32>,
    pkeys: Vec<u64>,
    counts: Vec<u32>,
    /// Events per time bin, then the write cursor of each bin.
    bins: Vec<u32>,
    /// One word per event, `ts_rel << 32 | local person id`: in input order
    /// and then in bin order.
    words: Vec<u64>,
    binned: Vec<u64>,
    /// `ts_rel` of every event, person after person, each in bin order.
    ev: Vec<u32>,
    /// The purchases: `(value, local person id)` as they come, then the
    /// values person after person, with the end offset of each person.
    buys: Vec<(u64, u32)>,
    bcounts: Vec<u32>,
    bought: Vec<u64>,
    ccounts: Vec<u32>,
    cids: Vec<u32>,
    cl: Vec<[u64; 2]>,
    // The raw pass: one link lookup per raw user.
    links: LinkMap,
    shift2: u32,
    /// Per raw user: the partition of its person, or `LOCAL` for a user
    /// in no link, who is a whole person and is finished where it is.
    gpart: Vec<u32>,
    gperson: Vec<u64>,
    /// Forwarded users, as runs of words: `[events | clicks << 32, person,
    /// purchases]`, then the event times two a word, the purchase values
    /// and the `(click tc, click id)` pairs.
    fwd: Option<Scatter<u64>>,
    // The person pass: the chunks of a partition, and its clicks as they
    // come with `(local person id, clicks)` per run.
    chunks: Vec<(*const u64, usize)>,
    cbuf: Vec<u64>,
    cruns: Vec<(u32, u32)>,
    // sliding window of campaigns with a click in reach
    cnt: Vec<u32>,
    pos: Vec<u32>,
    active: Vec<u32>,
    catsum: Vec<u64>,
    touched: Vec<u8>,
    /// (campaign, day) -> [sessions, purchasing, revenue, assisted]
    agg: Vec<[u64; 4]>,
    out: UserOut<'a>,
    /// Summed sessions and revenue of the rows emitted, for the self-check.
    check: (u64, u64),
}

const LOCAL: u32 = u32::MAX;

/// Events are never sorted by time. They are put in order of their time
/// bin only: `ts_rel >> BIN_SHIFT`, bins of 2^20 ms. Two events of one bin
/// are under 2^20 ms apart, which is less than the session gap, so a bin
/// never holds a session break. A break can only fall between two
/// consecutive non-empty bins of a person, and it does exactly when the
/// smallest time of the later bin is more than the gap after the largest
/// time of the earlier one: those two are neighbours in time order.
///
/// So the events of a person are only times here. The purchases (3% of the
/// events) travel on the side, sorted by time per person, and a session
/// takes those up to its last event when it closes.
const BIN_SHIFT: u32 = 20;
const _: () = assert!((1i64 << BIN_SHIFT) - 1 <= SESSION_GAP_MS);

#[inline(always)]
fn sort_u64(v: &mut [u64]) {
    if v.len() <= 20 {
        for i in 1..v.len() {
            let x = v[i];
            let mut j = i;
            while j > 0 && v[j - 1] > x {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[j] = x;
        }
    } else {
        v.sort_unstable();
    }
}

impl<'a> PersonPhase<'a> {
    fn new(fast: &'a Fast<'a>, out: UserOut<'a>, links: LinkMap, bitsf: u32) -> Self {
        PersonPhase {
            links,
            shift2: 64 - bitsf,
            gpart: Vec::new(),
            gperson: Vec::new(),
            fwd: Some(Scatter::new(1 << bitsf, CHUNK_BYTES)),
            chunks: Vec::new(),
            cbuf: Vec::new(),
            cruns: Vec::new(),
            fast,
            tab_keys: Vec::new(),
            tab_ids: Vec::new(),
            pkeys: Vec::new(),
            counts: Vec::new(),
            bins: Vec::new(),
            words: Vec::new(),
            binned: Vec::new(),
            ev: Vec::new(),
            buys: Vec::new(),
            bcounts: Vec::new(),
            bought: Vec::new(),
            ccounts: Vec::new(),
            cids: Vec::new(),
            cl: Vec::new(),
            cnt: vec![0; fast.ncamp],
            pos: vec![0; fast.ncamp],
            active: Vec::new(),
            catsum: vec![0; 256],
            touched: Vec::new(),
            agg: vec![[0; 4]; (fast.ncamp + 1) * fast.ndays],
            out,
            check: (0, 0),
        }
    }

    #[cold]
    #[inline(never)]
    fn grow_table(&mut self) {
        let cap = self.tab_keys.len() * 2;
        self.tab_keys.clear();
        self.tab_keys.resize(cap, 0);
        self.tab_ids.clear();
        self.tab_ids.resize(cap, 0);
        let mask = cap - 1;
        for (id, &key) in self.pkeys.iter().enumerate() {
            let mut i = (mix(key) >> 8) as usize & mask;
            while self.tab_keys[i] != 0 {
                i = (i + 1) & mask;
            }
            self.tab_keys[i] = key;
            self.tab_ids[i] = id as u32;
        }
    }

    /// Empty tables for a partition of at most `n` events.
    #[inline(always)]
    fn begin(&mut self, n: usize) {
        let cap = (n / 8).next_power_of_two().max(256);
        self.tab_keys.clear();
        self.tab_keys.resize(cap, 0);
        self.tab_ids.clear();
        self.tab_ids.resize(cap, 0);
        self.pkeys.clear();
        self.counts.clear();
        // `ts_rel < wlen`, so the last bin is that of `wlen - 1`.
        let nbins = ((self.fast.wlen - 1) >> BIN_SHIFT) as usize + 1;
        self.bins.clear();
        self.bins.resize(nbins, 0);
        self.words.clear();
        self.words.reserve(n);
        self.buys.clear();
    }

    /// The local id of `key`. A new id starts with a count of zero.
    #[inline(always)]
    unsafe fn intern(&mut self, key: u64) -> u32 {
        let mut mask = self.tab_keys.len() - 1;
        let mut i = (mix(key) >> 8) as usize & mask;
        loop {
            let k = *self.tab_keys.get_unchecked(i);
            if k == key {
                return *self.tab_ids.get_unchecked(i);
            }
            if k == 0 {
                if (self.pkeys.len() + 1) * 2 > self.tab_keys.len() {
                    self.grow_table();
                    mask = self.tab_keys.len() - 1;
                    i = (mix(key) >> 8) as usize & mask;
                    continue;
                }
                let id = self.pkeys.len() as u32;
                *self.tab_keys.get_unchecked_mut(i) = key;
                *self.tab_ids.get_unchecked_mut(i) = id;
                self.pkeys.push(key);
                self.counts.push(0);
                return id;
            }
            i = (i + 1) & mask;
        }
    }

    /// From `words` (one per event, input order), the event count per time
    /// bin in `bins` and per id in `counts`, and `buys`: every id's event
    /// times in bin order in `ev`, and its purchases in `bought`.
    /// `counts[i]` and `bcounts[i]` become the end offsets of id `i`.
    #[inline(always)]
    unsafe fn order(&mut self) {
        let n = self.words.len();
        let ids = self.pkeys.len();
        // The whole partition into bin order.
        let mut sum = 0u32;
        for c in self.bins.iter_mut() {
            let k = *c;
            *c = sum;
            sum += k;
        }
        self.binned.clear();
        self.binned.reserve(n);
        self.binned.set_len(n);
        for &word in &self.words {
            let w = self.bins.get_unchecked_mut((word >> (32 + BIN_SHIFT)) as usize);
            *self.binned.get_unchecked_mut(*w as usize) = word;
            *w += 1;
        }
        // A stable scatter by id, so each id's events are in bin order.
        // counts -> start offsets (counts[i] becomes the write cursor).
        let mut sum = 0u32;
        for c in self.counts.iter_mut() {
            let k = *c;
            *c = sum;
            sum += k;
        }
        self.ev.clear();
        // Room for one more: a forwarded run reads its times as whole words.
        self.ev.reserve(n + 2);
        self.ev.set_len(n);
        for &word in &self.binned {
            let w = self.counts.get_unchecked_mut(word as u32 as usize);
            *self.ev.get_unchecked_mut(*w as usize) = (word >> 32) as u32;
            *w += 1;
        }

        // The purchases, id after id.
        self.bcounts.clear();
        self.bcounts.resize(ids, 0);
        for &(_, id) in &self.buys {
            *self.bcounts.get_unchecked_mut(id as usize) += 1;
        }
        let mut sum = 0u32;
        for c in self.bcounts.iter_mut() {
            let k = *c;
            *c = sum;
            sum += k;
        }
        self.bought.clear();
        self.bought.reserve(self.buys.len());
        self.bought.set_len(self.buys.len());
        for &(v, id) in &self.buys {
            let w = self.bcounts.get_unchecked_mut(id as usize);
            *self.bought.get_unchecked_mut(*w as usize) = v;
            *w += 1;
        }
    }

    /// The raw pass, over a partition of records keyed by RAW user. One
    /// link lookup per user, not per event. A user in no link is a whole
    /// person and is finished here. Every other user is forwarded as one
    /// run to the partition of its person's name; for the name itself that
    /// is this partition, in the person pass.
    unsafe fn partition_raw(&mut self, evs: &[List<Ev>], cls: &[List<Click>]) {
        let t = prof::start();
        let n: usize = evs.iter().map(|l| l.count()).sum();
        let nc: usize = cls.iter().map(|l| l.count()).sum();
        if n == 0 && nc == 0 {
            return;
        }
        self.begin(n);
        // A local id per user, how many events each user has and each time
        // bin has, and one word per event.
        let words = self.words.as_mut_ptr();
        let mut at = 0usize;
        for list in evs {
            list.for_each_chunk(|chunk| {
                for rec in chunk.iter() {
                    let id = self.intern(rec.pkey);
                    *self.counts.get_unchecked_mut(id as usize) += 1;
                    *self.bins.get_unchecked_mut((rec.v >> (32 + BIN_SHIFT)) as usize) += 1;
                    *words.add(at) = (rec.v & !0xFFFF_FFFF) | u64::from(id);
                    at += 1;
                    if rec.v as u32 != 0 {
                        self.buys.push((rec.v, id));
                    }
                }
            });
        }
        self.words.set_len(n);
        let users = self.pkeys.len();
        let t = prof::add(10, t);

        // Who is linked. A block of lookups a stage, so their cache misses
        // overlap.
        let links = self.links;
        let shift2 = self.shift2;
        self.gpart.clear();
        self.gpart.resize(users, LOCAL);
        self.gperson.clear();
        self.gperson.resize(users, 0);
        {
            let mut hashes = [0u64; BLOCK];
            let mut starts = [0usize; BLOCK];
            let mut g = 0usize;
            for block in self.pkeys.chunks(BLOCK) {
                for (h, &key) in hashes.iter_mut().zip(block) {
                    *h = links.stage1(key);
                }
                for (at, &h) in starts.iter_mut().zip(&hashes[..block.len()]) {
                    *at = links.stage2(h);
                }
                for (&at, &key) in starts.iter().zip(block) {
                    if at != links.nowhere {
                        let person = links.stage3(at, key);
                        *self.gperson.get_unchecked_mut(g) = person;
                        *self.gpart.get_unchecked_mut(g) = (mix(person) >> shift2) as u32;
                    }
                    g += 1;
                }
            }
        }
        let t = prof::add(11, t);

        self.order();
        let t = prof::add(12, t);

        // Clicks, user after user. One whose user has no events here may
        // still belong to a linked person with events elsewhere: it goes
        // there as a run of no events and one click.
        self.ccounts.clear();
        self.ccounts.resize(users + 1, 0);
        self.cl.clear();
        if nc > 0 {
            self.cids.clear();
            self.cids.reserve(nc);
            let fwd = self.fwd.as_mut().unwrap_unchecked();
            let mask = self.tab_keys.len() - 1;
            for list in cls {
                list.for_each_chunk(|chunk| {
                    for rec in chunk.iter() {
                        let mut i = (mix(rec.key) >> 8) as usize & mask;
                        let id = loop {
                            let k = *self.tab_keys.get_unchecked(i);
                            if k == rec.key {
                                break *self.tab_ids.get_unchecked(i);
                            }
                            if k == 0 {
                                break u32::MAX;
                            }
                            i = (i + 1) & mask;
                        };
                        self.cids.push(id);
                        if id != u32::MAX {
                            *self.ccounts.get_unchecked_mut(id as usize) += 1;
                        } else {
                            let at = links.start(mix(rec.key));
                            if at != links.nowhere {
                                let person = links.stage3(at, rec.key);
                                let part = (mix(person) >> shift2) as usize;
                                fwd.push(part, 1 << 32);
                                fwd.push(part, person);
                                fwd.push(part, 0);
                                fwd.push(part, rec.tc);
                                fwd.push(part, rec.ckey);
                            }
                        }
                    }
                });
            }
            let mut sum = 0u32;
            for c in self.ccounts.iter_mut() {
                let k = *c;
                *c = sum;
                sum += k;
            }
            self.cl.reserve(sum as usize);
            self.cl.set_len(sum as usize);
            let mut at = 0usize;
            for list in cls {
                list.for_each_chunk(|chunk| {
                    for rec in chunk.iter() {
                        let id = *self.cids.get_unchecked(at);
                        at += 1;
                        if id != u32::MAX {
                            let w = self.ccounts.get_unchecked_mut(id as usize);
                            *self.cl.get_unchecked_mut(*w as usize) = [rec.tc, rec.ckey];
                            *w += 1;
                        }
                    }
                });
            }
        }
        let t = prof::add(13, t);

        // Forward the linked users: times, purchases and clicks.
        {
            let fwd = self.fwd.as_mut().unwrap_unchecked();
            let mut ev_start = 0usize;
            let mut cl_start = 0usize;
            let mut buy_start = 0usize;
            for g in 0..users {
                let ev_end = *self.counts.get_unchecked(g) as usize;
                let cl_end = if nc > 0 { *self.ccounts.get_unchecked(g) as usize } else { 0 };
                let buy_end = *self.bcounts.get_unchecked(g) as usize;
                let part = *self.gpart.get_unchecked(g);
                if part != LOCAL {
                    let part = part as usize;
                    let evc = ev_end - ev_start;
                    fwd.push(part, evc as u64 | ((cl_end - cl_start) as u64) << 32);
                    fwd.push(part, *self.gperson.get_unchecked(g));
                    fwd.push(part, (buy_end - buy_start) as u64);
                    // Two times a word; an odd count drags one stray time
                    // along, which the reader skips.
                    let times = self.ev.as_ptr().add(ev_start) as *const u64;
                    fwd.push_words(part, std::slice::from_raw_parts(times, evc.div_ceil(2)));
                    fwd.push_words(part, self.bought.get_unchecked(buy_start..buy_end));
                    if cl_end > cl_start {
                        let pairs = self.cl.as_ptr().add(cl_start) as *const u64;
                        fwd.push_words(part, std::slice::from_raw_parts(pairs, (cl_end - cl_start) * 2));
                    }
                }
                ev_start = ev_end;
                cl_start = cl_end;
                buy_start = buy_end;
            }
        }
        let t = prof::add(15, t);

        // Finish the unlinked ones.
        let mut ev_start = 0usize;
        let mut cl_start = 0usize;
        let mut buy_start = 0usize;
        for g in 0..users {
            let ev_end = self.counts[g] as usize;
            let cl_end = if nc > 0 { self.ccounts[g] as usize } else { 0 };
            let buy_end = self.bcounts[g] as usize;
            if self.gpart[g] == LOCAL {
                self.person(g, ev_start, ev_end, cl_start, cl_end, buy_start, buy_end);
            }
            ev_start = ev_end;
            cl_start = cl_end;
            buy_start = buy_end;
        }
        let t = prof::add(14, t);

        // This partition's chunks are dead: they become forwarding chunks.
        let fwd = self.fwd.as_mut().unwrap_unchecked();
        let bytes = fwd.chunk_bytes();
        for list in evs {
            list.release(|chunk| fwd.recycle(chunk, bytes));
        }
        for list in cls {
            list.release(|chunk| fwd.recycle(chunk, bytes));
        }
        let t = prof::add(16, t);
        if self.out.flush_full() {
            prof::add(9, t);
        }
    }

    /// The person pass, over a partition of forwarded runs keyed by person.
    /// One table probe per run. A run never crosses from one list to the
    /// next, but it may cross chunks.
    unsafe fn partition(&mut self, runs: &[List<u64>]) {
        let t = prof::start();
        self.chunks.clear();
        for list in runs {
            list.chunks_oldest_first(&mut self.chunks);
        }
        let total: usize = self.chunks.iter().map(|c| c.1).sum();
        if total == 0 {
            return;
        }
        self.begin(total);
        // A word holds two event times.
        self.words.reserve(total * 2);
        self.ccounts.clear();
        self.cbuf.clear();
        self.cbuf.reserve(total);
        self.cruns.clear();
        let words = self.words.as_mut_ptr();
        let cbuf = self.cbuf.as_mut_ptr();
        let (mut n, mut nc) = (0usize, 0usize);
        // What the next word is: 0 the counts, 1 the person, 2 the number
        // of purchases, then the run's times, purchases and clicks.
        let mut stage = 0u32;
        let mut head = 0u64;
        let mut id = 0u32;
        let (mut left_ev, mut left_buy, mut left_cl) = (0usize, 0usize, 0usize);
        for at in 0..self.chunks.len() {
            let (ptr, len) = self.chunks[at];
            let mut i = 0usize;
            while i < len {
                if left_ev > 0 {
                    // Two times a word.
                    let take = left_ev.div_ceil(2).min(len - i);
                    let times = ptr.add(i) as *const u32;
                    let count = (take * 2).min(left_ev);
                    for k in 0..count {
                        let ts = *times.add(k);
                        *self.bins.get_unchecked_mut((ts >> BIN_SHIFT) as usize) += 1;
                        *words.add(n) = u64::from(ts) << 32 | u64::from(id);
                        n += 1;
                    }
                    i += take;
                    left_ev -= count;
                    continue;
                }
                if left_buy > 0 {
                    let take = left_buy.min(len - i);
                    for k in i..i + take {
                        self.buys.push((*ptr.add(k), id));
                    }
                    i += take;
                    left_buy -= take;
                    continue;
                }
                if left_cl > 0 {
                    let take = left_cl.min(len - i);
                    sys::copy_words(ptr.add(i), cbuf.add(nc), take);
                    nc += take;
                    i += take;
                    left_cl -= take;
                    continue;
                }
                let w = *ptr.add(i);
                i += 1;
                if stage == 0 {
                    head = w;
                    stage = 1;
                } else if stage == 1 {
                    id = self.intern(w);
                    if self.ccounts.len() < self.pkeys.len() {
                        self.ccounts.push(0);
                    }
                    stage = 2;
                } else {
                    let (evc, clc) = (head as u32, (head >> 32) as u32);
                    *self.counts.get_unchecked_mut(id as usize) += evc;
                    if clc > 0 {
                        *self.ccounts.get_unchecked_mut(id as usize) += clc;
                        self.cruns.push((id, clc));
                    }
                    left_ev = evc as usize;
                    left_buy = w as usize;
                    left_cl = clc as usize * 2;
                    stage = 0;
                }
            }
        }
        self.words.set_len(n);
        let people = self.pkeys.len();
        self.order();

        // Clicks, person after person.
        let mut sum = 0u32;
        for c in self.ccounts.iter_mut() {
            let k = *c;
            *c = sum;
            sum += k;
        }
        self.cl.clear();
        self.cl.reserve(sum as usize);
        self.cl.set_len(sum as usize);
        let cl = self.cl.as_mut_ptr() as *mut u64;
        let mut from = cbuf as *const u64;
        for &(id, clc) in &self.cruns {
            let w = self.ccounts.get_unchecked_mut(id as usize);
            sys::copy_words(from, cl.add(*w as usize * 2), clc as usize * 2);
            *w += clc;
            from = from.add(clc as usize * 2);
        }

        let t = prof::add(7, t);
        let mut ev_start = 0usize;
        let mut cl_start = 0usize;
        let mut buy_start = 0usize;
        for p in 0..people {
            let ev_end = self.counts[p] as usize;
            let cl_end = self.ccounts[p] as usize;
            let buy_end = self.bcounts[p] as usize;
            self.person(p, ev_start, ev_end, cl_start, cl_end, buy_start, buy_end);
            ev_start = ev_end;
            cl_start = cl_end;
            buy_start = buy_end;
        }
        let t = prof::add(8, t);
        if self.out.flush_full() {
            prof::add(9, t);
        }
    }

    #[inline(always)]
    fn emit(&mut self, pkey: u64, day_rel: u32, sessions: u32, events: u32, revenue: u64) {
        let mut top = 0xFFu8;
        if !self.touched.is_empty() {
            let mut best = 0u64;
            for &c in &self.touched {
                let sum = self.catsum[c as usize];
                if sum > best || (sum == best && c < top) {
                    best = sum;
                    top = c;
                }
                self.catsum[c as usize] = 0;
            }
            self.touched.clear();
        }
        self.check.0 += u64::from(sessions);
        self.check.1 += revenue;
        self.out.push(pkey, day_rel, sessions, events, revenue, top);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn person(
        &mut self,
        p: usize,
        ev_start: usize,
        ev_end: usize,
        cl_start: usize,
        cl_end: usize,
        buy_start: usize,
        buy_end: usize,
    ) {
        let pkey = self.pkeys[p];
        let ev = std::slice::from_raw_parts(self.ev.as_ptr().add(ev_start), ev_end - ev_start);
        let cl = std::slice::from_raw_parts_mut(self.cl.as_mut_ptr().add(cl_start), cl_end - cl_start);
        if cl.len() > 1 {
            // By ts, then by click id descending: the last click at or
            // before a session start is the latest with the smallest id.
            cl.sort_unstable_by(|a, b| (a[0] >> 32).cmp(&(b[0] >> 32)).then_with(|| cmp_ids(b[1], a[1])));
        }
        if buy_end - buy_start > 1 {
            // The time is the top half of a value.
            sort_u64(std::slice::from_raw_parts_mut(self.bought.as_mut_ptr().add(buy_start), buy_end - buy_start));
        }
        let mut buy = buy_start;
        let ws_in_day = self.fast.ws_in_day;
        let gap = SESSION_GAP_MS as u32;
        let n = ev.len();
        let mut win = (0usize, 0usize);
        let mut cur_day = u32::MAX;
        let (mut d_sessions, mut d_events, mut d_revenue) = (0u32, 0u32, 0u64);
        // The open session: its start and the largest time seen in it.
        let (mut start, mut last) = (0u32, 0u32);
        let mut i = 0usize;
        loop {
            // One run of events in the same bin, and its extremes.
            let (mut first, mut latest) = (u32::MAX, 0u32);
            let mut j = i;
            if i < n {
                let bin = *ev.get_unchecked(i) >> BIN_SHIFT;
                while j < n {
                    let t = *ev.get_unchecked(j);
                    if t >> BIN_SHIFT != bin {
                        break;
                    }
                    first = first.min(t);
                    latest = latest.max(t);
                    j += 1;
                }
            }
            // `first` is past every time when the events have run out.
            if i == 0 || first - last > gap {
                if i != 0 {
                    // The session is complete: its purchases are those not
                    // taken yet that are no later than its last event.
                    let mut revenue = 0u64;
                    while buy < buy_end && (*self.bought.get_unchecked(buy) >> 32) as u32 <= last {
                        let pc = *self.bought.get_unchecked(buy) as u32;
                        let price = u64::from(pc >> 8);
                        let cat = (pc & 0xFF) as u8;
                        revenue += price;
                        let sum = self.catsum.get_unchecked_mut(cat as usize);
                        if *sum == 0 {
                            self.touched.push(cat);
                        }
                        *sum += price;
                        buy += 1;
                    }
                    d_revenue += revenue;
                    self.credit(cl, &mut win, u64::from(start), cur_day as usize, revenue);
                }
                if i == n {
                    break;
                }
                start = first;
                let day_rel = ((u64::from(start) + ws_in_day) / DAY_MS as u64) as u32;
                if day_rel != cur_day {
                    if cur_day != u32::MAX {
                        self.emit(pkey, cur_day, d_sessions, d_events, d_revenue);
                    }
                    cur_day = day_rel;
                    d_sessions = 0;
                    d_events = 0;
                    d_revenue = 0;
                }
                d_sessions += 1;
            }
            last = latest;
            d_events += (j - i) as u32;
            i = j;
        }
        if cur_day != u32::MAX {
            self.emit(pkey, cur_day, d_sessions, d_events, d_revenue);
        }
        for &c in &self.active {
            *self.cnt.get_unchecked_mut(c as usize) = 0;
        }
        self.active.clear();
    }

    /// Credit one finished session to a click and count its assists. `cl`
    /// is the person's clicks in order, `win` the range of them in reach of
    /// the previous session; sessions come in order of their start.
    #[inline(always)]
    unsafe fn credit(&mut self, cl: &[[u64; 2]], win: &mut (usize, usize), start: u64, day: usize, revenue: u64) {
        let ndays = self.fast.ndays;
        let mut credit = self.fast.ncamp;
        if !cl.is_empty() {
            let reach = CLICK_REACH_MS as u64;
            let (mut lo, mut hi) = *win;
            // Click times are relative to window_start - 7 days, so a
            // session start S is S + reach on that clock.
            let upto = start + reach;
            while hi < cl.len() && (cl.get_unchecked(hi)[0] >> 32) <= upto {
                let c = cl.get_unchecked(hi)[0] as u32 as usize;
                if *self.cnt.get_unchecked(c) == 0 {
                    *self.pos.get_unchecked_mut(c) = self.active.len() as u32;
                    self.active.push(c as u32);
                }
                *self.cnt.get_unchecked_mut(c) += 1;
                hi += 1;
            }
            while lo < hi && (cl.get_unchecked(lo)[0] >> 32) < start {
                let c = cl.get_unchecked(lo)[0] as u32 as usize;
                *self.cnt.get_unchecked_mut(c) -= 1;
                if *self.cnt.get_unchecked(c) == 0 {
                    let at = *self.pos.get_unchecked(c) as usize;
                    let moved = *self.active.last().unwrap_unchecked();
                    *self.active.get_unchecked_mut(at) = moved;
                    *self.pos.get_unchecked_mut(moved as usize) = at as u32;
                    self.active.pop();
                }
                lo += 1;
            }
            if hi > lo {
                credit = cl.get_unchecked(hi - 1)[0] as u32 as usize;
            }
            for &c in &self.active {
                self.agg.get_unchecked_mut(c as usize * ndays + day)[3] += 1;
            }
            *win = (lo, hi);
        }
        let cell = self.agg.get_unchecked_mut(credit * ndays + day);
        cell[0] += 1;
        cell[1] += u64::from(revenue > 0);
        cell[2] += revenue;
    }
}

// ---------------------------------------------------------------- driver

/// Partition count, as a power of two, for about `per_part` estimated rows a
/// partition.
fn bits_for(estimate: u64, per_part: u64, forced: Option<u32>) -> u32 {
    match forced {
        Some(b) => b.clamp(1, 14),
        None => {
            let parts = (estimate / per_part).next_power_of_two().clamp(16, 4096);
            parts.trailing_zeros()
        }
    }
}

struct Timer {
    on: bool,
    t0: Instant,
    last: Instant,
}

impl Timer {
    fn lap(&mut self, name: &str, cpu: f64) {
        if self.on {
            let now = Instant::now();
            eprintln!(
                "[engine] {name:<10} wall {:7.1} ms  cpu {:7.1} ms  (at {:7.1} ms)",
                (now - self.last).as_secs_f64() * 1e3,
                cpu * 1e3,
                (now - self.t0).as_secs_f64() * 1e3
            );
            self.last = now;
        }
    }
}

/// Run the fast engine. Returns `None` without doing any work when the
/// dataset breaks a precondition (see `Fast::new`). With `cfg.collect` the
/// rows come back in memory; otherwise they are written under `out`.
pub fn run(data: &Path, out: Option<&Path>, lk: &Lookups, cfg: &Config, started: Instant) -> Option<Tables> {
    let mut timer = Timer { on: cfg.timing, t0: started, last: started };
    timer.lap("lookups", 0.0);
    prof::ON.store(cfg.timing, Ordering::Relaxed);
    let event_files: Vec<Mapped> = list_csv(&data.join("events")).iter().map(|p| sys::map_file(p)).collect();
    let click_files: Vec<Mapped> = list_csv(&data.join("clicks")).iter().map(|p| sys::map_file(p)).collect();
    let link_files: Vec<Mapped> = list_csv(&data.join("links")).iter().map(|p| sys::map_file(p)).collect();
    let event_bytes: u64 = event_files.iter().map(|m| m.len as u64).sum();
    let fast = Fast::new(lk, event_bytes)?;
    let fast = &fast;
    let threads = cfg.threads;

    let estimate = event_bytes / 64;
    // A scatter costs more per record the more partitions it has, and the
    // consumer of a partition is flat up to about 40k records. Measured at
    // 20M events, person side: 1024, 2048 and 4096 partitions cost 150, 166
    // and 196 ms of resolve. At 60M events, event side: 2048 against 4096
    // saves 57 ms of parse and costs 28 ms of resolve. So both sides aim
    // at 20k to 40k records a partition.
    let bits1 = bits_for(estimate, 48_000, cfg.p1_bits);
    let bits2 = bits_for(estimate, 48_000, cfg.p2_bits);
    let (parts1, parts2) = (1usize << bits1, 1usize << bits2);
    // Partitions of the person pass, which sees the linked users only.
    let bitsf = std::env::var("ENGINE_PF_BITS").ok().and_then(|v| v.parse::<u32>().ok()).map_or(bits2, |b| b.clamp(1, 14));

    // Jobs: the links job first, then event chunks (largest files first),
    // then click chunks.
    let mut jobs: Vec<Job> = vec![Job { kind: Kind::Links, file: 0, start: 0, end: 0 }];
    for (kind, files) in [(Kind::Events, &event_files), (Kind::Clicks, &click_files)] {
        let mut order: Vec<usize> = (0..files.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(files[i].len));
        for file in order {
            let len = files[file].len;
            let mut start = 0;
            while start < len {
                let end = if len - start < cfg.chunk_bytes * 3 / 2 { len } else { start + cfg.chunk_bytes };
                jobs.push(Job { kind, file, start, end });
                start = end;
            }
        }
    }
    timer.lap("setup", 0.0);

    // ---- phase 1
    // Parsed chunks are unmapped by the worker that parsed them. Measured
    // neutral on time at 20M, a little faster at 60M, and it halves peak
    // RSS (the kernel no longer tears the mappings down inside exit).
    let drop_pages = std::env::var("ENGINE_DROP").map_or(true, |v| v != "0");
    let next = AtomicUsize::new(0);
    let link_map: Mutex<Option<LinkMap>> = Mutex::new(None);
    let cpu_total = Mutex::new(0.0f64);
    let mut entry_lists: Vec<Vec<List<Entry>>> = Vec::new();
    let mut click_blocks: Vec<Vec<Click>> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| unsafe {
                    let mut parser = Parser {
                        st: ParseState {
                            fast,
                            cache: DateCache::new(),
                            entries: Scatter::new(parts1, CHUNK_BYTES),
                            shift1: 64 - bits1,
                            clicks: Vec::new(),
                            pairs: Vec::new(),
                        },
                        scan: Scan::new(WINDOW),
                    };
                    loop {
                        let at = next.fetch_add(1, Ordering::Relaxed);
                        let Some(job) = jobs.get(at) else {
                            break;
                        };
                        match job.kind {
                            Kind::Links => {
                                let t = prof::start();
                                for m in &link_files {
                                    parser.run_chunk(*m, 0, m.len, Kind::Links);
                                }
                                let t = prof::add(2, t);
                                let map = LinkMap::build(&parser.st.pairs);
                                prof::add(3, t);
                                *link_map.lock().unwrap() = Some(map);
                            }
                            Kind::Events => {
                                parser.run_chunk(event_files[job.file], job.start, job.end, Kind::Events);
                                sys::drop_pages(event_files[job.file], job.start, job.end, drop_pages);
                            }
                            Kind::Clicks => {
                                parser.run_chunk(click_files[job.file], job.start, job.end, Kind::Clicks);
                                sys::drop_pages(click_files[job.file], job.start, job.end, drop_pages);
                            }
                        }
                    }
                    *cpu_total.lock().unwrap() += sys::thread_cpu();
                    (parser.st.entries.finish(), parser.st.clicks)
                })
            })
            .collect();
        for h in handles {
            let (entries, clicks) = h.join().unwrap();
            entry_lists.push(entries);
            click_blocks.extend(clicks);
        }
    });
    // The inputs are dead from here on. Unmapping them on a side thread
    // moves about 10 ms of page-table work out of exit at 20M events, but
    // costs the resolve phase as much (measured), so it is off by default.
    let unmapper = std::env::var_os("ENGINE_UNMAP").is_some().then(|| {
        let maps: Vec<Mapped> = event_files.iter().chain(&click_files).chain(&link_files).copied().collect();
        std::thread::spawn(move || {
            for m in maps {
                sys::unmap_file(m);
            }
        })
    });
    let links = link_map.into_inner().unwrap().unwrap();
    let links = &links;
    timer.lap("parse", std::mem::take(&mut *cpu_total.lock().unwrap()));

    // ---- phase 2
    let next = AtomicUsize::new(0);
    let entry_lists = &entry_lists;
    let click_blocks = &click_blocks;
    let mut ev_lists: Vec<Vec<List<Ev>>> = Vec::new();
    let mut cl_lists: Vec<Vec<List<Click>>> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| unsafe {
                    let mut r = Resolver {
                        table: Vec::new(),
                        ev: Scatter::new(parts2, CHUNK_BYTES),
                        cl: Scatter::new(parts2, CHUNK_BYTES),
                        shift2: 64 - bits2,
                        entry_chunk_bytes: Scatter::<Entry>::new(1, CHUNK_BYTES).chunk_bytes(),
                    };
                    let mut lists: Vec<List<Entry>> = Vec::with_capacity(threads);
                    loop {
                        let at = next.fetch_add(1, Ordering::Relaxed);
                        if at < parts1 {
                            lists.clear();
                            lists.extend(entry_lists.iter().map(|t| t[at]));
                            r.events(&lists);
                        } else if let Some(block) = click_blocks.get(at - parts1) {
                            r.clicks(block);
                        } else {
                            break;
                        }
                    }
                    *cpu_total.lock().unwrap() += sys::thread_cpu();
                    (r.ev.finish(), r.cl.finish())
                })
            })
            .collect();
        for h in handles {
            let (ev, cl) = h.join().unwrap();
            ev_lists.push(ev);
            cl_lists.push(cl);
        }
    });
    timer.lap("resolve", std::mem::take(&mut *cpu_total.lock().unwrap()));

    // ---- phase 3: biggest partitions first, so the tail is short jobs.
    // Both passes run on the same threads, so a thread keeps its tables
    // and its output file.
    let mut order: Vec<usize> = (0..parts2).collect();
    let weight = |p: usize| -> u32 { ev_lists.iter().map(|t| t[p].chunks).sum() };
    order.sort_by_key(|&p| std::cmp::Reverse(weight(p)));
    let next = AtomicUsize::new(0);
    let ev_lists = &ev_lists;
    let cl_lists = &cl_lists;
    let order = &order;
    // The raw pass forwards into per-thread lists; the person pass starts
    // when every thread has handed its lists in.
    let barrier = std::sync::Barrier::new(threads);
    let forwarded: Mutex<Vec<Vec<List<u64>>>> = Mutex::new(Vec::new());
    let order2: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    let next2 = AtomicUsize::new(0);
    let (barrier, forwarded, order2, next2) = (&barrier, &forwarded, &order2, &next2);
    let collected: Mutex<Vec<UserRow>> = Mutex::new(Vec::new());
    let collected_ref = &collected;
    let mut agg = vec![[0u64; 4]; (fast.ncamp + 1) * fast.ndays];
    let mut total_rows = 0u64;
    let out_dir = out.map(|o| o.join("user_daily"));
    let out_dir = &out_dir;
    let next = &next;
    let cpu_total = &cpu_total;
    let mut check = (0u64, 0u64);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || unsafe {
                    let user_out =
                        UserOut::new(out_dir.as_deref(), t, lk, fast.ws_day, fast.ndays, cfg.collect.then_some(collected_ref));
                    let mut phase = PersonPhase::new(fast, user_out, *links, bitsf);
                    let mut evs: Vec<List<Ev>> = Vec::with_capacity(threads);
                    let mut cls: Vec<List<Click>> = Vec::with_capacity(threads);
                    loop {
                        let at = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&p) = order.get(at) else {
                            break;
                        };
                        evs.clear();
                        evs.extend(ev_lists.iter().map(|t| t[p]));
                        cls.clear();
                        cls.extend(cl_lists.iter().map(|t| t[p]));
                        phase.partition_raw(&evs, &cls);
                    }
                    forwarded.lock().unwrap().push(phase.fwd.take().unwrap().finish());
                    barrier.wait();
                    let all = forwarded.lock().unwrap().clone();
                    let order = order2.get_or_init(|| {
                        let mut order: Vec<usize> = (0..1usize << bitsf).collect();
                        order.sort_by_key(|&p| std::cmp::Reverse(all.iter().map(|t| t[p].chunks).sum::<u32>()));
                        order
                    });
                    let mut runs: Vec<List<u64>> = Vec::with_capacity(threads);
                    loop {
                        let at = next2.fetch_add(1, Ordering::Relaxed);
                        let Some(&p) = order.get(at) else {
                            break;
                        };
                        runs.clear();
                        runs.extend(all.iter().map(|t| t[p]));
                        phase.partition(&runs);
                    }
                    let t0 = prof::start();
                    phase.out.finish();
                    prof::add(9, t0);
                    *cpu_total.lock().unwrap() += sys::thread_cpu();
                    (phase.agg, phase.out.rows_written, phase.check)
                })
            })
            .collect();
        for h in handles {
            let (part, rows, sums) = h.join().unwrap();
            check.0 += sums.0;
            check.1 += sums.1;
            for (a, b) in agg.iter_mut().zip(&part) {
                for k in 0..4 {
                    a[k] += b[k];
                }
            }
            total_rows += rows;
        }
    });
    timer.lap("person", std::mem::take(&mut *cpu_total.lock().unwrap()));

    let mut tables = Tables::default();
    for c in 0..=fast.ncamp {
        for d in 0..fast.ndays {
            let cell = agg[c * fast.ndays + d];
            if cell[0] == 0 && cell[3] == 0 {
                continue;
            }
            let (campaign, channel) = if c == fast.ncamp {
                (b"organic".to_vec(), b"organic".to_vec())
            } else {
                (lk.camp_ids[c].clone(), lk.camp_channels[c].clone())
            };
            tables.camp.push(CampRow {
                campaign,
                channel,
                day: (fast.ws_day + d as i64) as i32,
                sessions: cell[0] as i64,
                purchasing: cell[1] as i64,
                revenue: cell[2] as i64,
                assisted: cell[3] as i64,
            });
        }
    }
    if let Some(out) = out {
        write_camp_rows(&out.join("campaign_daily/part-0.parquet"), &tables.camp);
        if total_rows == 0 {
            crate::output::write_user_rows(&out.join("user_daily/part-0.parquet"), &[]);
        }
    }
    if let Some(handle) = unmapper {
        let _ = handle.join();
    }
    timer.lap("finish", 0.0);
    if cfg.timing {
        prof::report();
        let camp = tables.camp.iter().fold((0i64, 0i64), |a, r| (a.0 + r.sessions, a.1 + r.revenue));
        eprintln!(
            "[engine] self-check user_daily sessions {} revenue {} rows {}; campaign_daily sessions {} revenue {} rows {}; {}",
            check.0,
            check.1,
            total_rows,
            camp.0,
            camp.1,
            tables.camp.len(),
            if (check.0 as i64, check.1 as i64) == camp { "OK" } else { "MISMATCH" }
        );
    }
    tables.user = collected.into_inner().unwrap();
    Some(tables)
}

#[cfg(test)]
mod link_tests {
    use super::*;
    use std::collections::HashMap;

    fn rng(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(*state)
    }

    fn root(parent: &HashMap<u64, u64>, mut x: u64) -> u64 {
        while let Some(&p) = parent.get(&x) {
            if p == x {
                break;
            }
            x = p;
        }
        x
    }

    /// The link map (filter, staged lookup, far probes) against a plain
    /// union-find over a `HashMap`, for keys in the map and keys not in it.
    #[test]
    fn link_map_matches_union_find() {
        let shapes = [(0usize, 8u64), (1, 4), (7, 16), (50, 60), (3000, 4000), (3000, 500), (200_000, 300_000)];
        for (round, &(pairs_n, universe)) in shapes.iter().enumerate() {
            let mut state = round as u64 + 1;
            // Packed IDs are below 2^63 and never 0.
            let key = |state: &mut u64| 1 + (rng(state) % universe).wrapping_mul(0x0001_0000_0100_0001) % (1 << 62);
            let pairs: Vec<(u64, u64)> = (0..pairs_n).map(|_| (key(&mut state), key(&mut state))).collect();
            let mut parent: HashMap<u64, u64> = HashMap::new();
            for &(a, b) in &pairs {
                parent.entry(a).or_insert(a);
                parent.entry(b).or_insert(b);
                let (ra, rb) = (root(&parent, a), root(&parent, b));
                if ra != rb {
                    let (lo, hi) = if cmp_ids(ra, rb).is_lt() { (ra, rb) } else { (rb, ra) };
                    parent.insert(hi, lo);
                }
            }
            let map = LinkMap::build(&pairs);
            let mut probes: Vec<u64> = parent.keys().copied().collect();
            for _ in 0..(2 * pairs_n + 1000) {
                probes.push(key(&mut state));
                probes.push(1 + rng(&mut state) % (1 << 62));
            }
            probes.push(0);
            let mut renamed = 0;
            for block in probes.chunks(BLOCK) {
                let hashes: Vec<u64> = block.iter().map(|&k| map.stage1(k)).collect();
                let starts: Vec<usize> = hashes.iter().map(|&h| map.stage2(h)).collect();
                for (&k, &at) in block.iter().zip(&starts) {
                    let want = if parent.contains_key(&k) { root(&parent, k) } else { k };
                    assert_eq!(map.stage3(at, k), want, "round {round} key {k:#x}");
                    renamed += usize::from(want != k);
                }
            }
            assert!(pairs_n < 50 || renamed > pairs_n / 4, "round {round}: only {renamed} renamed keys probed");
        }
    }
}

#[cfg(test)]
mod long_line_tests {
    use super::*;

    fn rng(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(*state)
    }

    fn check<const N: usize>(line: &[u8]) {
        let mut want = Vec::new();
        let ok = split_line(line, N, &mut want);
        match split_long::<N>(line) {
            Some(got) => assert!(ok && got[..] == want[..], "{N} fields of {:?}", String::from_utf8_lossy(line)),
            None => assert!(!ok, "{N} fields of {:?}", String::from_utf8_lossy(line)),
        }
    }

    /// The slice splitter accepts exactly the lines the plain splitter
    /// accepts, with the same fields.
    #[test]
    fn split_long_matches_split_line() {
        let mut state = 17;
        let mut accepted = 0;
        for round in 0..200_000 {
            let n = (rng(&mut state) % 24) as usize;
            let alphabet: &[u8] = match round % 4 {
                0 => b"a,\"",
                1 => b"ab,,\"\r 1",
                2 => b"abcdefgh,",
                _ => b"abc\"\",,,",
            };
            let line: Vec<u8> = (0..n).map(|_| alphabet[(rng(&mut state) % alphabet.len() as u64) as usize]).collect();
            check::<2>(&line);
            check::<4>(&line);
            check::<7>(&line);
            accepted += usize::from(split_long::<4>(&line).is_some());
        }
        assert!(accepted > 2_000, "only {accepted} lines accepted");
        for line in [
            "", ",", "\"", "\"\"", "\"\",\"\"", "a,b", "\"a\",\"b\"", "\"a,b\"", "a\"b,c", "\"a\"b,c", ",,,,,,", ",,,,,,,",
            "a,b,c,d", "a,b,c,d,", "\"\"\"\",b", "a,\"", "\"a\",b\"",
        ] {
            check::<2>(line.as_bytes());
            check::<4>(line.as_bytes());
            check::<7>(line.as_bytes());
        }
        // A megabyte of commas is rejected at the first comma too many.
        assert!(split_long::<7>(&vec![b','; 1 << 20]).is_none());
    }
}
