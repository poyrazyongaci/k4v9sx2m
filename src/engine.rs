//! The fast engine. Three parallel phases, each a pool of threads pulling
//! jobs off an atomic counter:
//!
//! 1. parse: event and click chunks are scanned and parsed; every
//!    well-formed event copy becomes a 32-byte entry scattered by a hash of
//!    its event key. One job meanwhile reads all links and builds the
//!    user -> person map (union-find).
//! 2. resolve: per event partition, dedup on the exact event key keeping the
//!    smallest ingest, then map the survivors' users to persons and scatter
//!    16-byte records by person. Clicks get the same person scatter.
//! 3. person: per person partition, group by person, sort by ts, sessionize,
//!    attribute clicks, aggregate, and write user_daily row groups.

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

/// A valid click. `key` is the user key after parsing and the person key
/// after resolve. `tc` is `ts_rel << 32 | campaign`, ts relative to the
/// start of the click window.
#[repr(C)]
#[derive(Clone, Copy)]
struct Click {
    key: u64,
    ckey: u64,
    tc: u64,
}

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

/// A cheaper hash: one multiply, then the high half folded into the low.
/// The top bits are the top bits of the product.
#[inline(always)]
fn mix1(k: u64) -> u64 {
    let x = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
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
/// means the ID is its own person. On huge pages where the kernel gives
/// them.
///
/// The table is an array of buckets of one cache line: four keys, then
/// their four persons. A key lives in the bucket its hash names unless
/// that bucket was full when it came, in which case it went to the next
/// one with room. So a lookup reads one line, compares four keys at once,
/// and goes further only when the bucket is full and does not hold the
/// key. The table has room for two to four times its keys, so a bucket
/// holds one to two on average and under 1% of the keys overflow at 1.5 a
/// bucket, about 4% at 2.
///
/// Most IDs that are looked up are their own person's name and so are not
/// in the map at all. A small blocked Bloom filter (two bits in one word
/// per key, about a megabyte at 20M events) answers "not in the map"
/// from the cache, so only likely hits pay for a probe of the big table. A
/// filter miss is exact; a filter hit is always confirmed by the table.
#[derive(Clone, Copy)]
pub struct LinkMap {
    /// During the build of the node ids this memory is a plain table of
    /// (key, value) slots; the final map reads it as buckets.
    slots: *mut [u64; 2],
    /// Slots (node id table) or buckets (final map), minus one.
    mask: usize,
    /// Index of a bucket past the table that is always empty: where
    /// lookups the filter rejected are sent, so the probe needs no extra
    /// branch.
    nowhere: usize,
    filter: *mut u64,
    fmask: usize,
}
unsafe impl Send for LinkMap {}
unsafe impl Sync for LinkMap {}

/// Keys in a bucket of the link map: a 64-byte line of keys and persons.
const BUCKET: usize = 4;

impl LinkMap {
    /// A table of `cap` slots, or of `cap` buckets when `bucket` is set, and
    /// a filter sized for `keys` keys.
    fn with_slots(cap: usize, keys: usize, bucket: bool) -> LinkMap {
        // Fresh anonymous memory is zero, and 0 is the empty key. One extra
        // slot or bucket at the end stays empty for ever (`nowhere`).
        let unit = if bucket { BUCKET * 16 } else { 16 };
        let words = (keys / 8).next_power_of_two().clamp(64, 1 << 21);
        // The filter is small; put it on a 2 MiB boundary so one huge page
        // can back it.
        const HUGE: usize = 2 << 20;
        let raw = sys::anon(words * 8 + HUGE) as usize;
        let filter = ((raw + HUGE - 1) & !(HUGE - 1)) as *mut u64;
        LinkMap { slots: sys::anon((cap + 1) * unit) as *mut [u64; 2], mask: cap - 1, nowhere: cap, filter, fmask: words - 1 }
    }
    /// Bucket `b` of the final map: `BUCKET` keys, then their persons.
    #[inline(always)]
    fn bucket(&self, b: usize) -> *mut u64 {
        unsafe { self.slots.add(b * BUCKET) as *mut u64 }
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
    /// Where the lookup of a key with hash `h` starts: its home bucket if
    /// the filter says the key may be present, else the empty bucket.
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
    /// `stage2` reads the filter and asks for the bucket (or for the empty
    /// bucket), `stage3` reads the bucket and returns the person.
    #[inline(always)]
    fn stage1(&self, key: u64) -> u64 {
        let h = mix(key);
        prefetch(self.filter_word(h));
        h
    }
    #[inline(always)]
    fn stage2(&self, h: u64) -> usize {
        let i = self.start(h);
        prefetch(self.bucket(i));
        i
    }
    /// Which of the bucket's keys equal `key`, as a bit mask.
    #[inline(always)]
    unsafe fn matches(b: *const u64, key: u64) -> u32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            use core::arch::x86_64::*;
            let keys = _mm256_load_si256(b as *const __m256i);
            let eq = _mm256_cmpeq_epi64(keys, _mm256_set1_epi64x(key as i64));
            _mm256_movemask_pd(_mm256_castsi256_pd(eq)) as u32
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            let mut m = 0u32;
            for j in 0..BUCKET {
                m |= u32::from(*b.add(j) == key) << j;
            }
            m
        }
    }
    #[inline(always)]
    fn stage3(&self, i: usize, key: u64) -> u64 {
        unsafe {
            let b = self.bucket(i);
            let m = Self::matches(b, key);
            // Hit or miss is another coin toss. A full bucket without the
            // key is rare and goes out of line.
            let person = *b.add(BUCKET + (m.trailing_zeros() as usize & (BUCKET - 1)));
            let found = std::hint::select_unpredictable(m != 0, person, key);
            if (m == 0) & (*b.add(BUCKET - 1) != 0) {
                return self.person_far(key, i);
            }
            found
        }
    }
    /// The rest of the probe sequence after a full bucket without the key.
    #[inline(never)]
    fn person_far(&self, key: u64, first: usize) -> u64 {
        let mut i = (first + 1) & self.mask;
        loop {
            unsafe {
                let b = self.bucket(i);
                let m = Self::matches(b, key);
                if m != 0 {
                    return *b.add(BUCKET + m.trailing_zeros() as usize);
                }
                if *b.add(BUCKET - 1) == 0 {
                    return key;
                }
            }
            i = (i + 1) & self.mask;
        }
    }

    fn build(pairs: &[(u64, u64)]) -> LinkMap {
        // Room for twice the keys, so a bucket holds one to two on average.
        Self::build_with_room(pairs, 8)
    }

    /// `room` is the table's key capacity in quarters of the key count.
    fn build_with_room(pairs: &[(u64, u64)], room: usize) -> LinkMap {
        // Pass 1: a dense node id per distinct ID, and union-find over them.
        let cap = (pairs.len() * 4).next_power_of_two().max(16);
        let ids = LinkMap::with_slots(cap, 0, false);
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
        for id in 0..keys.len() as u32 {
            let root = find(&mut parent, id);
            if root != id {
                renamed.push([keys[id as usize], keys[root as usize]]);
            }
        }
        let buckets = (renamed.len() * room / 4).next_power_of_two().max(16) / BUCKET;
        let map = LinkMap::with_slots(buckets, renamed.len(), true);
        for block in renamed.chunks(16) {
            for pair in block {
                prefetch(map.bucket(map.home(pair[0])));
            }
            for pair in block {
                let h = mix(pair[0]);
                let mut i = map.home_of(h);
                unsafe {
                    // Buckets fill from the front and nothing is ever
                    // removed, so the last key tells whether one is full.
                    while *map.bucket(i).add(BUCKET - 1) != 0 {
                        i = (i + 1) & map.mask;
                    }
                    let b = map.bucket(i);
                    let mut j = 0;
                    while *b.add(j) != 0 {
                        j += 1;
                    }
                    *b.add(j) = pair[0];
                    *b.add(BUCKET + j) = pair[1];
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

struct Resolver<'a> {
    links: &'a LinkMap,
    table: Vec<u32>,
    /// The entry chunks of the partition at hand, by number from 1.
    bases: Vec<*mut Entry>,
    ev: Scatter<Ev>,
    cl: Scatter<Click>,
    shift2: u32,
    entry_chunk_bytes: usize,
}

/// Entries per chunk fit in this many bits of a dedup slot.
const OFF_BITS: u32 = 9;
const OFF_MASK: u32 = (1 << OFF_BITS) - 1;
const _: () = assert!((CHUNK_BYTES - 64) / std::mem::size_of::<Entry>() <= 1 << OFF_BITS);
/// Dedup is the first reader of entries the parser wrote with streaming
/// stores, so they come from memory, and the hardware prefetcher starts
/// over at every 4 KiB page and every chunk. Measured at 20M events:
/// 256, 512, 1024 and 2048 bytes ahead give 374, 367, 355 and 366 CPU ms
/// of dedup against 404 to 415 without.
const DEDUP_AHEAD: usize = 1024;
/// The same read-ahead for the second walk of the entries (person lookup).
const EMIT_AHEAD: usize = 1024;
/// And for the click blocks, which the parser wrote long before.
const CLICK_AHEAD: usize = 1024;
/// Link lookups run in blocks of this many records, one stage at a time.
const BLOCK: usize = 128;

impl<'a> Resolver<'a> {
    /// Dedup one event partition and scatter its surviving valid events by
    /// person.
    unsafe fn events(&mut self, lists: &[List<Entry>]) {
        let t = prof::start();
        let n: usize = lists.iter().map(|l| l.count()).sum();
        if n == 0 {
            return;
        }
        // Slots of 32 bits: a tag from the hash above the position of the
        // entry, `chunk << OFF_BITS | offset`, chunks counted from 1 so a
        // used slot is never 0. Half the bytes of a pointer table at the
        // same load. A partition has far fewer than 2^23 chunks (the
        // events are under 16 GB), so the position always fits; the tag
        // gets what is left and only ever filters: equality is decided by
        // the full key.
        let chunks: usize = lists.iter().map(|l| l.chunks as usize).sum();
        let pos_bits = usize::BITS - chunks.leading_zeros() + OFF_BITS;
        let pos_mask = if pos_bits >= 32 { u32::MAX } else { (1u32 << pos_bits) - 1 };
        let cap = (n * 2).next_power_of_two().max(64);
        self.table.clear();
        self.table.resize(cap, 0);
        self.bases.clear();
        self.bases.push(std::ptr::null_mut());
        let mask = cap - 1;
        let table = self.table.as_mut_ptr();
        let bases = &mut self.bases;
        for list in lists {
            list.for_each_chunk_ahead(|chunk, next| {
                let end = chunk.as_ptr().add(chunk.len()) as usize;
                let first = (bases.len() as u32) << OFF_BITS;
                bases.push(chunk.as_mut_ptr());
                let bases = bases.as_ptr();
                for (k, entry) in chunk.iter_mut().enumerate() {
                    // Ask for the entries a little ahead, into the next
                    // chunk at the end of this one.
                    let ahead = entry as *const Entry as usize + DEDUP_AHEAD;
                    let ahead = if ahead < end || next.is_null() { ahead } else { next as usize + (ahead - end) };
                    prefetch(ahead as *const u8);
                    let h = mix1(entry.ekey);
                    // Low hash bits, apart from the home bits (20 and up).
                    let tag = ((h as u32) << (pos_bits & 31)) & !pos_mask;
                    let me = first | k as u32;
                    let mut i = (h >> 20) as usize & mask;
                    loop {
                        let slot = *table.add(i);
                        if slot == 0 {
                            *table.add(i) = tag | me;
                            break;
                        }
                        if slot & !pos_mask == tag {
                            let at = slot & pos_mask;
                            let other = (*bases.add((at >> OFF_BITS) as usize)).add((at & OFF_MASK) as usize);
                            if (*other).ekey == entry.ekey {
                                // Same event: the smaller ingest stays. On
                                // a tie the copies are identical.
                                if entry.ingest < (*other).ingest {
                                    (*other).w3 = DEAD;
                                    *table.add(i) = tag | me;
                                } else {
                                    entry.w3 = DEAD;
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
        // The link map is far bigger than the cache. Lookups go a block at
        // a time, each stage for the whole block, so the misses of a block
        // are all in flight together and every loop stays small.
        let links = *self.links;
        let shift2 = self.shift2;
        let ev = &mut self.ev;
        let mut hashes = [0u64; BLOCK];
        let mut starts = [0usize; BLOCK];
        for list in lists {
            list.for_each_chunk_ahead(|chunk, next| {
                // The entries left the cache during the dedup walk: read
                // ahead again, across chunk ends.
                let end = chunk.as_ptr().add(chunk.len()) as usize;
                for block in chunk.chunks(BLOCK) {
                    for (h, entry) in hashes.iter_mut().zip(block) {
                        let ahead = entry as *const Entry as usize + EMIT_AHEAD;
                        let ahead = if ahead < end || next.is_null() { ahead } else { next as usize + (ahead - end) };
                        prefetch(ahead as *const u8);
                        *h = links.stage1(entry.ukey);
                    }
                    for (at, &h) in starts.iter_mut().zip(&hashes[..block.len()]) {
                        *at = links.stage2(h);
                    }
                    for (&at, entry) in starts.iter().zip(block) {
                        if entry.w3 != DEAD {
                            let pkey = links.stage3(at, entry.ukey);
                            ev.push((mix1(pkey) >> shift2) as usize, Ev { pkey, v: entry.w3 });
                        }
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
        let links = *self.links;
        let shift2 = self.shift2;
        let cl = &mut self.cl;
        let mut hashes = [0u64; BLOCK];
        let mut starts = [0usize; BLOCK];
        for part in block.chunks(BLOCK) {
            for (h, click) in hashes.iter_mut().zip(part) {
                // A prefetch past the end of the block is harmless.
                prefetch((click as *const Click as *const u8).wrapping_add(CLICK_AHEAD));
                *h = links.stage1(click.key);
            }
            for (at, &h) in starts.iter_mut().zip(&hashes[..part.len()]) {
                *at = links.stage2(h);
            }
            for (&at, click) in starts.iter().zip(part) {
                let pkey = links.stage3(at, click.key);
                cl.push((mix1(pkey) >> shift2) as usize, Click { key: pkey, ckey: click.ckey, tc: click.tc });
            }
        }
        prof::add(6, t);
    }
}

// ---------------------------------------------------------------- phase 3

struct PersonPhase<'a> {
    fast: &'a Fast<'a>,
    /// Person key -> local id: an open-addressing table of `id + 1`, 0 for
    /// an empty slot, and the keys by `id + 1`. `keys[0]` is 0, which is no
    /// person, so one compare tells a hit from an empty or foreign slot.
    tab: Vec<u32>,
    keys: Vec<u64>,
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
    cl: Vec<(u64, u64)>,
    /// Per campaign, for the person at hand: how many of its clicks are in
    /// reach of the current session, and the session and day slot at which
    /// it came into reach.
    reach: Vec<[u32; 4]>,
    /// The person's days so far, in order: the day, and the index of the
    /// first session of that day. One slot past the last holds `u32::MAX`.
    slot_day: Vec<u32>,
    slot_first: Vec<u32>,
    /// The sessions of the person at hand, when the person has clicks: the
    /// start on the click clock, the day and the revenue.
    s_at: Vec<u32>,
    s_day: Vec<u32>,
    s_revenue: Vec<u64>,
    /// Per campaign: the sessions of the person it assists, one bit each
    /// (people with few sessions only).
    cover: Vec<u64>,
    catsum: Vec<u64>,
    touched: Vec<u8>,
    /// (campaign, day) -> [sessions, purchasing, revenue, 0]
    agg: Vec<[u64; 4]>,
    /// (campaign, day) -> assisted sessions. Apart from `agg` and narrow:
    /// these cells are hit at random, far more often than the others, and
    /// this way they all stay in the cache.
    assists: Vec<u32>,
    out: UserOut<'a>,
    /// Summed sessions and revenue of the rows emitted, for the self-check.
    check: (u64, u64),
}

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

/// The hash of a person key inside one partition: the top bits of one
/// multiply. The partition was chosen by `mix`, which shares no constant
/// with this.
#[inline(always)]
fn person_hash(key: u64) -> u64 {
    key.wrapping_mul(0xD6E8_FEB8_6659_FD93)
}

/// Sessions are compared with a click eight at a time.
const LANES: usize = 8;
/// The most sessions a person can have for their clicks to be attributed
/// without sorting them (`PersonPhase::few_sessions`).
const FEW: usize = 32;

/// One click against a block of `LANES` sessions. `at` holds their starts,
/// `best_t` and `best_c` the time plus one and the campaign of the latest
/// click in reach of each so far. The click is at `t`, of campaign `camp`,
/// and reaches the starts in `t..=upto`. All times are under 2^31.
///
/// Returns the sessions of the block the click reaches, one bit each, and
/// whether the click ties with the latest click of a session while being of
/// another campaign.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn click_block(at: *const u32, best_t: *mut u32, best_c: *mut u32, t: u32, upto: u32, camp: u32) -> (u64, bool) {
    use core::arch::x86_64::*;
    let starts = _mm256_loadu_si256(at as *const __m256i);
    let before = _mm256_cmpgt_epi32(_mm256_set1_epi32(t as i32), starts);
    let after = _mm256_cmpgt_epi32(starts, _mm256_set1_epi32(upto as i32));
    let out = _mm256_or_si256(before, after);
    let key = _mm256_set1_epi32(t as i32 + 1);
    let camps = _mm256_set1_epi32(camp as i32);
    let old_t = _mm256_loadu_si256(best_t as *const __m256i);
    let old_c = _mm256_loadu_si256(best_c as *const __m256i);
    let later = _mm256_andnot_si256(out, _mm256_cmpgt_epi32(key, old_t));
    let same = _mm256_andnot_si256(out, _mm256_cmpeq_epi32(key, old_t));
    let tie = _mm256_andnot_si256(_mm256_cmpeq_epi32(old_c, camps), same);
    _mm256_storeu_si256(best_t as *mut __m256i, _mm256_blendv_epi8(old_t, key, later));
    _mm256_storeu_si256(best_c as *mut __m256i, _mm256_blendv_epi8(old_c, camps, later));
    let reached = !_mm256_movemask_ps(_mm256_castsi256_ps(out)) as u32 & 0xFF;
    (u64::from(reached), _mm256_testz_si256(tie, tie) == 0)
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
unsafe fn click_block(at: *const u32, best_t: *mut u32, best_c: *mut u32, t: u32, upto: u32, camp: u32) -> (u64, bool) {
    let key = t + 1;
    let (mut reached, mut tie) = (0u64, false);
    for k in 0..LANES {
        let start = *at.add(k);
        let covers = (start >= t) & (start <= upto);
        reached |= u64::from(covers) << k;
        let (old_t, old_c) = (*best_t.add(k), *best_c.add(k));
        let later = covers & (key > old_t);
        tie |= covers & (key == old_t) & (old_c != camp);
        *best_t.add(k) = if later { key } else { old_t };
        *best_c.add(k) = if later { camp } else { old_c };
    }
    (reached, tie)
}

/// The run of events in the bin of `ev[i]` that starts at `i`, up to `n`:
/// its smallest and largest time and its end. `ev` is readable for
/// `LANES` values past `n`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn bin_run(ev: *const u32, i: usize, n: usize) -> (u32, u32, usize) {
    use core::arch::x86_64::*;
    let bin = _mm256_set1_epi32((*ev.add(i) >> BIN_SHIFT) as i32);
    let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let (mut lo, mut hi) = (_mm256_set1_epi32(-1), _mm256_setzero_si256());
    let mut j = i;
    loop {
        // A run of `LANES` goes on in the next block: rare, so a branch.
        let v = _mm256_loadu_si256(ev.add(j) as *const __m256i);
        let same = _mm256_cmpeq_epi32(_mm256_srli_epi32(v, BIN_SHIFT as i32), bin);
        let m = _bzhi_u32(_mm256_movemask_ps(_mm256_castsi256_ps(same)) as u32, (n - j).min(LANES) as u32);
        let len = (!m).trailing_zeros();
        let take = _mm256_cmpgt_epi32(_mm256_set1_epi32(len as i32), lane);
        lo = _mm256_min_epu32(lo, _mm256_or_si256(v, _mm256_xor_si256(take, _mm256_set1_epi32(-1))));
        hi = _mm256_max_epu32(hi, _mm256_and_si256(v, take));
        j += len as usize;
        if len < LANES as u32 {
            break;
        }
    }
    let lo = _mm256_min_epu32(lo, _mm256_permute2x128_si256(lo, lo, 1));
    let lo = _mm256_min_epu32(lo, _mm256_shuffle_epi32(lo, 0b01_00_11_10));
    let lo = _mm256_min_epu32(lo, _mm256_shuffle_epi32(lo, 0b10_11_00_01));
    let hi = _mm256_max_epu32(hi, _mm256_permute2x128_si256(hi, hi, 1));
    let hi = _mm256_max_epu32(hi, _mm256_shuffle_epi32(hi, 0b01_00_11_10));
    let hi = _mm256_max_epu32(hi, _mm256_shuffle_epi32(hi, 0b10_11_00_01));
    (_mm256_cvtsi256_si32(lo) as u32, _mm256_cvtsi256_si32(hi) as u32, j)
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2")))]
#[inline(always)]
unsafe fn bin_run(ev: *const u32, i: usize, n: usize) -> (u32, u32, usize) {
    let bin = *ev.add(i) >> BIN_SHIFT;
    let (mut first, mut latest) = (u32::MAX, 0u32);
    let mut j = i;
    while j < n {
        let t = *ev.add(j);
        if t >> BIN_SHIFT != bin {
            break;
        }
        first = first.min(t);
        latest = latest.max(t);
        j += 1;
    }
    (first, latest, j)
}

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
    fn new(fast: &'a Fast<'a>, out: UserOut<'a>) -> Self {
        PersonPhase {
            fast,
            tab: Vec::new(),
            keys: Vec::new(),
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
            reach: vec![[0; 4]; fast.ncamp],
            slot_day: vec![0; fast.ndays + 2],
            slot_first: vec![0; fast.ndays + 2],
            s_at: Vec::new(),
            s_day: Vec::new(),
            s_revenue: Vec::new(),
            cover: vec![0; fast.ncamp],
            catsum: vec![0; 256],
            touched: Vec::new(),
            agg: vec![[0; 4]; (fast.ncamp + 1) * fast.ndays],
            assists: vec![0; (fast.ncamp + 1) * fast.ndays],
            out,
            check: (0, 0),
        }
    }

    /// Double the table and put the `people` keys back in. Returns the new
    /// shift of the hash.
    #[cold]
    #[inline(never)]
    unsafe fn grow_table(tab: &mut Vec<u32>, keys: *const u64, people: usize) -> u32 {
        let cap = tab.len() * 2;
        tab.clear();
        tab.resize(cap, 0);
        let shift = 64 - cap.trailing_zeros();
        for id1 in 1..=people {
            let mut i = (person_hash(*keys.add(id1)) >> shift) as usize;
            while tab[i] != 0 {
                i = (i + 1) & (cap - 1);
            }
            tab[i] = id1 as u32;
        }
        shift
    }

    unsafe fn partition(&mut self, evs: &[List<Ev>], cls: &[List<Click>]) {
        let t = prof::start();
        let n: usize = evs.iter().map(|l| l.count()).sum();
        if n == 0 {
            return;
        }
        let cap = (n / 8).next_power_of_two().max(256);
        self.tab.clear();
        self.tab.resize(cap, 0);
        // At most one person per event. The vectors are filled through
        // pointers, so the loop below reloads nothing.
        self.keys.clear();
        self.keys.reserve(n + 1);
        self.counts.clear();
        self.counts.reserve(n);
        let keys = self.keys.as_mut_ptr();
        let counts = self.counts.as_mut_ptr();
        *keys = 0;
        let mut tab = self.tab.as_mut_ptr();
        let mut mask = cap - 1;
        let mut shift = 64 - cap.trailing_zeros();
        let mut people = 0usize;
        // `ts_rel < wlen`, so the last bin is that of `wlen - 1`.
        let nbins = ((self.fast.wlen - 1) >> BIN_SHIFT) as usize + 1;
        self.bins.clear();
        self.bins.resize(nbins, 0);
        self.words.clear();
        self.words.reserve(n);
        self.buys.clear();
        let words = self.words.as_mut_ptr();
        let bins = self.bins.as_mut_ptr();
        let mut at = 0usize;

        // Pass 1: a local id per person, how many events each person has
        // and each time bin has, and one word per event.
        for list in evs {
            list.for_each_chunk(|chunk| {
                for rec in chunk.iter() {
                    let mut i = (person_hash(rec.pkey) >> shift) as usize;
                    let id = loop {
                        let id1 = *tab.add(i) as usize;
                        if *keys.add(id1) == rec.pkey {
                            break id1 as u32 - 1;
                        }
                        if id1 == 0 {
                            if (people + 1) * 2 > mask + 1 {
                                shift = Self::grow_table(&mut self.tab, keys, people);
                                tab = self.tab.as_mut_ptr();
                                mask = self.tab.len() - 1;
                                i = (person_hash(rec.pkey) >> shift) as usize;
                                continue;
                            }
                            *counts.add(people) = 0;
                            people += 1;
                            *keys.add(people) = rec.pkey;
                            *tab.add(i) = people as u32;
                            break people as u32 - 1;
                        }
                        i = (i + 1) & mask;
                    };
                    *counts.add(id as usize) += 1;
                    *bins.add((rec.v >> (32 + BIN_SHIFT)) as usize) += 1;
                    *words.add(at) = (rec.v & !0xFFFF_FFFF) | u64::from(id);
                    at += 1;
                    if rec.v as u32 != 0 {
                        self.buys.push((rec.v, id));
                    }
                }
            });
        }
        self.words.set_len(n);
        self.keys.set_len(people + 1);
        self.counts.set_len(people);
        // Pass 2: the whole partition into bin order.
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
        // Pass 3: a stable scatter by person, so each person's events are
        // in bin order.
        // counts -> start offsets (counts[i] becomes the write cursor).
        let mut sum = 0u32;
        for c in self.counts.iter_mut() {
            let k = *c;
            *c = sum;
            sum += k;
        }
        self.ev.clear();
        // Padding: the run finder reads a block of `LANES` past any start.
        self.ev.reserve(n + LANES);
        self.ev.set_len(n + LANES);
        self.ev[n..].fill(0);
        for &word in &self.binned {
            let w = self.counts.get_unchecked_mut(word as u32 as usize);
            *self.ev.get_unchecked_mut(*w as usize) = (word >> 32) as u32;
            *w += 1;
        }
        // counts[i] is now the end offset of person i.

        // The purchases, person after person.
        self.bcounts.clear();
        self.bcounts.resize(people, 0);
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

        // Clicks of people that have events here; the rest cannot matter.
        let nc: usize = cls.iter().map(|l| l.count()).sum();
        self.ccounts.clear();
        self.ccounts.resize(people + 1, 0);
        self.cl.clear();
        if nc > 0 {
            self.cids.clear();
            self.cids.reserve(nc);
            for list in cls {
                list.for_each_chunk(|chunk| {
                    for rec in chunk.iter() {
                        let mut i = (person_hash(rec.key) >> shift) as usize;
                        let id = loop {
                            let id1 = *tab.add(i) as usize;
                            if *keys.add(id1) == rec.key {
                                break id1 as u32 - 1;
                            }
                            if id1 == 0 {
                                break u32::MAX;
                            }
                            i = (i + 1) & mask;
                        };
                        self.cids.push(id);
                        if id != u32::MAX {
                            *self.ccounts.get_unchecked_mut(id as usize) += 1;
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
                            *self.cl.get_unchecked_mut(*w as usize) = (rec.tc, rec.ckey);
                            *w += 1;
                        }
                    }
                });
            }
        }

        let t = prof::add(7, t);
        let mut ev_start = 0usize;
        let mut cl_start = 0usize;
        let mut buy_start = 0usize;
        for p in 0..people {
            let ev_end = self.counts[p] as usize;
            let cl_end = if nc > 0 { self.ccounts[p] as usize } else { 0 };
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
        let pkey = self.keys[p + 1];
        let ev = std::slice::from_raw_parts(self.ev.as_ptr().add(ev_start), ev_end - ev_start);
        let cl = std::slice::from_raw_parts_mut(self.cl.as_mut_ptr().add(cl_start), cl_end - cl_start);
        if buy_end - buy_start > 1 {
            // The time is the top half of a value.
            sort_u64(std::slice::from_raw_parts_mut(self.bought.as_mut_ptr().add(buy_start), buy_end - buy_start));
        }
        let mut buy = buy_start;
        let ws_in_day = self.fast.ws_in_day;
        let ndays = self.fast.ndays;
        let organic = self.fast.ncamp;
        let gap = SESSION_GAP_MS as u32;
        let n = ev.len();
        let clicked = !cl.is_empty();
        if clicked {
            // Room for every session and a block of padding.
            for v in [&mut self.s_at, &mut self.s_day] {
                if v.capacity() < n + LANES {
                    v.reserve(n + LANES - v.len());
                }
            }
            if self.s_revenue.capacity() < n {
                self.s_revenue.reserve(n - self.s_revenue.len());
            }
        }
        // Sessions opened and day slots filled so far.
        let (mut opened, mut slots) = (0u32, 0usize);
        let mut cur_day = u32::MAX;
        let (mut d_sessions, mut d_events, mut d_revenue) = (0u32, 0u32, 0u64);
        // The open session: its start and the largest time seen in it.
        let (mut start, mut last) = (0u32, 0u32);
        let mut i = 0usize;
        loop {
            // One run of events in the same bin, and its extremes.
            let (first, latest, j) = if i < n { bin_run(ev.as_ptr(), i, n) } else { (u32::MAX, 0, i) };
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
                    if clicked {
                        // Click times are relative to window_start - 7
                        // days, so a session start S is S + reach on that
                        // clock.
                        let at = opened as usize - 1;
                        *self.s_at.as_mut_ptr().add(at) = start + CLICK_REACH_MS as u32;
                        *self.s_day.as_mut_ptr().add(at) = cur_day;
                        *self.s_revenue.as_mut_ptr().add(at) = revenue;
                    } else {
                        let cell = self.agg.get_unchecked_mut(organic * ndays + cur_day as usize);
                        cell[0] += 1;
                        cell[1] += u64::from(revenue > 0);
                        cell[2] += revenue;
                    }
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
                    *self.slot_day.get_unchecked_mut(slots) = day_rel;
                    *self.slot_first.get_unchecked_mut(slots) = opened;
                    slots += 1;
                    *self.slot_first.get_unchecked_mut(slots) = u32::MAX;
                }
                d_sessions += 1;
                opened += 1;
            }
            last = latest;
            d_events += (j - i) as u32;
            i = j;
        }
        if cur_day != u32::MAX {
            self.emit(pkey, cur_day, d_sessions, d_events, d_revenue);
        }
        if clicked {
            let sessions = opened as usize;
            // The few-sessions path compares times as signed 32-bit.
            let narrow = self.fast.cwlen < (1 << 31) - 1;
            if sessions > FEW || !narrow || !self.few_sessions(cl, sessions, slots) {
                self.many_sessions(cl, sessions);
            }
        }
    }

    /// Attribute the clicks of a person with at most `FEW` sessions, the
    /// clicks in any order. False, with nothing done, if two clicks of
    /// different campaigns tie for a session: then the click ids decide and
    /// the caller takes the sorted path.
    ///
    /// A click at `t` is in reach of the sessions that start in `t..=t +
    /// reach`. Each session keeps the latest click in reach, and each
    /// campaign the set of sessions its clicks reach.
    #[inline(always)]
    unsafe fn few_sessions(&mut self, cl: &[(u64, u64)], sessions: usize, slots: usize) -> bool {
        let ndays = self.fast.ndays;
        let reach = CLICK_REACH_MS as u64;
        // Whole blocks: past the last session come starts that no click
        // reaches. A real start is under `LAST - 1` (the caller checked).
        const LAST: u32 = i32::MAX as u32;
        let lanes = sessions.div_ceil(LANES) * LANES;
        let at = self.s_at.as_mut_ptr();
        for pad in sessions..lanes {
            *at.add(pad) = LAST;
        }
        // Per session: the time of the latest click in reach, plus one, and
        // its campaign.
        let mut best_t = [0u32; FEW];
        let mut best_c = [self.fast.ncamp as u32; FEW];
        let mut tie = false;
        for click in cl.iter() {
            let t = (click.0 >> 32) as u32;
            let camp = click.0 as u32;
            let upto = (u64::from(t) + reach).min(u64::from(LAST) - 1) as u32;
            let mut reached = 0u64;
            let mut lane = 0;
            while lane < lanes {
                let (bits, tied) = click_block(at.add(lane), best_t.as_mut_ptr().add(lane), best_c.as_mut_ptr().add(lane), t, upto, camp);
                reached |= bits << lane;
                tie |= tied;
                lane += LANES;
            }
            *self.cover.get_unchecked_mut(camp as usize) |= reached;
        }
        if tie {
            for click in cl.iter() {
                *self.cover.get_unchecked_mut(click.0 as u32 as usize) = 0;
            }
            return false;
        }
        // Assists: per campaign and day, the sessions of that day it covers.
        let mut day_bits = [0u64; FEW + 1];
        for k in 0..slots {
            let from = *self.slot_first.get_unchecked(k);
            let to = (*self.slot_first.get_unchecked(k + 1)).min(sessions as u32);
            *day_bits.get_unchecked_mut(k) = ((1u64 << to) - 1) & !((1u64 << from) - 1);
        }
        for click in cl.iter() {
            let c = click.0 as u32 as usize;
            let bits = *self.cover.get_unchecked(c);
            if bits != 0 {
                *self.cover.get_unchecked_mut(c) = 0;
                let row = c * ndays;
                for k in 0..slots {
                    let count = (bits & *day_bits.get_unchecked(k)).count_ones();
                    *self.assists.get_unchecked_mut(row + *self.slot_day.get_unchecked(k) as usize) += count;
                }
            }
        }
        for k in 0..sessions {
            let revenue = *self.s_revenue.as_ptr().add(k);
            let cell = self.agg.get_unchecked_mut(*best_c.get_unchecked(k) as usize * ndays + *self.s_day.as_ptr().add(k) as usize);
            cell[0] += 1;
            cell[1] += u64::from(revenue > 0);
            cell[2] += revenue;
        }
        true
    }

    /// Attribute the clicks of a person with any number of sessions: sort
    /// the clicks and slide over them, session by session.
    #[inline(never)]
    unsafe fn many_sessions(&mut self, cl: &mut [(u64, u64)], sessions: usize) {
        if cl.len() > 1 {
            // By ts, then by click id descending: the last click at or
            // before a session start is the latest with the smallest id.
            cl.sort_unstable_by(|a, b| (a.0 >> 32).cmp(&(b.0 >> 32)).then_with(|| cmp_ids(b.1, a.1)));
        }
        let mut win = Window { lo: 0, hi: 0, session: 0, slot: 0 };
        for k in 0..sessions {
            win.session = k as u32;
            win.slot += u32::from(*self.slot_first.get_unchecked(win.slot as usize + 1) == k as u32);
            let (at, day, revenue) = (*self.s_at.as_ptr().add(k), *self.s_day.as_ptr().add(k), *self.s_revenue.as_ptr().add(k));
            self.credit(cl, &mut win, u64::from(at), day as usize, revenue);
        }
        // The campaigns still in reach after the last session assisted up
        // to it.
        for click in cl.get_unchecked(win.lo..win.hi) {
            let c = click.0 as u32 as usize;
            let state = *self.reach.get_unchecked(c);
            if state[0] != 0 {
                self.reach.get_unchecked_mut(c)[0] = 0;
                self.assisted(c, state[1], state[2], sessions as u32);
            }
        }
    }

    /// Credit one session to a click. `cl` is the person's clicks in
    /// order, `win` the range of them in reach of the previous session, `at`
    /// the start of the session on the click clock; sessions come in order
    /// of their start.
    ///
    /// Assists are not counted session by session. A campaign assists every
    /// session from the one at which its count of clicks in reach leaves
    /// zero to the one at which it returns to zero, so that stretch of
    /// sessions is counted when it ends, once per day it touches.
    #[inline(always)]
    unsafe fn credit(&mut self, cl: &[(u64, u64)], win: &mut Window, at: u64, day: usize, revenue: u64) {
        let ndays = self.fast.ndays;
        let mut credit = self.fast.ncamp;
        {
            let (mut lo, mut hi) = (win.lo, win.hi);
            let start = at - CLICK_REACH_MS as u64;
            while hi < cl.len() && (cl.get_unchecked(hi).0 >> 32) <= at {
                let c = cl.get_unchecked(hi).0 as u32 as usize;
                let state = self.reach.get_unchecked_mut(c);
                let fresh = state[0] == 0;
                state[1] = if fresh { win.session } else { state[1] };
                state[2] = if fresh { win.slot } else { state[2] };
                state[0] += 1;
                hi += 1;
            }
            while lo < hi && (cl.get_unchecked(lo).0 >> 32) < start {
                let c = cl.get_unchecked(lo).0 as u32 as usize;
                let state = self.reach.get_unchecked_mut(c);
                state[0] -= 1;
                if state[0] == 0 {
                    let (from, slot) = (state[1], state[2]);
                    self.assisted(c, from, slot, win.session);
                }
                lo += 1;
            }
            if hi > lo {
                credit = cl.get_unchecked(hi - 1).0 as u32 as usize;
            }
            win.lo = lo;
            win.hi = hi;
        }
        let cell = self.agg.get_unchecked_mut(credit * ndays + day);
        cell[0] += 1;
        cell[1] += u64::from(revenue > 0);
        cell[2] += revenue;
    }

    /// Campaign `c` assisted the person's sessions `from..to`; `slot` is the
    /// day slot of session `from`. Sessions of a day are consecutive, so
    /// each day gets the length of its overlap with the stretch.
    #[inline(always)]
    unsafe fn assisted(&mut self, c: usize, from: u32, slot: u32, to: u32) {
        let row = c * self.fast.ndays;
        let mut k = slot as usize;
        loop {
            let a = from.max(*self.slot_first.get_unchecked(k));
            let next = *self.slot_first.get_unchecked(k + 1);
            let b = to.min(next);
            *self.assists.get_unchecked_mut(row + *self.slot_day.get_unchecked(k) as usize) += b - a;
            k += 1;
            if next >= to {
                break;
            }
        }
    }

    /// The campaign cells of this thread, assists folded in.
    fn cells(&mut self) -> Vec<[u64; 4]> {
        for (cell, &assists) in self.agg.iter_mut().zip(&self.assists) {
            cell[3] = u64::from(assists);
        }
        std::mem::take(&mut self.agg)
    }
}

/// The clicks in reach of the session being credited, and where that
/// session stands among the person's sessions and days.
struct Window {
    lo: usize,
    hi: usize,
    session: u32,
    slot: u32,
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
                        links,
                        table: Vec::new(),
                        bases: Vec::new(),
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
    let mut order: Vec<usize> = (0..parts2).collect();
    let weight = |p: usize| -> u32 { ev_lists.iter().map(|t| t[p].chunks).sum() };
    order.sort_by_key(|&p| std::cmp::Reverse(weight(p)));
    let next = AtomicUsize::new(0);
    let ev_lists = &ev_lists;
    let cl_lists = &cl_lists;
    let order = &order;
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
                    let mut phase = PersonPhase::new(fast, user_out);
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
                        phase.partition(&evs, &cls);
                    }
                    let t0 = prof::start();
                    phase.out.finish();
                    prof::add(9, t0);
                    *cpu_total.lock().unwrap() += sys::thread_cpu();
                    (phase.cells(), phase.out.rows_written, phase.check)
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
            // The table as the engine builds it (at most half full), and a
            // denser one (up to 80% full) where far more buckets overflow.
            for room in [8, 5] {
                let map = LinkMap::build_with_room(&pairs, room);
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
                        assert_eq!(map.stage3(at, k), want, "round {round} room {room} key {k:#x}");
                        renamed += usize::from(want != k);
                    }
                }
                assert!(pairs_n < 50 || renamed > pairs_n / 4, "round {round}: only {renamed} renamed keys probed");
                // Keys that are not in their own bucket were found above by
                // the out-of-line probe; make sure the big round has some.
                let far = parent
                    .keys()
                    .filter(|&&k| root(&parent, k) != k && unsafe { LinkMap::matches(map.bucket(map.home(k)), k) } == 0)
                    .count();
                assert!(pairs_n < 100_000 || far > 20, "round {round} room {room}: only {far} keys overflowed");
            }
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
