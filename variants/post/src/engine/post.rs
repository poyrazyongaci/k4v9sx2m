//! The post shape (`ENGINE_POST=1`, off by default): everything after the
//! parse on dense integer IDs.
//!
//! The generator writes an ID as a prefix letter and the base 36 of a keyed
//! 40-bit bijection of a number (`decode.rs`). Event numbers are a dense
//! counter, and a user number is `person * 512 + alias`. The keys are
//! recovered from a sample; if either is not found, or a bound fails, this
//! shape declines before it changes anything and the three-phase shape
//! runs on the same parsed entries.
//!
//! 1. Convert: each entry's event key is decoded to its number and the
//!    entry goes to the range partition of that number (`R` numbers each).
//!    Non-canonical keys and numbers past the last range go to a general
//!    list. Each click's user is resolved to a dense person (below) and the
//!    click goes to the bucket of that person.
//! 2. Dedup and walk: per range, the copies land in a table indexed by
//!    number, keeping the smallest ingest. The table is then walked in
//!    number order. Survivors come out grouped by person (the generator
//!    numbers a person's events consecutively), so a person's events form
//!    runs; each run is counted on its person.
//! 3. Persons: a person with one run is processed straight from it; the
//!    others (a person spanning ranges, persons merged by links) gather
//!    their runs. A run that is not in time order is sorted.
//!
//! Identity. A person index `P` has a descriptor: 0 when none of its IDs is
//! linked (then alias 0 is person `P`, named by itself); else the dense
//! person of its home component (the component of its smallest linked
//! alias) and the set of its aliases below 31 in that component. Every
//! other linked ID is in an exact map to its dense person. An ID that is
//! neither is a person on its own and takes the general path, keyed by the
//! ID itself. Dense persons are named by the smallest ID of their
//! component (the link map's root).

use super::*;
use crate::decode;
use std::collections::HashMap;
use std::sync::atomic::AtomicU32;
use std::sync::Barrier;

/// Numbers per range partition.
const RBITS: u32 = 15;
const R: usize = 1 << RBITS;
/// Dense persons per click bucket.
const CB: u32 = 12;
const CLOCAL: u64 = (1 << CB) - 1;
/// Where the local person index sits in a click's `tc` (above the campaign).
const CSHIFT: u32 = 16;
/// No dense person: the general path.
const GEN: u32 = u32::MAX;
/// Aliases below this are in the descriptor's set; bit 31 marks a linked P.
const MASK_ALIASES: u64 = 31;
const LINKED: u64 = 1 << 31;

/// The 8 7-bit fields of the low 56 bits of `k`, one a byte, the first
/// (bits 49..56) in the top byte.
#[inline(always)]
fn spread(k: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        core::arch::x86_64::_pdep_u64(k, 0x7F7F_7F7F_7F7F_7F7F)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let mut x = 0u64;
        for i in 0..8 {
            x |= ((k >> (7 * i)) & 0x7F) << (8 * i);
        }
        x
    }
}

const ONES: u64 = 0x0101_0101_0101_0101;
const HIGH: u64 = 0x8080_8080_8080_8080;

/// The base-36 value of a packed key (see `ids.rs`) whose ID is `prefix`
/// then 1 to 8 lowercase base-36 digits with no leading zero and a value
/// below 2^40; `None` for any other key. One to one on what it accepts,
/// like `decode::decode`.
#[inline(always)]
pub(super) fn packed_y(k: u64, prefix: u8) -> Option<u64> {
    // Bit 63 (an interned key) or another first byte.
    if k >> 56 != u64::from(prefix) {
        return None;
    }
    let x = spread(k);
    if x == 0 {
        return None;
    }
    // Padding: the zero bytes below the last digit.
    let tz = x.trailing_zeros() & 56;
    let pad = (1u64 << tz) - 1;
    let y = x | (pad & (0x30 * ONES));
    // Every byte a digit or a lowercase letter (bytes are below 0x80, so
    // no add carries into the next byte).
    let digit = y.wrapping_add(0x50 * ONES) & !y.wrapping_add(0x46 * ONES);
    let letter = y.wrapping_add(0x1F * ONES) & !y.wrapping_add(0x05 * ONES);
    if (digit | letter) & HIGH != HIGH {
        return None;
    }
    // A leading zero, unless the ID is the one digit `0`.
    if x >> 56 == 0x30 && tz < 56 {
        return None;
    }
    let v = (y - 0x30 * ONES) - ((letter & HIGH) >> 7) * 39;
    let v = v >> tz;
    let t = (v & 0x00FF_00FF_00FF_00FF) + ((v >> 8) & 0x00FF_00FF_00FF_00FF) * 36;
    let t = (t & 0x0000_FFFF_0000_FFFF) + ((t >> 16) & 0x0000_FFFF_0000_FFFF) * 1296;
    let value = (t & 0xFFFF_FFFF) + (t >> 32) * 1_679_616;
    (value <= decode::MASK).then_some(value)
}

/// Dense identity of user keys (see the top of this file).
struct Ident {
    desc: Vec<u64>,
    names: Vec<u64>,
    exc: HashMap<u64, u32, Fx>,
    key: u64,
}

impl Ident {
    /// The dense person of user key `k` and the person's name, or `GEN`
    /// and `k` for a person of one ID that is not dense.
    #[inline(always)]
    fn resolve(&self, k: u64) -> (u32, u64) {
        if let Some(y) = packed_y(k, b'u') {
            let u = decode::unscramble(y, self.key);
            let (p, a) = ((u >> 9) as usize, u & 511);
            if let Some(&d) = self.desc.get(p) {
                if d == 0 {
                    // No alias of P is linked.
                    return if a == 0 { (p as u32, k) } else { (GEN, k) };
                }
                if a < MASK_ALIASES && (d >> a) & 1 != 0 {
                    let c = (d >> 32) as u32;
                    return (c, self.names[c as usize]);
                }
            }
        }
        match self.exc.get(&k) {
            Some(&c) => (c, self.names[c as usize]),
            None => (GEN, k),
        }
    }
}

/// A small direct-mapped cache in front of `Ident::resolve`.
struct Cache {
    keys: [u64; 64],
    vals: [(u32, u64); 64],
}

impl Cache {
    fn new() -> Self {
        // Key 0 is no ID (an ID has a non-zero first byte).
        Cache { keys: [0; 64], vals: [(GEN, 0); 64] }
    }
    #[inline(always)]
    fn get(&mut self, id: &Ident, k: u64) -> (u32, u64) {
        let i = (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) as usize;
        unsafe {
            if *self.keys.get_unchecked(i) == k {
                return *self.vals.get_unchecked(i);
            }
            let v = id.resolve(k);
            *self.keys.get_unchecked_mut(i) = k;
            *self.vals.get_unchecked_mut(i) = v;
            v
        }
    }
}

/// A run of one person's survivors in a range's walk.
#[derive(Clone, Copy)]
struct Run {
    start: u32,
    len: u32,
    comp: u32,
    name: u64,
}

#[derive(Default)]
struct PartOut {
    vals: Vec<u64>,
    runs: Vec<Run>,
}

#[derive(Default)]
struct Bucket {
    /// `(tc, click key)` by local person, and where each person starts.
    cl: Vec<(u64, u64)>,
    offs: Vec<u32>,
}

/// A run of a person with several runs: `(comp, part, start, len, name)`.
type Multi = (u32, u32, u32, u32, u64);

struct SyncPtr<T>(*mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

/// Up to `n` decoded keys from the entries, spread over partitions.
unsafe fn sample(entry_lists: &[Vec<List<Entry>>], n: usize, f: impl Fn(&Entry) -> u64, prefix: u8) -> Vec<u64> {
    let mut out = Vec::new();
    let parts = entry_lists.first().map_or(0, |t| t.len());
    let per = (n / parts.max(1)).max(8);
    'outer: for p in 0..parts {
        let mut took = 0;
        for t in entry_lists {
            for chunk in t[p].chunks() {
                for e in chunk {
                    if let Some(y) = packed_y(f(e), prefix) {
                        out.push(y);
                        took += 1;
                        if out.len() >= n {
                            break 'outer;
                        }
                        if took >= per {
                            continue 'outer;
                        }
                    }
                }
            }
        }
    }
    out
}

/// A hasher for keys that one multiply mixes well enough.
#[derive(Default, Clone, Copy)]
pub(super) struct MulHash(u64);
impl std::hash::Hasher for MulHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    #[inline(always)]
    fn write_u64(&mut self, k: u64) {
        let x = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = x ^ (x >> 29);
    }
}
pub(super) type Fx = std::hash::BuildHasherDefault<MulHash>;

/// The dense identity from the link pairs and the link map's roots.
#[inline(never)]
fn build_ident(pairs: &[(u64, u64)], links: &LinkMap, key: u64, np: usize) -> Ident {
    let person = |k: u64| links.stage3(links.stage2(links.stage1(k)), k);
    // Every linked ID: `(user number, key, root)`, or `(key, root)` when it
    // is not a dense user.
    let mut canon: Vec<(u64, u64, u64)> = Vec::with_capacity(pairs.len() * 2);
    let mut others: Vec<(u64, u64)> = Vec::new();
    // Smallest linked alias of each P, plus one (0: P not linked).
    let mut min_alias = vec![0u16; np];
    for &(a, b) in pairs {
        for k in [a, b] {
            let root = person(k);
            match packed_y(k, b'u').map(|y| decode::unscramble(y, key)) {
                Some(u) if ((u >> 9) as usize) < np => {
                    let (p, a) = ((u >> 9) as usize, (u & 511) as u16 + 1);
                    let m = &mut min_alias[p];
                    if *m == 0 || a < *m {
                        *m = a;
                    }
                    canon.push((u, k, root));
                }
                _ => others.push((k, root)),
            }
        }
    }
    // The home root of each linked P, in `names[P]`.
    let mut names: Vec<u64> = vec![0; np];
    for &(u, _, root) in &canon {
        let p = (u >> 9) as usize;
        if (u & 511) as u16 + 1 == min_alias[p] {
            names[p] = root;
        }
    }
    // A home component's id is the first P whose home it is, so
    // `names[id]` is its root.
    let mut comp_of: HashMap<u64, u32, Fx> = HashMap::default();
    let mut desc = vec![0u64; np];
    for p in 0..np {
        if min_alias[p] != 0 {
            let id = *comp_of.entry(names[p]).or_insert(p as u32);
            desc[p] = u64::from(id) << 32 | LINKED;
        }
    }
    let mut exc: HashMap<u64, u32, Fx> = HashMap::default();
    let mut id_of = |root: u64, names: &mut Vec<u64>| -> u32 {
        *comp_of.entry(root).or_insert_with(|| {
            names.push(root);
            names.len() as u32 - 1
        })
    };
    for &(u, k, root) in &canon {
        let (p, a) = ((u >> 9) as usize, u & 511);
        if root == names[p] && a < MASK_ALIASES {
            desc[p] |= 1 << a;
        } else {
            let c = id_of(root, &mut names);
            exc.insert(k, c);
        }
    }
    for (k, root) in others {
        let c = id_of(root, &mut names);
        exc.insert(k, c);
    }
    Ident { desc, names, exc, key }
}

/// Sort `v` by time if it is not in time order; the result is a slice of
/// `v` or of `scratch`.
unsafe fn in_order<'s>(v: &'s [u64], scratch: &'s mut Vec<u64>) -> &'s [u64] {
    let mut bad = false;
    for k in 1..v.len() {
        bad |= (*v.get_unchecked(k) >> 32) < (*v.get_unchecked(k - 1) >> 32);
    }
    if !bad {
        return v;
    }
    scratch.clear();
    scratch.extend_from_slice(v);
    scratch.sort_unstable();
    scratch
}

impl<'a> PersonPhase<'a> {
    /// One person from its survivors in time order (`ts_rel << 32 | price
    /// << 8 | category`) and its clicks `(tc, click key)`. `reserve` must
    /// have made room for a row per event.
    #[inline(never)]
    unsafe fn person_v(&mut self, pkey: u64, vals: &[u64], cl: &mut [(u64, u64)]) {
        let ws_in_day = self.fast.ws_in_day;
        let ndays = self.fast.ndays;
        let organic = self.fast.ncamp;
        let n = vals.len();
        let clicked = !cl.is_empty();
        if clicked {
            for v in [&mut self.s_at, &mut self.s_day] {
                if v.capacity() < n + LANES {
                    v.reserve(n + LANES - v.len());
                }
            }
            if self.s_revenue.capacity() < n {
                self.s_revenue.reserve(n - self.s_revenue.len());
            }
        }
        let (mut opened, mut slots) = (0u32, 0usize);
        let mut cur_day = u32::MAX;
        let (mut d_sessions, mut d_events, mut d_revenue) = (0u32, 0u32, 0u64);
        let p = vals.as_ptr();
        let mut i = 0usize;
        while i < n {
            let start = (*p.add(i) >> 32) as u32;
            let mut last = start;
            let mut revenue = 0u64;
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
            let mut j = i;
            while j < n {
                let v = *p.add(j);
                let t = (v >> 32) as u32;
                if t - last > SESSION_GAP_MS as u32 {
                    break;
                }
                last = t;
                let pc = v as u32;
                if pc != 0 {
                    let price = u64::from(pc >> 8);
                    let cat = (pc & 0xFF) as u8;
                    revenue += price;
                    let sum = self.catsum.get_unchecked_mut(cat as usize);
                    if *sum == 0 {
                        self.touched.push(cat);
                    }
                    *sum += price;
                }
                j += 1;
            }
            d_sessions += 1;
            opened += 1;
            d_events += (j - i) as u32;
            d_revenue += revenue;
            if clicked {
                let at = opened as usize - 1;
                *self.s_at.as_mut_ptr().add(at) = start + CLICK_REACH_MS as u32;
                *self.s_day.as_mut_ptr().add(at) = cur_day;
                *self.s_revenue.as_mut_ptr().add(at) = revenue;
            } else {
                let cell = self.agg.get_unchecked_mut(organic * ndays + cur_day as usize);
                cell[0] += SESSION | (u64::from(revenue > 0) << 32);
                cell[1] += revenue;
            }
            i = j;
        }
        if cur_day != u32::MAX {
            self.emit(pkey, cur_day, d_sessions, d_events, d_revenue);
        }
        if clicked {
            let sessions = opened as usize;
            let narrow = self.fast.cwlen < (1 << 31) - 1;
            if sessions > FEW || !narrow || !self.few_sessions(cl, sessions, slots) {
                self.many_sessions(cl, sessions);
            }
        }
    }
}

/// Everything the post shape needs from the parse phase.
pub(super) struct Parsed<'p> {
    pub entry_lists: &'p [Vec<List<Entry>>],
    pub click_blocks: &'p [Vec<Click>],
    pub pairs: &'p [(u64, u64)],
    pub links: &'p LinkMap,
}

/// Phase 1 of a hash partition: decode and route every entry.
#[inline(never)]
unsafe fn convert_part(lists: &[List<Entry>], ekey: u64, nr: usize, sc: &mut Scatter<Entry>, general: &mut Vec<Entry>) {
    let hot = sc.hot();
    for list in lists {
        for chunk in list.chunks() {
            for e in chunk {
                if let Some(y) = packed_y(e.ekey, b'e') {
                    let n = decode::unscramble(y, ekey);
                    let part = (n >> RBITS) as usize;
                    if part < nr {
                        sc.push_hot(&hot, part, Entry { ekey: n, ..*e });
                        continue;
                    }
                }
                general.push(*e);
            }
        }
    }
}

/// Phase 1 of a click block: resolve the user and route the click.
#[inline(never)]
unsafe fn convert_clicks(block: &[Click], id: &Ident, sc: &mut Scatter<(u64, u64)>, general: &mut Vec<Click>) {
    let hot = sc.hot();
    for c in block {
        let (comp, _) = id.resolve(c.key);
        if comp != GEN {
            let b = (comp >> CB) as usize;
            let local = u64::from(comp) & CLOCAL;
            sc.push_hot(&hot, b, (c.tc | local << CSHIFT, c.ckey));
        } else {
            general.push(*c);
        }
    }
}

/// Phase 2 of a range partition: dedup by number, then the walk.
#[inline(never)]
unsafe fn dedup_walk(
    lists: &[List<Entry>],
    base: u64,
    table: &mut [Entry],
    id: &Ident,
    cache: &mut Cache,
    runcount: &[AtomicU32],
    gen_events: &mut Vec<(u64, u64)>,
) -> PartOut {
    let total = dedup(lists, table);
    let t = table.as_mut_ptr();
    walk(t, base, total, id, cache, runcount, gen_events)
}

/// Copies of a range into the table by number, the smallest ingest kept.
#[inline(never)]
unsafe fn dedup(lists: &[List<Entry>], table: &mut [Entry]) -> usize {
    let t = table.as_mut_ptr();
    let mut total = 0usize;
    for list in lists {
        total += list.count();
        for chunk in list.chunks() {
            for e in chunk {
                let s = t.add((e.ekey as usize) & (R - 1));
                if (*s).ekey != e.ekey || e.ingest < (*s).ingest {
                    *s = *e;
                }
            }
        }
    }
    total
}

/// The walk's person state: the open run and where runs go.
struct Walker<'w> {
    id: &'w Ident,
    cache: &'w mut Cache,
    runcount: &'w [AtomicU32],
    runs: Vec<Run>,
    cur: Run,
}

impl Walker<'_> {
    /// A survivor with user key `k` follows one with another key: find its
    /// person and open a new run if the person changes. True when the key
    /// is a general person (its survivors go aside and the run stays open).
    #[inline(never)]
    fn switch(&mut self, k: u64, nv: usize) -> bool {
        let (comp, name) = self.cache.get(self.id, k);
        if comp == GEN {
            return true;
        }
        if comp != self.cur.comp {
            self.close(nv);
            self.cur = Run { start: nv as u32, len: 0, comp, name };
        }
        false
    }
    #[inline(always)]
    fn close(&mut self, nv: usize) {
        if self.cur.comp != GEN {
            self.cur.len = nv as u32 - self.cur.start;
            unsafe { self.runcount.get_unchecked(self.cur.comp as usize) }.fetch_add(1, Ordering::Relaxed);
            self.runs.push(self.cur);
        }
    }
}

/// The table in number order: survivors as runs of one person. The key of
/// the previous survivor is kept, so only a change of key costs a lookup.
#[inline(never)]
unsafe fn walk(
    t: *const Entry,
    base: u64,
    total: usize,
    id: &Ident,
    cache: &mut Cache,
    runcount: &[AtomicU32],
    gen_events: &mut Vec<(u64, u64)>,
) -> PartOut {
    let mut vals: Vec<u64> = Vec::with_capacity(total);
    let v = vals.as_mut_ptr();
    let mut nv = 0usize;
    let mut w = Walker { id, cache, runcount, runs: Vec::new(), cur: Run { start: 0, len: 0, comp: GEN, name: 0 } };
    // Key 0 is no ID, so the first survivor always looks up.
    let mut last = 0u64;
    let mut general = false;
    let mut s = t;
    let mut want = base;
    let stop = t.add(R);
    while s < stop {
        let e = &*s;
        s = s.add(1);
        want += 1;
        if e.ekey != want - 1 || e.w3 == DEAD {
            continue;
        }
        if e.ukey != last {
            last = e.ukey;
            general = w.switch(last, nv);
        }
        if general {
            gen_events.push((e.ukey, e.w3));
            continue;
        }
        *v.add(nv) = e.w3;
        nv += 1;
    }
    w.close(nv);
    vals.set_len(nv);
    PartOut { vals, runs: w.runs }
}

/// Phase 2 of a click bucket: counting sort by local person.
#[inline(never)]
unsafe fn bucket_sort(lists: &[List<(u64, u64)>]) -> Bucket {
    let mut offs = vec![0u32; (1 << CB) + 1];
    let mut n = 0usize;
    for list in lists {
        for chunk in list.chunks() {
            for c in chunk {
                *offs.get_unchecked_mut(((c.0 >> CSHIFT) & CLOCAL) as usize + 1) += 1;
            }
            n += chunk.len();
        }
    }
    for k in 1..offs.len() {
        offs[k] += offs[k - 1];
    }
    let mut cur: Vec<u32> = offs.clone();
    let mut cl: Vec<(u64, u64)> = Vec::with_capacity(n);
    let out = cl.as_mut_ptr();
    for list in lists {
        for chunk in list.chunks() {
            for c in chunk {
                let local = ((c.0 >> CSHIFT) & CLOCAL) as usize;
                let w = cur.get_unchecked_mut(local);
                *out.add(*w as usize) = (c.0 & !(CLOCAL << CSHIFT), c.1);
                *w += 1;
            }
        }
    }
    cl.set_len(n);
    Bucket { cl, offs }
}

/// The post shape from the parsed entries, or `None` (nothing changed)
/// when the IDs are not of the generator's form.
pub(super) fn run(
    fast: &Fast,
    lk: &Lookups,
    cfg: &Config,
    out: Option<&Path>,
    parsed: &Parsed,
    cpu_total: &Mutex<f64>,
    timer: &mut Timer,
) -> Option<Tables> {
    let threads = cfg.threads;
    let entry_lists = parsed.entry_lists;
    let nentries: usize = entry_lists.iter().map(|t| t.iter().map(|l| unsafe { l.count() }).sum::<usize>()).sum();
    if nentries == 0 || fast.ncamp >= 1 << CSHIFT {
        return None;
    }
    // Keys.
    let ebound = (2 * nentries as u64).next_power_of_two();
    let ubound = (32 * nentries as u64).next_power_of_two();
    if ubound > 1 << 38 {
        return None;
    }
    let setup_cpu0 = sys::thread_cpu();
    let ekey = decode::recover_key(&unsafe { sample(entry_lists, 4096, |e| e.ekey, b'e') }, ebound)?;
    let ukey = decode::recover_key(&unsafe { sample(entry_lists, 4096, |e| e.ukey, b'u') }, ubound)?;
    // Range partitions cover every number below the entry count and a bit
    // more; a number past them takes the general path.
    let nr = nentries / R + 2;
    let np = nentries / 8 + 1024;
    let id = build_ident(parsed.pairs, parsed.links, ukey, np);
    let ncomp = id.names.len();
    let nb = (ncomp >> CB) + 1;
    // Phase clocks for `ENGINE_TIMING`: summed thread CPU of convert, of the
    // rest of resolve, and of the person stage; the wall ends of convert
    // and resolve (taken by thread 0 after each barrier).
    let setup_cpu = sys::thread_cpu() - setup_cpu0;
    let cpus: Mutex<[f64; 3]> = Mutex::new([0.0; 3]);
    let walls: Mutex<[Option<Instant>; 2]> = Mutex::new([None; 2]);
    let t_threads = Instant::now();

    let runcount: Vec<AtomicU32> = (0..ncomp).map(|_| AtomicU32::new(0)).collect();
    let parts1 = entry_lists[0].len();
    let ranges: Mutex<Vec<Vec<List<Entry>>>> = Mutex::new(Vec::new());
    let cranges: Mutex<Vec<Vec<List<(u64, u64)>>>> = Mutex::new(Vec::new());
    let gen_entries: Mutex<Vec<Entry>> = Mutex::new(Vec::new());
    let gen_clicks: Mutex<Vec<Click>> = Mutex::new(Vec::new());
    let gen_ev: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
    let part_out: Vec<Mutex<PartOut>> = (0..=nr).map(|_| Mutex::new(PartOut::default())).collect();
    let buckets: Vec<Mutex<Bucket>> = (0..nb).map(|_| Mutex::new(Bucket::default())).collect();
    let multi: Mutex<Vec<Multi>> = Mutex::new(Vec::new());
    let groups: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let next = [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)];
    let barrier = Barrier::new(threads);
    let collected: Mutex<Vec<UserRow>> = Mutex::new(Vec::new());
    let shared_cells: Mutex<Cells> = Mutex::new(Cells::default());
    let out_dir = out.map(|o| o.join("user_daily"));
    let mut total_rows = 0u64;
    let mut check = (0u64, 0u64);
    let click_blocks = parsed.click_blocks;
    let id = &id;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (runcount, ranges, cranges, gen_entries, gen_clicks, gen_ev) = (&runcount, &ranges, &cranges, &gen_entries, &gen_clicks, &gen_ev);
                let (part_out, buckets, multi, groups, next, barrier) = (&part_out, &buckets, &multi, &groups, &next, &barrier);
                let (collected, shared_cells, out_dir) = (&collected, &shared_cells, &out_dir);
                let (cpus, walls) = (&cpus, &walls);
                scope.spawn(move || unsafe {
                    let mut cache = Cache::new();
                    // ---- 1: convert
                    let mut sc: Scatter<Entry> = Scatter::new(nr, CHUNK_BYTES);
                    let mut csc: Scatter<(u64, u64)> = Scatter::new(nb, CHUNK_BYTES);
                    let mut general: Vec<Entry> = Vec::new();
                    let mut gclicks: Vec<Click> = Vec::new();
                    let mut lists: Vec<List<Entry>> = Vec::with_capacity(threads);
                    loop {
                        let at = next[0].fetch_add(1, Ordering::Relaxed);
                        if at < parts1 {
                            lists.clear();
                            lists.extend(entry_lists.iter().map(|l| l[at]));
                            convert_part(&lists, ekey, nr, &mut sc, &mut general);
                        } else if let Some(block) = click_blocks.get(at - parts1) {
                            convert_clicks(block, id, &mut csc, &mut gclicks);
                        } else {
                            break;
                        }
                    }
                    ranges.lock().unwrap().push(sc.finish());
                    cranges.lock().unwrap().push(csc.finish());
                    gen_entries.lock().unwrap().extend(general);
                    gen_clicks.lock().unwrap().extend(gclicks);
                    let c1 = sys::thread_cpu();
                    cpus.lock().unwrap()[0] += c1;
                    barrier.wait();
                    if t == 0 {
                        walls.lock().unwrap()[0] = Some(Instant::now());
                    }
                    // ---- 2: dedup and walk, click buckets
                    let ranges = ranges.lock().unwrap().clone();
                    let cranges = cranges.lock().unwrap().clone();
                    let mut table: Vec<Entry> = vec![Entry { ekey: u64::MAX, ukey: 0, ingest: 0, w3: 0 }; R];
                    let mut gev: Vec<(u64, u64)> = Vec::new();
                    let mut cl_lists: Vec<List<(u64, u64)>> = Vec::with_capacity(threads);
                    loop {
                        let at = next[1].fetch_add(1, Ordering::Relaxed);
                        if at < nr {
                            lists.clear();
                            lists.extend(ranges.iter().map(|l| l[at]));
                            let po = dedup_walk(&lists, (at as u64) << RBITS, &mut table, id, &mut cache, runcount, &mut gev);
                            *part_out[at].lock().unwrap() = po;
                        } else if at == nr {
                            // The general entries: exact dedup by key.
                            let mut best: HashMap<u64, Entry> = HashMap::new();
                            for e in gen_entries.lock().unwrap().iter() {
                                best.entry(e.ekey).and_modify(|b| if e.ingest < b.ingest { *b = *e }).or_insert(*e);
                            }
                            let mut po = PartOut::default();
                            for e in best.values() {
                                if e.w3 == DEAD {
                                    continue;
                                }
                                let (comp, name) = id.resolve(e.ukey);
                                if comp == GEN {
                                    gev.push((e.ukey, e.w3));
                                } else {
                                    runcount[comp as usize].fetch_add(1, Ordering::Relaxed);
                                    po.runs.push(Run { start: po.vals.len() as u32, len: 1, comp, name });
                                    po.vals.push(e.w3);
                                }
                            }
                            *part_out[nr].lock().unwrap() = po;
                        } else if at < nr + 1 + nb {
                            let b = at - nr - 1;
                            cl_lists.clear();
                            cl_lists.extend(cranges.iter().map(|l| l[b]));
                            *buckets[b].lock().unwrap() = bucket_sort(&cl_lists);
                        } else {
                            break;
                        }
                    }
                    gen_ev.lock().unwrap().extend(gev);
                    let c2 = sys::thread_cpu();
                    cpus.lock().unwrap()[1] += c2 - c1;
                    barrier.wait();
                    if t == 0 {
                        walls.lock().unwrap()[1] = Some(Instant::now());
                    }
                    // ---- 3: persons with one run
                    let user_out = UserOut::new(out_dir.as_deref(), t, lk, fast.ws_day, fast.ndays, cfg.collect.then_some(collected));
                    let mut phase = PersonPhase::new(fast, user_out);
                    let bucket_ptrs: Vec<(SyncPtr<(u64, u64)>, *const u32)> = buckets
                        .iter()
                        .map(|b| {
                            let mut g = b.lock().unwrap();
                            (SyncPtr(g.cl.as_mut_ptr()), g.offs.as_ptr())
                        })
                        .collect();
                    let clicks_of = |c: u32| -> &mut [(u64, u64)] {
                        let (cl, offs) = &bucket_ptrs[(c >> CB) as usize];
                        let local = (u64::from(c) & CLOCAL) as usize;
                        let (a, b) = (*offs.add(local) as usize, *offs.add(local + 1) as usize);
                        std::slice::from_raw_parts_mut(cl.0.add(a), b - a)
                    };
                    let mut scratch: Vec<u64> = Vec::new();
                    let mut mine: Vec<Multi> = Vec::new();
                    loop {
                        let at = next[2].fetch_add(1, Ordering::Relaxed);
                        if at > nr {
                            break;
                        }
                        let po = part_out[at].lock().unwrap();
                        phase.out.reserve(po.vals.len());
                        for r in &po.runs {
                            if runcount[r.comp as usize].load(Ordering::Relaxed) == 1 {
                                let v = std::slice::from_raw_parts(po.vals.as_ptr().add(r.start as usize), r.len as usize);
                                let v = in_order(v, &mut scratch);
                                phase.person_v(r.name, v, clicks_of(r.comp));
                            } else {
                                mine.push((r.comp, at as u32, r.start, r.len, r.name));
                            }
                        }
                        phase.out.flush_full();
                    }
                    multi.lock().unwrap().extend(mine);
                    barrier.wait();
                    if t == 0 {
                        let mut m = multi.lock().unwrap();
                        m.sort_unstable_by_key(|x| (x.0, x.1, x.2));
                        let mut g = groups.lock().unwrap();
                        let mut i = 0;
                        while i < m.len() {
                            let mut j = i + 1;
                            while j < m.len() && m[j].0 == m[i].0 {
                                j += 1;
                            }
                            g.push((i, j));
                            i = j;
                        }
                    }
                    barrier.wait();
                    // ---- 4: persons with several runs, and the general persons
                    let m = multi.lock().unwrap().clone();
                    let g = groups.lock().unwrap().clone();
                    let pv: Vec<*const u64> = part_out.iter().map(|p| p.lock().unwrap().vals.as_ptr()).collect();
                    let mut gather: Vec<u64> = Vec::new();
                    loop {
                        let at = next[3].fetch_add(1, Ordering::Relaxed);
                        if at < g.len() {
                            let (i, j) = g[at];
                            gather.clear();
                            for &(_, part, start, len, _) in &m[i..j] {
                                gather.extend_from_slice(std::slice::from_raw_parts(pv[part as usize].add(start as usize), len as usize));
                            }
                            phase.out.reserve(gather.len());
                            let v = in_order(&gather, &mut scratch);
                            phase.person_v(m[i].4, v, clicks_of(m[i].0));
                            phase.out.flush_full();
                        } else if at == g.len() {
                            // General persons: one ID each.
                            let mut ev: HashMap<u64, Vec<u64>> = HashMap::new();
                            for &(k, w) in gen_ev.lock().unwrap().iter() {
                                ev.entry(k).or_default().push(w);
                            }
                            let mut cl: HashMap<u64, Vec<(u64, u64)>> = HashMap::new();
                            for c in gen_clicks.lock().unwrap().iter() {
                                if ev.contains_key(&c.key) {
                                    cl.entry(c.key).or_default().push((c.tc, c.ckey));
                                }
                            }
                            for (k, mut v) in ev {
                                v.sort_unstable();
                                phase.out.reserve(v.len());
                                let mut none = Vec::new();
                                let c = cl.get_mut(&k).unwrap_or(&mut none);
                                phase.person_v(k, &v, c);
                                phase.out.flush_full();
                            }
                        } else {
                            break;
                        }
                    }
                    phase.out.finish();
                    let cells = phase.cells();
                    shared_cells.lock().unwrap().add(cells);
                    cpus.lock().unwrap()[2] += sys::thread_cpu() - c2;
                    (phase.out.rows_written, phase.check)
                })
            })
            .collect();
        for h in handles {
            let (rows, sums) = h.join().unwrap();
            check.0 += sums.0;
            check.1 += sums.1;
            total_rows += rows;
        }
    });
    let cells = std::mem::take(&mut *shared_cells.lock().unwrap());
    if timer.on {
        // Three standard lines (resolve = everything between the parser and
        // the person stage, setup and conversion included) and the
        // conversion alone, so it can be subtracted.
        let end = Instant::now();
        let [w1, w2] = *walls.lock().unwrap();
        let (w1, w2) = (w1.unwrap_or(end), w2.unwrap_or(end));
        let c = *cpus.lock().unwrap();
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        let line = |name: &str, wall: f64, cpu: f64, at: Instant| {
            eprintln!(
                "[engine] {name:<10} wall {wall:7.1} ms  cpu {:7.1} ms  (at {:7.1} ms)",
                cpu * 1e3,
                (at - timer.t0).as_secs_f64() * 1e3
            );
        };
        line("convert", ms(t_threads, w1), c[0], w1);
        line("resolve", ms(timer.last, w2), setup_cpu + c[0] + c[1], w2);
        line("person", ms(w2, end), c[2], end);
        timer.last = end;
    }
    let _ = cpu_total;
    let mut tables = finish(fast, lk, cfg, out, cells, total_rows, check, collected.into_inner().unwrap(), timer);
    tables.dense = true;
    if cfg.timing {
        let runs: usize = part_out.iter().map(|p| p.lock().unwrap().runs.len()).sum();
        let vals: usize = part_out.iter().map(|p| p.lock().unwrap().vals.len()).sum();
        let multi = multi.lock().unwrap().len();
        eprintln!("[post] runs {runs} vals {vals} multi runs {multi} comps {ncomp} groups {}", groups.lock().unwrap().len());
    }
    Some(tables)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_y_matches_decode() {
        let mut s = 5u64;
        let mut rng = || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let alphabet = b"0123456789abcdefghijklmnopqrstuvwxyz/:`{@[AZ \"\x01\x7f";
        for i in 0..400_000 {
            let len = (rng() % 11) as usize;
            let mut id: Vec<u8> = (0..len).map(|_| alphabet[(rng() % alphabet.len() as u64) as usize]).collect();
            if len > 0 && rng() % 4 != 0 {
                id[0] = if i % 2 == 0 { b'u' } else { b'e' };
            }
            if i % 3 == 0 {
                let mut e = Vec::new();
                decode::encode(b'u', rng() & ((1u64 << (rng() % 42)) - 1), &mut e);
                id = e;
            }
            let mut buf = id.clone();
            buf.extend_from_slice(&[0u8; 16]);
            let k = unsafe { ids::pack(buf.as_ptr(), id.len()) };
            for prefix in [b'u', b'e'] {
                assert_eq!(packed_y(k, prefix), decode::decode_slice(&id, prefix), "{id:?}");
            }
        }
    }
}
