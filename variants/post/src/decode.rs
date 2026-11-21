//! Generator IDs as small dense integers (not wired into the engine yet).
//!
//! The public generator writes an ID as one prefix letter and the lowercase
//! base 36 of `scramble(number, key)`, a bijection on 40 bits. Event and
//! click numbers are dense counters; a user number is `person * 512 + alias`.
//!
//! Exactness does not depend on the key. `decode` maps canonical text one to
//! one onto `0..2^40` (lowercase digits, no leading zero, 1 to 8 digits,
//! value below 2^40), and `unscramble(_, key)` is a bijection on `0..2^40`
//! for every key. So for a fixed prefix and any key, two canonical IDs get
//! the same number if and only if they are the same text. A wrong key can
//! only make numbers large: the engine range checks every number and sends
//! out-of-range and non-canonical IDs down the text path. Routing is a
//! function of the text alone, so the two key spaces hold disjoint sets of
//! IDs and never need to be compared with each other.
//!
//! What the key does decide is density. `recover_key` finds the key from a
//! sample and returns `None` unless exactly one key puts the whole sample
//! below the bound.

#![allow(dead_code)]

use crate::ts::load64;

pub const MASK: u64 = (1 << 40) - 1;
const M1: u64 = 0x9E37_79B9_7F4B;
const M2: u64 = 0xD6E8_FEB8_6659;

/// The inverse of an odd `m` modulo 2^40 (Newton: each step doubles the
/// correct low bits; `m` itself is right to 3 bits).
const fn inverse(m: u64) -> u64 {
    let mut x = m;
    let mut i = 0;
    while i < 6 {
        x = x.wrapping_mul(2u64.wrapping_sub(m.wrapping_mul(x)));
        i += 1;
    }
    x & MASK
}

pub const M1_INV: u64 = inverse(M1);
pub const M2_INV: u64 = inverse(M2);

/// The key-free half of the inverse: undo the last multiply and the
/// xor-shift. `scramble` is `M2 * s(M1 * v ^ key)` with `s(x) = x ^ x >> 17`
/// on 40 bits; `s` is undone by `x ^ x >> 17 ^ x >> 34` (51 > 40).
#[inline(always)]
pub fn unmix(y: u64) -> u64 {
    let x = y.wrapping_mul(M2_INV) & MASK;
    x ^ (x >> 17) ^ (x >> 34)
}

/// The keyed half: `z = M1 * v ^ key`, so `v = M1^-1 * (z ^ key)`. Key bits
/// above 40 only reach product bits above 40, which the mask drops.
#[inline(always)]
pub fn unkey(z: u64, key: u64) -> u64 {
    (z ^ key).wrapping_mul(M1_INV) & MASK
}

/// Exact inverse of the generator's `scramble` on `0..2^40`.
#[inline(always)]
pub fn unscramble(value: u64, key: u64) -> u64 {
    unkey(unmix(value), key)
}

// ---------------------------------------------------------------- decode

/// Smallest value with `n` digits and no leading zero (`0` is the one
/// 1-digit exception), and the number of values from there to the end of
/// the n-digit range, capped at 2^40.
const LO: [u64; 9] = {
    let mut t = [0u64; 9];
    let mut n = 2;
    while n <= 8 {
        t[n] = 36u64.pow(n as u32 - 1);
        n += 1;
    }
    t
};
const SPAN: [u64; 9] = {
    let mut t = [0u64; 9];
    let mut n = 1;
    while n <= 8 {
        let hi = if 36u64.pow(n as u32) < (1 << 40) { 36u64.pow(n as u32) } else { 1 << 40 };
        t[n] = hi - LO[n];
        n += 1;
    }
    t
};

/// The 40-bit scrambled value of a canonical ID: `prefix`, then 1 to 8
/// lowercase base-36 digits with no leading zero, value below 2^40. Any
/// other text gives `None`, so text and value correspond one to one.
///
/// Reads `p[0..9]` when `2 <= len <= 9` and nothing otherwise; bytes past
/// `len` do not change the result.
#[inline(always)]
pub unsafe fn decode(p: *const u8, len: usize, prefix: u8) -> Option<u64> {
    #[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
    {
        decode_sse(p, len, prefix)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "ssse3")))]
    {
        decode_swar(p, len, prefix)
    }
}

const H: u64 = 0x8080_8080_8080_8080;
const ONES: u64 = 0x0101_0101_0101_0101;

/// Low `n` bytes set.
const LANES: [u64; 9] = {
    let mut t = [0u64; 9];
    let mut n = 1;
    while n <= 8 {
        t[n] = if n == 8 { u64::MAX } else { (1u64 << (8 * n)) - 1 };
        n += 1;
    }
    t
};

/// Portable twin of `decode`: eight bytes in one register.
#[inline(always)]
pub unsafe fn decode_swar(p: *const u8, len: usize, prefix: u8) -> Option<u64> {
    let n = len.wrapping_sub(2);
    if n > 7 || *p != prefix {
        return None;
    }
    let n = n + 1; // digits, 1..=8
    let x = load64(p.add(1)); // byte k of the digits in lane k
    let lanes = *LANES.get_unchecked(n);
    // Lane flags in bit 7. y <= 0x7F and every addend <= 0x50, so no lane
    // carries into the next.
    let y = x & !H;
    let ge_0 = y.wrapping_add(0x50 * ONES); // >= '0'
    let ge_colon = y.wrapping_add(0x46 * ONES); // >= ':'
    let ge_a = y.wrapping_add(0x1F * ONES); // >= 'a'
    let ge_brace = y.wrapping_add(0x05 * ONES); // >= '{'
    let digit = ge_0 & !ge_colon;
    let letter = ge_a & !ge_brace;
    let ok = (digit | letter) & !x & H;
    if (ok ^ H) & lanes & H != 0 {
        return None;
    }
    // Digit values: c - 48 for '0'..'9', c - 87 for 'a'..'z'. Lanes past
    // the ID may borrow, but only upward into other lanes past the ID,
    // which the mask clears.
    let val = (y.wrapping_sub(0x30 * ONES).wrapping_sub(((letter & H) >> 7) * 39)) & lanes;
    // Right-align: last digit in lane 7, leading lanes zero.
    let a = val << (8 * (8 - n));
    // Lane 2j is the more significant digit of each pair.
    let pair = (a & 0x00FF_00FF_00FF_00FF) * 36 + ((a >> 8) & 0x00FF_00FF_00FF_00FF);
    let quad = (pair & 0x0000_FFFF_0000_FFFF) * 1296 + ((pair >> 16) & 0x0000_FFFF_0000_FFFF);
    let value = (quad & 0xFFFF_FFFF) * 1_679_616 + (quad >> 32);
    if value.wrapping_sub(*LO.get_unchecked(n)) < *SPAN.get_unchecked(n) {
        Some(value)
    } else {
        None
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
static SHUF: [[u8; 16]; 9] = {
    let mut t = [[0x80u8; 16]; 9];
    let mut n = 1;
    while n <= 8 {
        let mut k = 8 - n;
        while k < 8 {
            t[n][k] = (k - (8 - n)) as u8;
            k += 1;
        }
        n += 1;
    }
    t
};

/// `decode` with SSSE3: range checks with unsigned min, right-align with
/// one shuffle, then `pmaddubsw` and `pmaddwd` fold digits to two 4-digit
/// halves.
#[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
#[inline(always)]
pub unsafe fn decode_sse(p: *const u8, len: usize, prefix: u8) -> Option<u64> {
    use core::arch::x86_64::*;
    let n = len.wrapping_sub(2);
    if n > 7 || *p != prefix {
        return None;
    }
    let n = n + 1;
    let x = _mm_loadl_epi64(p.add(1) as *const __m128i);
    let d = _mm_sub_epi8(x, _mm_set1_epi8(b'0' as i8));
    let digit = _mm_cmpeq_epi8(_mm_min_epu8(d, _mm_set1_epi8(9)), d);
    let l = _mm_sub_epi8(d, _mm_set1_epi8(49));
    let letter = _mm_cmpeq_epi8(_mm_min_epu8(l, _mm_set1_epi8(25)), l);
    let ok = _mm_movemask_epi8(_mm_or_si128(digit, letter)) as u32;
    if !ok & ((1u32 << n) - 1) != 0 {
        return None;
    }
    let val = _mm_sub_epi8(d, _mm_andnot_si128(digit, _mm_set1_epi8(39)));
    let a = _mm_shuffle_epi8(val, _mm_loadu_si128(SHUF.get_unchecked(n).as_ptr() as *const __m128i));
    let pair = _mm_maddubs_epi16(a, _mm_set1_epi16(0x0124)); // bytes 36, 1
    let quad = _mm_madd_epi16(pair, _mm_set1_epi32(0x0001_0510)); // words 1296, 1
    let q = _mm_cvtsi128_si64(quad) as u64;
    let value = (q & 0xFFFF_FFFF) * 1_679_616 + (q >> 32);
    if value.wrapping_sub(*LO.get_unchecked(n)) < *SPAN.get_unchecked(n) {
        Some(value)
    } else {
        None
    }
}

/// Safe wrapper for tests and cold paths.
pub fn decode_slice(bytes: &[u8], prefix: u8) -> Option<u64> {
    let mut buf = [0u8; 16];
    if bytes.len() > 9 {
        return None;
    }
    buf[..bytes.len()].copy_from_slice(bytes);
    unsafe { decode(buf.as_ptr(), bytes.len(), prefix) }
}

/// The canonical text of a scrambled value (the generator's `base36`).
pub fn encode(prefix: u8, value: u64, out: &mut Vec<u8>) {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut buf = [0u8; 13];
    let mut at = buf.len();
    let mut rest = value;
    loop {
        at -= 1;
        buf[at] = DIGITS[(rest % 36) as usize];
        rest /= 36;
        if rest == 0 {
            break;
        }
    }
    out.push(prefix);
    out.extend_from_slice(&buf[at..]);
}

// ---------------------------------------------------------- key recovery

/// True when `a` and `b` (unmixed) can both decode below `bound` under a
/// key that agrees with `key` on the bits where `a` and `b` differ.
///
/// `v_a - v_b = M1^-1 * ((a ^ key) - (b ^ key))`, and the difference
/// `(a ^ key) - (b ^ key)` is a signed sum of the powers of two where `a`
/// and `b` differ, with each sign set by that bit of the key. So the test
/// uses only those key bits, and with both values in `0..bound` the
/// difference lies in `(-bound, bound)`.
#[inline(always)]
fn pair_ok(a: u64, b: u64, key: u64, bound: u64) -> bool {
    let delta = unkey(a, key).wrapping_sub(unkey(b, key)) & MASK;
    (delta + bound - 1) & MASK < 2 * bound - 1
}

/// The 40-bit key for which every scrambled value in `sample` unscrambles
/// below `bound`, or `None` when there is no such key, more than one, or
/// the search would cost too much (sample too small or bound too large).
///
/// Method: sort the unmixed values; neighbours differ in few bits, and each
/// neighbour pair pins the key bits where it differs (`pair_ok`), so a
/// small tree over pairs fixes most low key bits. Then the low `t` key bits
/// give the low `t` bits of one sample's number; enumerating its high bits
/// below `bound` gives the whole key, which must pass the full sample.
/// Not generator specific: any key that passes is a key that passes.
pub fn recover_key(sample: &[u64], bound: u64) -> Option<u64> {
    const CAP: usize = 1 << 12; // candidate list limit
    const WORK: u64 = 1 << 22; // pair tests and final enumeration limit
    if sample.is_empty() || bound == 0 || bound > 1 << 38 || sample.iter().any(|&y| y > MASK) {
        return None;
    }
    let mut z: Vec<u64> = sample.iter().map(|&y| unmix(y)).collect();
    z.sort_unstable();
    z.dedup();
    let mut pairs: Vec<(u64, u64)> =
        z.windows(2).map(|w| (w[0], w[1])).filter(|(a, b)| (a ^ b).count_ones() <= 20).collect();
    pairs.sort_unstable_by_key(|(a, b)| ((a ^ b).count_ones(), 64 - (a ^ b).leading_zeros()));

    let mut known = 0u64;
    let mut cands = vec![0u64];
    let mut work = 0u64;
    for &(a, b) in &pairs {
        let new = (a ^ b) & !known;
        let cost = (cands.len() as u64) << new.count_ones();
        if work + cost > WORK {
            continue;
        }
        work += cost;
        let mut next = Vec::new();
        for &c in &cands {
            let mut s = 0u64;
            loop {
                let k = c | s;
                if pair_ok(a, b, k, bound) {
                    next.push(k);
                }
                s = s.wrapping_sub(new) & new;
                if s == 0 {
                    break;
                }
            }
        }
        if next.is_empty() {
            return None; // no key fits the whole sample
        }
        if next.len() > CAP {
            continue; // too loose to keep; try other pairs
        }
        known |= new;
        cands = next;
    }

    // Choose t: free low bits to enumerate times numbers per low pattern.
    let mut best = (u64::MAX, 0u32);
    for t in 0..=40u32 {
        let low = if t == 40 { MASK } else { (1u64 << t) - 1 };
        let free = (low & !known).count_ones();
        if free > 30 {
            continue;
        }
        let per = (bound + low) >> t; // ceil(bound / 2^t)
        let cost = per.saturating_mul(1u64 << free).saturating_mul(cands.len() as u64);
        if cost < best.0 {
            best = (cost, t);
        }
    }
    if best.0 > WORK {
        return None;
    }
    let t = best.1;
    let low = if t == 40 { MASK } else { (1u64 << t) - 1 };
    let free = low & !known;
    let z0 = z[0];
    let mut found: Option<u64> = None;
    for &c in &cands {
        let mut f = 0u64;
        loop {
            let lo = (c | f) & low;
            let r = unkey(z0, lo) & low; // the low t bits of z0's number
            let mut v = r;
            while v < bound {
                let key = (v.wrapping_mul(M1) & MASK) ^ z0;
                if key & known == c && z.iter().all(|&zi| unkey(zi, key) < bound) {
                    match found {
                        Some(k) if k != key => return None, // ambiguous
                        _ => found = Some(key),
                    }
                }
                if t == 40 {
                    break;
                }
                v += 1 << t;
            }
            f = f.wrapping_sub(free) & free;
            if f == 0 {
                break;
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Written from the kit generator's `text.rs`.
    fn scramble(value: u64, key: u64) -> u64 {
        let mut m = value & MASK;
        m = (m.wrapping_mul(M1) ^ key) & MASK;
        m ^= m >> 17;
        m.wrapping_mul(M2) & MASK
    }

    fn rng(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Plain reference for `decode`.
    fn reference(bytes: &[u8], prefix: u8) -> Option<u64> {
        if bytes.len() < 2 || bytes.len() > 9 || bytes[0] != prefix {
            return None;
        }
        let digits = &bytes[1..];
        if digits.len() > 1 && digits[0] == b'0' {
            return None;
        }
        let mut v = 0u64;
        for &c in digits {
            let d = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'z' => c - b'a' + 10,
                _ => return None,
            };
            v = v * 36 + u64::from(d);
        }
        (v <= MASK).then_some(v)
    }

    #[test]
    fn inverses() {
        assert_eq!(M1.wrapping_mul(M1_INV) & MASK, 1);
        assert_eq!(M2.wrapping_mul(M2_INV) & MASK, 1);
    }

    #[test]
    fn unscramble_round_trip() {
        let mut s = 1;
        let edges = [0, 1, 2, 511, 512, 1 << 39, (1 << 39) + 1, MASK - 1, MASK];
        let keys = [0, 1, MASK, u64::MAX, 0xDEAD_BEEF_0000_0001, 1 << 40, rng(&mut s)];
        for &k in &keys {
            for &v in &edges {
                assert_eq!(unscramble(scramble(v, k), k), v);
                assert_eq!(scramble(unscramble(v, k), k), v);
            }
        }
        for _ in 0..200_000 {
            let k = rng(&mut s);
            let v = rng(&mut s) & MASK;
            assert_eq!(unscramble(scramble(v, k), k), v);
            assert_eq!(scramble(unscramble(v, k), k), v);
        }
    }

    fn check(bytes: &[u8], prefix: u8) {
        let want = reference(bytes, prefix);
        let mut a = bytes.to_vec();
        a.extend_from_slice(&[0xFF; 16]);
        let mut b = bytes.to_vec();
        b.extend_from_slice(b",\"\n0azZ9\x00\x80abcdefgh");
        unsafe {
            assert_eq!(decode_swar(a.as_ptr(), bytes.len(), prefix), want, "{bytes:?}");
            assert_eq!(decode_swar(b.as_ptr(), bytes.len(), prefix), want, "{bytes:?}");
            assert_eq!(decode(a.as_ptr(), bytes.len(), prefix), want, "{bytes:?}");
            #[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
            {
                assert_eq!(decode_sse(a.as_ptr(), bytes.len(), prefix), want, "{bytes:?}");
                assert_eq!(decode_sse(b.as_ptr(), bytes.len(), prefix), want, "{bytes:?}");
            }
        }
        assert_eq!(decode_slice(bytes, prefix), want);
    }

    #[test]
    fn decode_matches_reference() {
        let mut s = 7;
        // Canonical encodings of edge and random values.
        let mut values = vec![0, 1, 35, 36, 1295, 1296, MASK, MASK - 1];
        for n in 1..=8u32 {
            let p = 36u64.pow(n);
            values.extend_from_slice(&[p - 1, p, p + 1, p.saturating_sub(36)]);
        }
        for _ in 0..100_000 {
            let bits = rng(&mut s) % 41;
            values.push(rng(&mut s) & ((1u64 << bits) - 1));
        }
        for &v in &values {
            let mut id = Vec::new();
            encode(b'e', v, &mut id);
            check(&id, b'e');
            check(&id, b'u');
            if v <= MASK {
                assert_eq!(decode_slice(&id, b'e'), Some(v), "{id:?}");
            }
            // Leading zero, uppercase, a changed byte: all must be refused
            // or decode to what the reference says.
            let mut z = id.clone();
            z.insert(1, b'0');
            check(&z, b'e');
            let mut u = id.clone();
            u.make_ascii_uppercase();
            check(&u, b'e');
            check(&u, b'E');
            let at = (rng(&mut s) as usize) % id.len();
            let mut c = id.clone();
            c[at] = rng(&mut s) as u8;
            check(&c, b'e');
        }
        // Values at and past 2^40 in 8 digits.
        for v in [MASK, MASK + 1, 36u64.pow(8) - 1] {
            let mut id = Vec::new();
            encode(b'k', v, &mut id);
            check(&id, b'k');
        }
        // Every byte in every position of short IDs.
        for len in 0..=10usize {
            for pos in 0..len {
                for byte in 0..=255u8 {
                    let mut id = vec![b'u'; len];
                    for x in id.iter_mut().skip(1) {
                        *x = b'7';
                    }
                    id[pos] = byte;
                    check(&id, b'u');
                }
            }
        }
        // Random junk.
        for _ in 0..300_000 {
            let len = (rng(&mut s) % 12) as usize;
            let style = rng(&mut s) % 4;
            let id: Vec<u8> = (0..len)
                .map(|i| match (i, style) {
                    (0, 0..=2) => b'e',
                    (_, 0) => b"0123456789abcdefghijklmnopqrstuvwxyz"[(rng(&mut s) % 36) as usize],
                    (_, 1) => b"0123456789abcdefghijklmnopqrstuvwxyz/:`{@[AZ \"\x00\x80\xff"[(rng(&mut s) % 49) as usize],
                    _ => rng(&mut s) as u8,
                })
                .collect();
            check(&id, b'e');
        }
    }

    fn sample_keyed(numbers: &[u64], key: u64) -> Vec<u64> {
        numbers.iter().map(|&v| scramble(v, key)).collect()
    }

    #[test]
    fn recover_events_users_and_refusals() {
        let mut s = 11;
        for &(events, n) in &[(2_000u64, 512usize), (100_000, 2048), (1_000_000, 4096), (20_000_000, 8192), (60_000_000, 8192)] {
            for _ in 0..4 {
                let key = rng(&mut s);
                let bound = (2 * events).next_power_of_two();
                let nums: Vec<u64> = (0..n).map(|_| rng(&mut s) % events).collect();
                let t = std::time::Instant::now();
                let got = recover_key(&sample_keyed(&nums, key), bound);
                let el = t.elapsed();
                assert_eq!(got, Some(key & MASK), "events {events} n {n}");
                assert!(el.as_millis() < 50, "{el:?}");
                // Users: person * 512 + alias, alias mostly 0.
                let persons = events / 30 + 20;
                let users: Vec<u64> = (0..n)
                    .map(|_| {
                        let p = rng(&mut s) % persons;
                        let a = if rng(&mut s) % 2 == 0 { 0 } else { rng(&mut s) % 4 };
                        p * 512 + a
                    })
                    .collect();
                let ubound = (persons * 512).next_power_of_two();
                assert_eq!(recover_key(&sample_keyed(&users, key), ubound), Some(key & MASK), "users {events}");
            }
        }
        // Random data, too small a bound, or one outlier: no key.
        let junk: Vec<u64> = (0..4096).map(|_| rng(&mut s) & MASK).collect();
        assert_eq!(recover_key(&junk, 1 << 27), None);
        let key = rng(&mut s);
        let nums: Vec<u64> = (0..4096).map(|_| rng(&mut s) % 1_000_000).collect();
        assert_eq!(recover_key(&sample_keyed(&nums, key), 1 << 16), None);
        let mut out = nums.clone();
        out.push(1 << 39);
        assert_eq!(recover_key(&sample_keyed(&out, key), 1 << 21), None);
        assert_eq!(recover_key(&[], 1 << 21), None);
        assert_eq!(recover_key(&[MASK + 1], 1 << 21), None);
    }

    // ------------------------------------------------------ measurements

    /// `SS_DEC_DATA=dir1,dir2 cargo test --release decode::tests::measure -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn measure() {
        let dirs = std::env::var("SS_DEC_DATA").expect("SS_DEC_DATA");
        for dir in dirs.split(',') {
            measure_one(std::path::Path::new(dir));
        }
    }

    struct Col {
        canon: u64,
        other: u64,
        values: Vec<u64>,
    }

    /// Per file: the canonical scrambled values of the chosen columns.
    fn read_kind(dir: &std::path::Path, kind: &str, fields: usize, cols: &[(usize, u8)]) -> (u64, Vec<Vec<Col>>) {
        let mut paths: Vec<_> = std::fs::read_dir(dir.join(kind)).unwrap().map(|e| e.unwrap().path()).collect();
        paths.sort();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results = std::sync::Mutex::new((0..paths.len()).map(|_| None).collect::<Vec<Option<Vec<Col>>>>());
        let lines_total = std::sync::atomic::AtomicU64::new(0);
        std::thread::scope(|sc| {
            for _ in 0..8 {
                sc.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= paths.len() {
                        break;
                    }
                    let data = std::fs::read(&paths[i]).unwrap();
                    let mut out: Vec<Col> = cols.iter().map(|_| Col { canon: 0, other: 0, values: Vec::new() }).collect();
                    let mut lines = 0u64;
                    for line in data.split(|&b| b == b'\n').skip(1) {
                        if line.is_empty() {
                            continue;
                        }
                        lines += 1;
                        let f: Vec<&[u8]> = line.split(|&b| b == b',').collect();
                        if f.len() != fields {
                            continue;
                        }
                        for (c, &(at, prefix)) in cols.iter().enumerate() {
                            let mut v = f[at];
                            if v.len() >= 2 && v[0] == b'"' && v[v.len() - 1] == b'"' {
                                v = &v[1..v.len() - 1];
                            }
                            match decode_slice(v, prefix) {
                                Some(y) => {
                                    out[c].canon += 1;
                                    out[c].values.push(y);
                                }
                                None => out[c].other += 1,
                            }
                        }
                    }
                    lines_total.fetch_add(lines, std::sync::atomic::Ordering::Relaxed);
                    results.lock().unwrap()[i] = Some(out);
                });
            }
        });
        let res = results.into_inner().unwrap().into_iter().map(|r| r.unwrap()).collect();
        (lines_total.into_inner(), res)
    }

    fn first_sample(files: &[Vec<Col>], col: usize, n: usize) -> Vec<u64> {
        // The start of the largest file, as an engine would see it.
        let f = files.iter().max_by_key(|f| f[col].values.len()).unwrap();
        f[col].values.iter().copied().take(n).collect()
    }

    fn timed_recover(sample: &[u64], bound: u64) -> (Option<u64>, f64) {
        let t = std::time::Instant::now();
        let k = recover_key(sample, bound);
        (k, t.elapsed().as_secs_f64() * 1e3)
    }

    fn share(c: u64, o: u64) -> String {
        format!("{:.4}% of {}", 100.0 * o as f64 / (c + o).max(1) as f64, c + o)
    }

    fn measure_one(dir: &std::path::Path) {
        println!("== {}", dir.display());
        let (ev_lines, ev) = read_kind(dir, "events", 7, &[(0, b'e'), (1, b'u')]);
        let (ck_lines, ck) = read_kind(dir, "clicks", 4, &[(0, b'k'), (1, b'u')]);
        let (_lk_lines, lk) = read_kind(dir, "links", 2, &[(0, b'u'), (1, b'u')]);
        let sum = |files: &[Vec<Col>], c: usize| {
            files.iter().fold((0, 0), |(a, b), f| (a + f[c].canon, b + f[c].other))
        };
        let (c, o) = sum(&ev, 0);
        println!("noncanonical: events.event_id {}", share(c, o));
        let (c, o) = sum(&ev, 1);
        println!("noncanonical: events.user_id {}", share(c, o));
        let (c, o) = sum(&ck, 0);
        println!("noncanonical: clicks.click_id {}", share(c, o));
        let (c, o) = sum(&ck, 1);
        println!("noncanonical: clicks.user_id {}", share(c, o));
        let (c1, o1) = sum(&lk, 0);
        let (c2, o2) = sum(&lk, 1);
        println!("noncanonical: links.user_id {}", share(c1 + c2, o1 + o2));

        for n in [1024usize, 4096, 16384] {
            let eb = (2 * ev_lines).next_power_of_two();
            let (ek, et) = timed_recover(&first_sample(&ev, 0, n), eb);
            let ub = (32 * ev_lines).next_power_of_two();
            let (uk, ut) = timed_recover(&first_sample(&ev, 1, n), ub);
            let cb = (2 * ck_lines).next_power_of_two();
            let (kk, kt) = timed_recover(&first_sample(&ck, 0, n), cb);
            println!(
                "sample {n}: event key {ek:x?} {et:.2} ms (bound 2^{}), user key {uk:x?} {ut:.2} ms (bound 2^{}), click key {kk:x?} {kt:.2} ms (bound 2^{})",
                eb.trailing_zeros(),
                ub.trailing_zeros(),
                cb.trailing_zeros()
            );
        }
        let ek = recover_key(&first_sample(&ev, 0, 16384), (2 * ev_lines).next_power_of_two()).unwrap();
        let uk = recover_key(&first_sample(&ev, 1, 16384), (32 * ev_lines).next_power_of_two()).unwrap();
        let kk = recover_key(&first_sample(&ck, 0, 16384), (2 * ck_lines).next_power_of_two()).unwrap();

        // Dense counters: events and clicks.
        for (name, files, key, lines) in [("event", &ev, ek, ev_lines), ("click", &ck, kk, ck_lines)] {
            let mut max = 0u64;
            let mut count = 0u64;
            let mut contiguous = 0;
            let mut spans = Vec::new();
            for f in files.iter() {
                let (mut lo, mut hi) = (u64::MAX, 0u64);
                for &y in &f[0].values {
                    let v = unscramble(y, key);
                    lo = lo.min(v);
                    hi = hi.max(v);
                    count += 1;
                }
                max = max.max(hi);
                if !f[0].values.is_empty() {
                    let mut d: Vec<u64> = f[0].values.iter().map(|&y| unscramble(y, key)).collect();
                    d.sort_unstable();
                    d.dedup();
                    if hi - lo + 1 == d.len() as u64 {
                        contiguous += 1;
                    }
                    spans.push((hi - lo + 1) as f64 / (max.max(1)) as f64);
                }
            }
            if max > 1 << 36 {
                println!("{name}: max {max} too large, key wrong?");
                continue;
            }
            let mut seen = vec![false; max as usize + 1];
            for f in files.iter() {
                for &y in &f[0].values {
                    seen[unscramble(y, key) as usize] = true;
                }
            }
            let distinct = seen.iter().filter(|&&b| b).count() as u64;
            let mut gaps = 0;
            let mut longest = 0;
            let mut run = 0;
            for &b in &seen {
                if b {
                    run = 0;
                } else {
                    gaps += 1;
                    run += 1;
                    longest = longest.max(run);
                }
            }
            println!(
                "{name} numbers: lines {lines}, canonical ids {count}, distinct {distinct}, max {max}, gaps in 0..=max {gaps} (longest run {longest}), files {} contiguous {contiguous}, min file span/max {:.3}",
                files.len(),
                spans.iter().cloned().fold(f64::MAX, f64::min)
            );
        }

        // Users.
        for (name, files, cols) in [("events.user", &ev, &[1usize][..]), ("clicks.user", &ck, &[1][..]), ("links", &lk, &[0, 1][..])] {
            let (mut maxp, mut maxa, mut n, mut hi, mut nonzero_alias) = (0u64, 0u64, 0u64, 0u64, 0u64);
            let mut hi_examples = Vec::new();
            for f in files.iter() {
                for &c in cols {
                    for &y in &f[c].values {
                        let v = unscramble(y, uk);
                        n += 1;
                        if v >= 1 << 39 {
                            hi += 1;
                            if hi_examples.len() < 8 && !hi_examples.contains(&(v - (1 << 39))) {
                                hi_examples.push(v - (1 << 39));
                            }
                            continue;
                        }
                        maxp = maxp.max(v >> 9);
                        maxa = maxa.max(v & 511);
                        nonzero_alias += u64::from(v & 511 != 0);
                    }
                }
            }
            println!(
                "{name}: ids {n}, max person {maxp}, max alias {maxa}, alias>0 {:.1}%, at or above 2^39: {hi} ({:.5}%) kiosk offsets {hi_examples:?}",
                100.0 * nonzero_alias as f64 / n.max(1) as f64,
                100.0 * hi as f64 / n.max(1) as f64
            );
        }
        println!("keys: event {ek:010x} user {uk:010x} click {kk:010x}");
    }
}
