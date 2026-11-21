//! Fast field parsers. Each has a narrow fast path for the shape the data
//! almost always has, and hands everything else to the plain reference
//! parser in common.rs, so the two cannot disagree on odd input: the fast
//! path only ever accepts, it never rejects on its own.
//!
//! All pointer-taking functions may read up to 16 bytes past the field.

use crate::common::*;

pub const BAD: i64 = i64::MIN;

#[inline(always)]
pub unsafe fn load64(p: *const u8) -> u64 {
    u64::from_le(std::ptr::read_unaligned(p as *const u64))
}

#[inline(always)]
pub unsafe fn load32(p: *const u8) -> u32 {
    u32::from_le(std::ptr::read_unaligned(p as *const u32))
}

const ZEROS8: u64 = 0x3030_3030_3030_3030;

/// All eight bytes are ASCII digits.
#[inline(always)]
pub fn all_digits(v: u64) -> bool {
    (v.wrapping_add(0x4646_4646_4646_4646) | v.wrapping_sub(ZEROS8)) & 0x8080_8080_8080_8080 == 0
}

/// Eight ASCII digits, first digit in the low byte, to their value.
#[inline(always)]
pub fn parse8(v: u64) -> u64 {
    let v = (v & 0x0F0F_0F0F_0F0F_0F0F).wrapping_mul(2561) >> 8;
    let v = (v & 0x00FF_00FF_00FF_00FF).wrapping_mul(6_553_601) >> 16;
    (v & 0x0000_FFFF_0000_FFFF).wrapping_mul(42_949_672_960_001) >> 32
}

/// The digits of a field of 1 to 8 bytes, left-padded with '0' to a full
/// word. Bytes past the field are shifted out.
#[inline(always)]
unsafe fn padded_digits(p: *const u8, len: usize) -> u64 {
    debug_assert!((1..=8).contains(&len));
    let shift = 8 * (8 - len as u32);
    (load64(p) << shift) | (ZEROS8 >> 1 >> (8 * len as u32 - 1))
}

/// Remembers the last two `YYYY-MM-` prefixes seen: the day number before
/// the first of the month and the month's length. A window touches one or
/// two months, so nearly every timestamp hits.
#[repr(C, align(16))]
pub struct DateCache {
    /// Upper bounds for `[DD - 1, HH, MM, SS]` as 16-bit lanes.
    lim: [[u16; 8]; 2],
    key: [u64; 2],
    /// Milliseconds at the start of the day before the month's first day.
    base_ms: [i64; 2],
    next: usize,
}

impl DateCache {
    /// Both slots start out holding real months. A slot is trusted whenever
    /// its key matches, and every eight bytes are a possible field prefix,
    /// so there is no key that could stand for "empty": a zeroed slot would
    /// accept eight NUL bytes as a month whose only day is the 1st.
    pub fn new() -> Self {
        let mut cache = DateCache { lim: [[0; 8]; 2], key: [0; 2], base_ms: [0; 2], next: 0 };
        for month in [b"1970-01-", b"1970-02-"] {
            let filled = cache.fill(u64::from_le_bytes(*month));
            assert!(filled);
        }
        cache
    }

    #[cold]
    #[inline(never)]
    fn fill(&mut self, a: u64) -> bool {
        let b = a.to_le_bytes();
        if b[4] != b'-' || b[7] != b'-' {
            return false;
        }
        if !(b[0..4].iter().all(u8::is_ascii_digit) && b[5..7].iter().all(u8::is_ascii_digit)) {
            return false;
        }
        let year = b[0..4].iter().fold(0i64, |acc, &c| acc * 10 + i64::from(c - b'0'));
        let month = i64::from(b[5] - b'0') * 10 + i64::from(b[6] - b'0');
        if !(1..=12).contains(&month) {
            return false;
        }
        let slot = self.next;
        self.next ^= 1;
        self.key[slot] = a;
        self.base_ms[slot] = (days_from_civil(year, month, 1) - 1) * DAY_MS;
        self.lim[slot] = [days_in_month(year, month) as u16 - 1, 23, 59, 59, 0, 0, 0, 0];
        true
    }
}

/// `YYYY-MM-DDTHH:MM:SS` at `p` as milliseconds, or `BAD`. With `need_z`
/// the next byte must be `Z` as well (form F1).
#[cfg(not(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3")))]
#[inline(always)]
pub unsafe fn civil19(p: *const u8, cache: &mut DateCache, need_z: bool) -> i64 {
    let a = load64(p);
    if a != cache.key[0] && a != cache.key[1] && !cache.fill(a) {
        return BAD;
    }
    let slot = usize::from(a != cache.key[0]);
    let base_ms = cache.base_ms[slot];
    let dim = u32::from(cache.lim[slot][0]) + 1;
    // "DDTHH:MM": turn the two separators into '0', then all must be digits.
    const PAT: u64 = ((b'T' ^ b'0') as u64) << 16 | ((b':' ^ b'0') as u64) << 40;
    const SEP: u64 = 0xFF << 16 | 0xFF << 40;
    let x = load64(p.add(8)) ^ PAT;
    let tail = load32(p.add(16));
    let c = u64::from(tail & 0x00FF_FFFF) ^ u64::from(b':' ^ b'0');
    let y = c | (ZEROS8 << 24);
    let d = x.wrapping_sub(ZEROS8);
    let dd = ((d & 0xFF) * 10 + ((d >> 8) & 0xFF)) as u32;
    let hh = (((d >> 24) & 0xFF) * 10 + ((d >> 32) & 0xFF)) as u32;
    let mm = (((d >> 48) & 0xFF) * 10 + (d >> 56)) as u32;
    let ss = ((((c >> 8) & 0xFF) * 10 + ((c >> 16) & 0xFF)) as u32).wrapping_sub(528);
    let ok = all_digits(x)
        & all_digits(y)
        & (x & SEP == ZEROS8 & SEP)
        & (c & 0xFF == u64::from(b'0'))
        & (dd.wrapping_sub(1) < dim)
        & (hh < 24)
        & (mm < 60)
        & (ss < 60)
        & (!need_z | (tail >> 24 == u32::from(b'Z')));
    if !ok {
        return BAD;
    }
    base_ms + (i64::from(dd) * 86_400 + i64::from(hh * 3_600 + mm * 60 + ss)) * 1_000
}

/// The same contract with SSE: one 16-byte load covers `-MM-DDTHH:MM:SS`
/// and the byte after. XOR with the expected layout leaves 0 at correct
/// separators and 0..=9 at digits; one unsigned max checks all of them.
/// The digit pairs then become `[DD, HH, MM, SS]` with one multiply-add,
/// are range-checked as a vector, and folded to seconds with another.
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"))]
#[inline(always)]
pub unsafe fn civil19(p: *const u8, cache: &mut DateCache, need_z: bool) -> i64 {
    use core::arch::x86_64::*;
    let a = load64(p);
    if a != cache.key[0] && a != cache.key[1] && !cache.fill(a) {
        return BAD;
    }
    let slot = usize::from(a != cache.key[0]);
    let base_ms = *cache.base_ms.get_unchecked(slot);
    let lim = _mm_load_si128(cache.lim.as_ptr().add(slot) as *const __m128i);
    // Bytes 4..20 of the field: - M M - D D T H H : M M : S S ?
    let v = _mm_loadu_si128(p.add(4) as *const __m128i);
    let expect = _mm_setr_epi8(
        0, 0, 0, 0, 0x30, 0x30, b'T' as i8, 0x30, 0x30, b':' as i8, 0x30, 0x30, b':' as i8, 0x30, 0x30, b'Z' as i8,
    );
    // The first four bytes are covered by the cache key; the last one only
    // matters for F1.
    let last = if need_z { 0 } else { -1 };
    let limit = _mm_setr_epi8(-1, -1, -1, -1, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, last);
    let x = _mm_xor_si128(v, expect);
    let ok_bytes = _mm_cmpeq_epi8(_mm_max_epu8(x, limit), limit);
    let digits = _mm_shuffle_epi8(x, _mm_setr_epi8(4, 5, 7, 8, 10, 11, 13, 14, -1, -1, -1, -1, -1, -1, -1, -1));
    let pairs = _mm_maddubs_epi16(digits, _mm_setr_epi8(10, 1, 10, 1, 10, 1, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0));
    // [DD - 1, HH, MM, SS] must not exceed the cached limits (unsigned, so
    // day 0 wraps and fails).
    let shifted = _mm_sub_epi16(pairs, _mm_setr_epi16(1, 0, 0, 0, 0, 0, 0, 0));
    let ok_range = _mm_cmpeq_epi16(_mm_min_epu16(shifted, lim), shifted);
    if _mm_movemask_epi8(_mm_and_si128(ok_bytes, ok_range)) != 0xFFFF {
        return BAD;
    }
    let sums = _mm_madd_epi16(pairs, _mm_setr_epi16(24, 1, 60, 1, 0, 0, 0, 0));
    let both = _mm_cvtsi128_si64(sums) as u64;
    let hours = both & 0xFFFF_FFFF;
    let seconds = both >> 32;
    base_ms + ((hours * 3_600 + seconds) * 1_000) as i64
}

/// Any `ts` or `ingest_ts` field: epoch milliseconds, `BAD` when it is not
/// a timestamp, `i64::MAX` for a digits-only value too long to matter.
#[inline(always)]
pub unsafe fn parse_ts(p: *const u8, len: usize, cache: &mut DateCache) -> i64 {
    if len == 20 {
        let v = civil19(p, cache, true);
        if v != BAD {
            return v;
        }
    } else if len == 13 {
        let hi = load64(p);
        let lo = (load64(p.add(8)) << 24) | 0x30_3030;
        if all_digits(hi) & all_digits(lo) {
            return (parse8(hi) * 100_000 + parse8(lo)) as i64;
        }
    } else if len == 25 {
        let v = civil19(p, cache, false);
        // Bytes 19..25: sign, "HH:MM".
        let t = load64(p.add(17)) >> 16;
        let sign = (t & 0xFF) as u8;
        let x = (t >> 8) ^ (u64::from(b':' ^ b'0') << 16) | (ZEROS8 << 40);
        let d = x.wrapping_sub(ZEROS8);
        let hh = (d & 0xFF) * 10 + ((d >> 8) & 0xFF);
        let mm = ((d >> 24) & 0xFF) * 10 + ((d >> 32) & 0xFF);
        let ok = (v != BAD)
            & ((sign == b'+') | (sign == b'-'))
            & all_digits(x)
            & ((x >> 16) & 0xFF == u64::from(b'0'))
            & (hh < 24)
            & (mm < 60);
        if ok {
            let offset = ((hh * 60 + mm) * 60_000) as i64;
            return if sign == b'+' { v - offset } else { v + offset };
        }
    }
    parse_ts_slow(std::slice::from_raw_parts(p, len))
}

#[cold]
#[inline(never)]
fn parse_ts_slow(b: &[u8]) -> i64 {
    parse_ts_naive(b).unwrap_or(BAD)
}

/// A purchase price: its value, or 0 when it is not valid.
#[inline(always)]
pub unsafe fn parse_price(p: *const u8, len: usize) -> u64 {
    if (1..=8).contains(&len) {
        let w = padded_digits(p, len);
        if all_digits(w) {
            let v = parse8(w);
            return if v <= MAX_PRICE { v } else { 0 };
        }
        return 0;
    }
    parse_price_naive(std::slice::from_raw_parts(p, len)).unwrap_or(0)
}

/// `<prefix><n>` with `n` a canonical decimal (no leading zero, 1 to 7
/// digits): returns `n`. `None` for any other shape, which the caller looks
/// up by exact bytes instead.
#[inline(always)]
pub unsafe fn canonical_number(p: *const u8, len: usize) -> Option<u32> {
    if !(1..=7).contains(&len) || *p == b'0' {
        return None;
    }
    let w = padded_digits(p, len);
    if !all_digits(w) {
        return None;
    }
    Some(parse8(w) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fast(text: &[u8], cache: &mut DateCache) -> i64 {
        let mut buf = text.to_vec();
        // Hostile padding: digits and separators after the field must not leak in.
        buf.extend_from_slice(b"9Z:59+0101234567890123456789");
        unsafe { parse_ts(buf.as_ptr(), text.len(), cache) }
    }

    fn check(text: &[u8], cache: &mut DateCache) {
        let want = parse_ts_naive(text).unwrap_or(BAD);
        assert_eq!(fast(text, cache), want, "{:?}", String::from_utf8_lossy(text));
    }

    #[test]
    fn rules_examples() {
        let mut cache = DateCache::new();
        let t = 1_791_071_890_000;
        assert_eq!(fast(b"2026-10-03T23:58:10Z", &mut cache), t);
        assert_eq!(fast(b"2026-10-03T19:58:10-04:00", &mut cache), t);
        assert_eq!(fast(b"1791071890000", &mut cache), t);
        assert_eq!(fast(b"0001791071890000", &mut cache), t);
        assert_eq!(fast(b"86400000", &mut cache), 86_400_000);
        assert_eq!(fast(b"2026-10-04T01:30:00+02:00", &mut cache), fast(b"2026-10-03T23:30:00Z", &mut cache));
        assert_eq!(fast(b"2026-10-03T23:58:10-00:00", &mut cache), t);
        assert_eq!(fast(b"2026-10-03T23:58:10+23:59", &mut cache), t - (23 * 60 + 59) * 60_000);
        assert_eq!(fast(b"2028-02-29T00:00:00Z", &mut cache), days_from_civil(2028, 2, 29) * DAY_MS);
        assert_eq!(fast(b"123456789012345678901", &mut cache), i64::MAX);
        for bad in [
            "",
            "2026-10-03T23:58:10.000Z",
            "2026-10-03T23:58:10z",
            "2026-10-03t23:58:10Z",
            "2026-10-03 23:58:10Z",
            "2026-10-03T23:58:10",
            "1791071890000.0",
            "1.79e12",
            "-5",
            "2026-13-03T23:58:10Z",
            "2026-02-30T23:58:10Z",
            "2027-02-29T23:58:10Z",
            "2026-10-03T24:58:10Z",
            "2026-10-03T23:58:60Z",
            "2026-10-03T23:60:10Z",
            "2026-10-03T23:58:10+24:00",
            "2026-10-03T23:58:10+05:60",
            "2026-10-03T23:58:10+0200",
            "2026-10-03T23:58:10+02:00Z",
            "2026-10-03T23:58:10 02:00",
            "2026-10-00T23:58:10Z",
            "2026-00-03T23:58:10Z",
            "2026-10-32T23:58:10Z",
            "2026-04-31T23:58:10Z",
            "2026-10-03T23:58:10Z\r",
            "1791071890000\r",
            " 1791071890000",
            "179107189000a",
            "\u{0662}\u{0665}\u{0660}",
            "\u{ff12}026-10-03T23:58:10Z",
            "2026/10/03T23:58:10Z",
            "2026-10-03T23-58-10Z",
            "2026-10-03T23:58:1Z",
            "+026-10-03T23:58:10Z",
            "2026-1 -03T23:58:10Z",
        ] {
            assert_eq!(fast(bad.as_bytes(), &mut cache), BAD, "{bad:?}");
            assert_eq!(parse_ts_naive(bad.as_bytes()), None, "{bad:?}");
        }
    }

    /// A fresh cache knows no month that is not one. Eight NUL bytes used
    /// to match its zeroed keys and parse as 1970-01-02.
    #[test]
    fn fresh_cache_rejects_odd_prefixes() {
        let tails: [&[u8]; 4] = [b"01T00:00:00Z", b"01T00:00:00+00:00", b"00T00:00:00Z", b"02T23:59:59-01:30"];
        let mut prefixes: Vec<[u8; 8]> = vec![[0; 8], [0xFF; 8], [b' '; 8], [b'0'; 8], [b'-'; 8], *b"0000-00-", *b"\0\0\0\0-\0\0-"];
        for byte in [0u8, 1, b'0', b'-', 0x80] {
            for at in 0..8 {
                let mut p = *b"1970-01-";
                p[at] = byte;
                prefixes.push(p);
                let mut p = [0u8; 8];
                p[at] = byte;
                prefixes.push(p);
            }
        }
        for prefix in prefixes {
            for tail in tails {
                let text = [&prefix[..], tail].concat();
                // A new cache each time: the first lookup is the one at risk.
                check(&text, &mut DateCache::new());
                let mut warm = DateCache::new();
                check(b"2026-10-03T23:58:10Z", &mut warm);
                check(&text, &mut warm);
            }
        }
        let nul = b"\0\0\0\0\0\0\0\001T00:00:00Z";
        assert_eq!(fast(nul, &mut DateCache::new()), BAD);
        // The months a cache starts with are real ones.
        assert_eq!(fast(b"1970-01-01T00:00:00Z", &mut DateCache::new()), 0);
        assert_eq!(fast(b"1970-02-28T00:00:01Z", &mut DateCache::new()), 58 * DAY_MS + 1000);
        assert_eq!(fast(b"1970-02-29T00:00:00Z", &mut DateCache::new()), BAD);
    }

    #[test]
    fn every_day_of_four_years() {
        let mut cache = DateCache::new();
        for day in days_from_civil(2025, 1, 1)..days_from_civil(2029, 1, 1) {
            let (y, m, d) = civil_from_days(day);
            let text = format!("{y:04}-{m:02}-{d:02}T12:34:56Z");
            assert_eq!(fast(text.as_bytes(), &mut cache), day * DAY_MS + 45_296_000, "{text}");
        }
        // Day numbers one past each month end are rejected.
        for year in [1900i64, 2000, 2024, 2025, 2026, 2027, 2028, 2100] {
            for month in 1..=12 {
                let over = days_in_month(year, month) + 1;
                let text = format!("{year:04}-{month:02}-{over:02}T00:00:00Z");
                check(text.as_bytes(), &mut cache);
                assert_eq!(fast(text.as_bytes(), &mut cache), BAD);
                let last = format!("{year:04}-{month:02}-{:02}T23:59:59+00:00", over - 1);
                check(last.as_bytes(), &mut cache);
                assert_ne!(fast(last.as_bytes(), &mut cache), BAD);
            }
        }
    }

    /// Valid timestamps of all three forms with single-byte mutations: the
    /// fast parser and the reference agree on every one.
    #[test]
    fn mutations_match_reference() {
        let mut cache = DateCache::new();
        let mut state = 11;
        let subs: &[u8] = b"0123456789-:TZtz+ .eE\r,\"/\x00\xd9\xa2";
        for _ in 0..200_000 {
            let day = days_from_civil(2024, 12, 20) + (rng(&mut state) % 1500) as i64;
            let (y, m, d) = civil_from_days(day);
            let (hh, mi, ss) = (rng(&mut state) % 24, rng(&mut state) % 60, rng(&mut state) % 60);
            let mut text = match rng(&mut state) % 4 {
                0 => format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mi:02}:{ss:02}Z"),
                1 => format!(
                    "{y:04}-{m:02}-{d:02}T{hh:02}:{mi:02}:{ss:02}{}{:02}:{:02}",
                    if rng(&mut state) % 2 == 0 { '+' } else { '-' },
                    rng(&mut state) % 26,
                    rng(&mut state) % 64
                ),
                2 => format!("{}", day * DAY_MS + (rng(&mut state) % DAY_MS as u64) as i64),
                _ => format!("{:0width$}", rng(&mut state) % 10_000_000_000_000_000, width = (rng(&mut state) % 22) as usize),
            }
            .into_bytes();
            check(&text, &mut cache);
            for _ in 0..3 {
                if text.is_empty() {
                    break;
                }
                let at = (rng(&mut state) % text.len() as u64) as usize;
                match rng(&mut state) % 8 {
                    0 => {
                        text.remove(at);
                    }
                    1 => text.insert(at, subs[(rng(&mut state) % subs.len() as u64) as usize]),
                    _ => text[at] = subs[(rng(&mut state) % subs.len() as u64) as usize],
                }
                check(&text, &mut cache);
            }
        }
    }

    #[test]
    fn prices_match_reference() {
        let mut state = 5;
        let mut cases: Vec<Vec<u8>> = [
            "", "0", "00", "1", "0250", "250", "10000000", "10000001", "250000000", "19.99", "-5", "+500", "1e3",
            "1000\r", " 1", "1 ", "00000000000000000250", "000000010000000", "99999999", "100000000",
            "\u{ff12}\u{ff15}\u{ff10}", "\u{0662}\u{0665}\u{0660}", "12a", "a12", "1234567", "12345678", "123456789",
        ]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
        for _ in 0..100_000 {
            let len = rng(&mut state) % 12;
            cases.push((0..len).map(|_| b"0123456789.-+ e"[(rng(&mut state) % 22).min(14) as usize]).collect());
        }
        for case in cases {
            let mut buf = case.clone();
            buf.extend_from_slice(b"1234567890123456");
            let got = unsafe { parse_price(buf.as_ptr(), case.len()) };
            assert_eq!(got, parse_price_naive(&case).unwrap_or(0), "{:?}", String::from_utf8_lossy(&case));
        }
    }

    #[test]
    fn canonical_numbers() {
        for (text, want) in [
            ("1", Some(1)),
            ("9999999", Some(9_999_999)),
            ("50000", Some(50_000)),
            ("0", None),
            ("01", None),
            ("12345678", None),
            ("", None),
            ("1a", None),
            (" 1", None),
            ("1 ", None),
            ("+1", None),
        ] {
            let mut buf = text.as_bytes().to_vec();
            buf.extend_from_slice(b"1234567890123456");
            assert_eq!(unsafe { canonical_number(buf.as_ptr(), text.len()) }, want, "{text:?}");
        }
    }
}
