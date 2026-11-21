//! Fast field parsers. Each has a narrow fast path for the shape the data
//! almost always has, and hands everything else to the plain reference
//! parser in common.rs, so the two cannot disagree on odd input: the fast
//! path only ever accepts, it never rejects on its own.
//!
//! All pointer-taking functions may read up to 16 bytes past the field,
//! and `parse_ts_ingest` reads the 20 bytes from the start of `ingest_ts`
//! whatever its length (20 past an empty one).

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

/// The digit values (0..=9 a byte) of a field of 1 to 8 bytes, left-padded
/// with zeros to a full word in `parse_values` order, or `None` unless
/// every byte of the field is an ASCII digit.
///
/// The `'0'` is taken off the whole word before the shift: a borrow only
/// runs from a byte to the one after it, so the field's bytes see no
/// borrow from the bytes past it (which the shift drops), and a byte below
/// `'0'` wraps above 9 itself whatever it does to the next.
#[inline(always)]
unsafe fn digit_values(p: *const u8, len: usize) -> Option<u64> {
    debug_assert!((1..=8).contains(&len));
    let x = load64(p).wrapping_sub(ZEROS8) << (8 * (8 - len as u32));
    // Bytes of at most 9 stay below 0x80 after adding 0x76.
    if (x.wrapping_add(0x7676_7676_7676_7676) | x) & 0x8080_8080_8080_8080 != 0 {
        return None;
    }
    Some(x)
}

/// `parse8` of eight digit values (0..=9 a byte) rather than characters.
#[inline(always)]
fn parse_values(v: u64) -> u64 {
    let v = v.wrapping_mul(2561) >> 8;
    let v = (v & 0x00FF_00FF_00FF_00FF).wrapping_mul(6_553_601) >> 16;
    (v & 0x0000_FFFF_0000_FFFF).wrapping_mul(42_949_672_960_001) >> 32
}

/// One cached month (or, in slots 2 and 3, "no calendar"): 32 bytes, so a
/// slot's byte offset addresses its key, limits and base at once.
#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct Slot {
    /// Upper bounds for `[DD - 1, HH, MM, SS]` as 16-bit lanes.
    lim: [u16; 8],
    /// The `YYYY-MM-` this slot holds.
    key: u64,
    /// Milliseconds at the start of the day before the month's first day.
    base_ms: i64,
}

const SLOT_BYTES: usize = std::mem::size_of::<Slot>();
const _: () = assert!(SLOT_BYTES == 32);

/// Byte offset of the "no calendar" slots from the month slots.
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"))]
const NOCAL: usize = 2 * SLOT_BYTES;

/// Remembers two `YYYY-MM-` prefixes: the day number before the first of
/// the month and the month's length. A month's slot is the parity of its
/// last digit, and two months in a row always differ in it (12 and 01
/// too), so dates that span two months never evict each other. A thread's
/// dates lie within a few weeks, so nearly every timestamp hits.
///
/// Slots 2 and 3 are not months. They stand for "no calendar": no bounds
/// and no base, which is what a digits-only timestamp needs when it runs
/// through the same code as the calendar forms (see `parse_ts`).
#[repr(C, align(32))]
pub struct DateCache {
    slots: [Slot; 4],
}

/// The byte offset of the slot for the month `a` (a field's first eight
/// bytes): bit 0 of byte 6, the month's last digit, moved to bit 5.
#[inline(always)]
fn slot_off(a: u64) -> usize {
    ((a >> 43) & SLOT_BYTES as u64) as usize
}

impl DateCache {
    /// Both slots start out holding real months. A slot is trusted whenever
    /// its key matches, and every eight bytes are a possible field prefix,
    /// so there is no key that could stand for "empty": a zeroed slot would
    /// accept eight NUL bytes as a month whose only day is the 1st.
    pub fn new() -> Self {
        let month = Slot { lim: [0; 8], key: 0, base_ms: 0 };
        let nocal = Slot { lim: [u16::MAX; 8], key: 0, base_ms: 0 };
        let mut cache = DateCache { slots: [month, month, nocal, nocal] };
        for month in [b"1970-01-", b"1970-02-"] {
            let filled = cache.fill(u64::from_le_bytes(*month));
            assert!(filled);
        }
        assert!(cache.slots[0].key != cache.slots[1].key);
        cache
    }

    /// The slot at byte offset `off`: 0 or 32 for a month, plus `NOCAL`
    /// for the slots without a calendar.
    #[inline(always)]
    unsafe fn at(&self, off: usize) -> *const Slot {
        debug_assert!(off % SLOT_BYTES == 0 && off < 4 * SLOT_BYTES);
        (self.slots.as_ptr() as *const u8).add(off) as *const Slot
    }

    /// Puts the month `a` in its slot; false, with no change, when `a` is
    /// not a month.
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
        self.slots[slot_off(a) / SLOT_BYTES] = Slot {
            lim: [days_in_month(year, month) as u16 - 1, 23, 59, 59, 0, 0, 0, 0],
            key: a,
            base_ms: (days_from_civil(year, month, 1) - 1) * DAY_MS,
        };
        true
    }

    /// The slot of the month `a` (the field's first eight bytes), filled
    /// first if need be; `None` when `a` is not a month.
    #[inline(always)]
    fn slot(&mut self, a: u64) -> Option<usize> {
        let slot = slot_off(a) / SLOT_BYTES;
        if a == self.slots[slot].key || self.fill(a) {
            return Some(slot);
        }
        None
    }
}

/// `YYYY-MM-DDTHH:MM:SS` at `p` as milliseconds, or `None`. With `need_z`
/// the next byte must be `Z` as well (form F1).
#[cfg(not(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3")))]
#[inline(always)]
pub unsafe fn civil19(p: *const u8, cache: &mut DateCache, need_z: bool) -> Option<i64> {
    let a = load64(p);
    let slot = cache.slot(a)?;
    let base_ms = cache.slots[slot].base_ms;
    let dim = u32::from(cache.slots[slot].lim[0]) + 1;
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
        return None;
    }
    Some(base_ms + (i64::from(dd) * 86_400 + i64::from(hh * 3_600 + mm * 60 + ss)) * 1_000)
}

/// The same contract with SSE: one 16-byte load covers `-MM-DDTHH:MM:SS`
/// and the byte after. XOR with the expected layout leaves 0 at correct
/// separators and 0..=9 at digits; one unsigned max checks all of them.
/// The digit pairs then become `[DD, HH, MM, SS]` with one multiply-add,
/// are range-checked as a vector, and folded to seconds with another.
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"))]
#[inline(always)]
pub unsafe fn civil19(p: *const u8, cache: &mut DateCache, need_z: bool) -> Option<i64> {
    use core::arch::x86_64::*;
    let a = load64(p);
    let slot = cache.slot(a)?;
    let s = cache.slots.get_unchecked(slot);
    let base_ms = s.base_ms;
    let lim = _mm_load_si128(s.lim.as_ptr() as *const __m128i);
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
        return None;
    }
    let sums = _mm_madd_epi16(pairs, _mm_setr_epi16(24, 1, 60, 1, 0, 0, 0, 0));
    let both = _mm_cvtsi128_si64(sums) as u64;
    let hours = both & 0xFFFF_FFFF;
    let seconds = both >> 32;
    Some(base_ms + ((hours * 3_600 + seconds) * 1_000) as i64)
}

/// An `ingest_ts` field: epoch milliseconds, or `i64::MAX` when it is not a
/// timestamp, so that such a copy sorts last. The rules promise form F1,
/// which is checked in full here; any other field goes through the general
/// reader, as it does in the reference.
#[inline(always)]
pub unsafe fn parse_ingest(p: *const u8, len: usize, cache: &mut DateCache) -> i64 {
    if len == 20 {
        if let Some(v) = civil19(p, cache, true) {
            return v;
        }
    }
    parse_ingest_cold(p, len, cache)
}

#[cold]
#[inline(never)]
unsafe fn parse_ingest_cold(p: *const u8, len: usize, cache: &mut DateCache) -> i64 {
    match parse_ts_general(p, len, cache) {
        BAD => i64::MAX,
        v => v,
    }
}

/// What tells the three timestamp forms apart in `parse_ts`: where the 16
/// bytes to convert start, what they must look like, and how their digits
/// add up. One entry per form, picked by the field length.
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"))]
#[repr(C, align(64))]
struct Form {
    /// XOR with this leaves 0 at a correct separator and 0..=9 at a digit.
    expect: [u8; 16],
    /// The largest value each byte may then have (255: not looked at).
    limit: [u8; 16],
    /// Moves the digits into place for the pairwise multiply-adds.
    shuf: [i8; 16],
    /// Weights that fold eight two-digit numbers into four numbers.
    fold4: [i16; 8],
    /// Weights that fold those into the two halves of the value.
    fold2: [i16; 8],
    /// value = low half * scale + high half + base of the cache slot.
    scale: u64,
    /// Offset of the 16 bytes in the field.
    off: usize,
    /// All ones when the first eight bytes must be a cached `YYYY-MM-`.
    keymask: u64,
    /// Added to the slot offset: `NOCAL` moves it to the slots without a
    /// calendar.
    slot: usize,
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"))]
const fn civil_form(last: u8) -> Form {
    Form {
        // Bytes 4..20 of the field: - M M - D D T H H : M M : S S, then Z.
        expect: [0, 0, 0, 0, b'0', b'0', b'T', b'0', b'0', b':', b'0', b'0', b':', b'0', b'0', b'Z'],
        // The first four are covered by the cache key.
        limit: [255, 255, 255, 255, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, last],
        shuf: [4, 5, 7, 8, 10, 11, 13, 14, -1, -1, -1, -1, -1, -1, -1, -1],
        // [DD, HH, MM, SS] to [hours, seconds] to seconds.
        fold4: [24, 1, 60, 1, 0, 0, 0, 0],
        fold2: [3600, 1, 0, 0, 0, 0, 0, 0],
        scale: 1000,
        off: 4,
        keymask: u64::MAX,
        slot: 0,
    }
}

/// Index 0: thirteen digits (F3 as the generator writes it). 1: F1. 2: F2,
/// whose byte 19 is the sign of the offset and is checked with the offset.
/// 3: no form. It always fails: its digits are all left out, so the day
/// lane is 0 - 1, above the day limit of any month, and its slot is a
/// month's.
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"))]
const FORM_LIST: [Form; 4] = [
    Form {
        expect: [b'0'; 16],
        limit: [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 255, 255, 255],
        // Right-aligned behind three zeros: sixteen digits.
        shuf: [-1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        fold4: [100, 1, 100, 1, 100, 1, 100, 1],
        fold2: [10000, 1, 10000, 1, 0, 0, 0, 0],
        scale: 100_000_000,
        off: 0,
        keymask: 0,
        slot: NOCAL,
    },
    civil_form(0),
    civil_form(255),
    Form {
        expect: [0; 16],
        limit: [0; 16],
        shuf: [-1; 16],
        fold4: [0; 8],
        fold2: [0; 8],
        scale: 0,
        off: 0,
        keymask: u64::MAX,
        slot: 0,
    },
];

#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"))]
static FORMS: [Form; 4] = FORM_LIST;

/// Field length (mod 32) to the byte offset of its entry in `FORMS`, the
/// "no form" entry for lengths that have none.
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"))]
static LEN_FORM: [u16; 32] = {
    let size = std::mem::size_of::<Form>();
    let mut t = [(3 * size) as u16; 32];
    t[13] = 0;
    t[20] = size as u16;
    t[25] = (2 * size) as u16;
    t
};

/// Any `ts` field: epoch milliseconds, `BAD` when it is not a timestamp,
/// `i64::MAX` for a digits-only value too long to matter.
///
/// The three forms come in a random mix (58% F1, 32% F3, 10% F2), so a
/// branch on the form is mispredicted on four rows in ten. Here the form
/// only selects constants: all three run the same instructions, and the
/// branches left are the ones that are almost never taken (whatever this
/// path does not recognise goes to `parse_ts_general`) and the F2 offset.
/// The form, the month and the bytes are checked with one branch.
#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"))]
#[inline(always)]
pub unsafe fn parse_ts(p: *const u8, len: usize, cache: &mut DateCache) -> i64 {
    use core::arch::x86_64::*;
    let f = FORMS.as_ptr().byte_add(usize::from(*LEN_FORM.get_unchecked(len & 31)));
    // The month, for the forms that have one: it must be in the cache. A
    // length of 32 or more has no form.
    let a = load64(p);
    let at = slot_off(a);
    let miss = ((a ^ (*cache.at(at)).key) & (*f).keymask) | (len >> 5) as u64;
    let s = cache.at(at + (*f).slot);
    let v = _mm_loadu_si128(p.add((*f).off) as *const __m128i);
    let x = _mm_xor_si128(v, _mm_load_si128((*f).expect.as_ptr() as *const __m128i));
    let limit = _mm_load_si128((*f).limit.as_ptr() as *const __m128i);
    let ok_bytes = _mm_cmpeq_epi8(_mm_max_epu8(x, limit), limit);
    let digits = _mm_shuffle_epi8(x, _mm_load_si128((*f).shuf.as_ptr() as *const __m128i));
    let pairs = _mm_maddubs_epi16(digits, _mm_set1_epi16(0x010A));
    // [DD - 1, HH, MM, SS] must not exceed the month's limits (unsigned, so
    // day 0 wraps and fails). The slots without a calendar allow anything.
    let shifted = _mm_sub_epi16(pairs, _mm_setr_epi16(1, 0, 0, 0, 0, 0, 0, 0));
    let lim = _mm_load_si128((*s).lim.as_ptr() as *const __m128i);
    let ok_range = _mm_cmpeq_epi16(_mm_min_epu16(shifted, lim), shifted);
    let ok = _mm_movemask_epi8(_mm_and_si128(ok_bytes, ok_range)) as u64;
    if (ok ^ 0xFFFF) | miss != 0 {
        return parse_ts_cold(p, len, cache);
    }
    let fours = _mm_madd_epi16(pairs, _mm_load_si128((*f).fold4.as_ptr() as *const __m128i));
    let packed = _mm_packus_epi32(fours, fours);
    let halves = _mm_madd_epi16(packed, _mm_load_si128((*f).fold2.as_ptr() as *const __m128i));
    let both = _mm_cvtsi128_si64(halves) as u64;
    let v = ((both & 0xFFFF_FFFF) * (*f).scale + (both >> 32)) as i64 + (*s).base_ms;
    if len == 25 {
        return match offset_minutes(p) {
            Some(minutes) => v - minutes * 60_000,
            None => parse_ts_cold(p, len, cache),
        };
    }
    v
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3")))]
#[inline(always)]
pub unsafe fn parse_ts(p: *const u8, len: usize, cache: &mut DateCache) -> i64 {
    parse_ts_general(p, len, cache)
}

/// `FORMS` with a second lane that is always form F1: the constants for
/// converting an event's `ts` (low lane, any form) and its `ingest_ts`
/// (high lane) in one pass of 256-bit instructions.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[repr(C, align(64))]
struct Form2 {
    expect: [[u8; 16]; 2],
    limit: [[u8; 16]; 2],
    shuf: [[i8; 16]; 2],
    fold4: [[i16; 8]; 2],
    fold2: [[i16; 8]; 2],
    scale: u64,
    off: usize,
    keymask: u64,
    slot: usize,
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const fn with_ingest(form: usize) -> Form2 {
    let (t, i) = (&FORM_LIST[form], &FORM_LIST[1]);
    Form2 {
        expect: [t.expect, i.expect],
        limit: [t.limit, i.limit],
        shuf: [t.shuf, i.shuf],
        fold4: [t.fold4, i.fold4],
        fold2: [t.fold2, i.fold2],
        scale: t.scale,
        off: t.off,
        keymask: t.keymask,
        slot: t.slot,
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
static FORMS2: [Form2; 4] = [with_ingest(0), with_ingest(1), with_ingest(2), with_ingest(3)];

/// Field length (mod 32) to the byte offset of its entry in `FORMS2`
/// (`LEN_FORM` scaled to the entry size, so no multiply).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
static LEN_FORM2: [u16; 32] = {
    let mut t = [0; 32];
    let mut len = 0;
    while len < 32 {
        let form = LEN_FORM[len] as usize / std::mem::size_of::<Form>();
        t[len] = (form * std::mem::size_of::<Form2>()) as u16;
        len += 1;
    }
    t
};

/// `ts` and `ingest_ts` of one event: what `parse_ts` and `parse_ingest`
/// return for them. Each timestamp takes one 128-bit lane of the same
/// instructions, so the pair costs little more than one of them. The
/// lengths, both months and all bytes are checked with one branch.
///
/// Reads: up to 20 bytes from `tp` and from `ip` whatever the lengths are
/// (they are checked after the loads), and the F2 offset inside `ts`. The only caller passes fields of a
/// line in a file mapping, so both pointers are at most the file's end,
/// and `sys::map_file` puts at least one page of zeros after it.
/// `pair_reader_ingest_lengths` covers `ilen` 0 to 40.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn parse_ts_ingest(
    tp: *const u8,
    tlen: usize,
    ip: *const u8,
    ilen: usize,
    cache: &mut DateCache,
) -> (i64, i64) {
    use core::arch::x86_64::*;
    let f = FORMS2.as_ptr().byte_add(usize::from(*LEN_FORM2.get_unchecked(tlen & 31)));
    // Both months (`ts` only if its form has one) must be in the cache. A
    // `ts` of 32 bytes or more has no form; `ingest_ts` must be F1.
    let (a, b) = (load64(tp), load64(ip));
    let (oa, ob) = (slot_off(a), slot_off(b));
    let miss = ((a ^ (*cache.at(oa)).key) & (*f).keymask) | (b ^ (*cache.at(ob)).key) | ((tlen >> 5) | (ilen ^ 20)) as u64;
    let (sa, sb) = (cache.at(oa + (*f).slot), cache.at(ob));
    let two = |low: *const u8, high: *const u8| -> __m256i {
        _mm256_inserti128_si256(
            _mm256_castsi128_si256(_mm_loadu_si128(low as *const __m128i)),
            _mm_loadu_si128(high as *const __m128i),
            1,
        )
    };
    let v = two(tp.add((*f).off), ip.add(4));
    let x = _mm256_xor_si256(v, _mm256_load_si256((*f).expect.as_ptr() as *const __m256i));
    let limit = _mm256_load_si256((*f).limit.as_ptr() as *const __m256i);
    let ok_bytes = _mm256_cmpeq_epi8(_mm256_max_epu8(x, limit), limit);
    let digits = _mm256_shuffle_epi8(x, _mm256_load_si256((*f).shuf.as_ptr() as *const __m256i));
    let pairs = _mm256_maddubs_epi16(digits, _mm256_set1_epi16(0x010A));
    let shifted = _mm256_sub_epi16(pairs, _mm256_setr_epi16(1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0));
    let lim = two((*sa).lim.as_ptr() as *const u8, (*sb).lim.as_ptr() as *const u8);
    let ok_range = _mm256_cmpeq_epi16(_mm256_min_epu16(shifted, lim), shifted);
    let ok = _mm256_movemask_epi8(_mm256_and_si256(ok_bytes, ok_range)) as u32;
    if u64::from(!ok) | miss != 0 {
        return parse_ts_ingest_cold(tp, tlen, ip, ilen, cache);
    }
    let fours = _mm256_madd_epi16(pairs, _mm256_load_si256((*f).fold4.as_ptr() as *const __m256i));
    let packed = _mm256_packus_epi32(fours, fours);
    let halves = _mm256_madd_epi16(packed, _mm256_load_si256((*f).fold2.as_ptr() as *const __m256i));
    let both = _mm_cvtsi128_si64(_mm256_castsi256_si128(halves)) as u64;
    // The high half of an F1 value is zero.
    let seconds = _mm_cvtsi128_si64(_mm256_extracti128_si256(halves, 1)) as u64;
    let ingest = (seconds * 1000) as i64 + (*sb).base_ms;
    let ts = ((both & 0xFFFF_FFFF) * (*f).scale + (both >> 32)) as i64 + (*sa).base_ms;
    if tlen == 25 {
        match offset_minutes(tp) {
            Some(minutes) => return (ts - minutes * 60_000, ingest),
            None => return parse_ts_ingest_cold(tp, tlen, ip, ilen, cache),
        }
    }
    (ts, ingest)
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
pub unsafe fn parse_ts_ingest(
    tp: *const u8,
    tlen: usize,
    ip: *const u8,
    ilen: usize,
    cache: &mut DateCache,
) -> (i64, i64) {
    (parse_ts(tp, tlen, cache), parse_ingest(ip, ilen, cache))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[cold]
#[inline(never)]
unsafe fn parse_ts_ingest_cold(
    tp: *const u8,
    tlen: usize,
    ip: *const u8,
    ilen: usize,
    cache: &mut DateCache,
) -> (i64, i64) {
    (parse_ts_general(tp, tlen, cache), parse_ingest(ip, ilen, cache))
}

/// The offset of a 25-byte F2 field at `p` (bytes 19..25: a sign, then
/// `HH:MM`) in minutes east of UTC, or `None` when it is not one.
#[allow(dead_code)]
#[inline(always)]
unsafe fn offset_minutes(p: *const u8) -> Option<i64> {
    let t = load64(p.add(17)) >> 16;
    let sign = (t & 0xFF) as u8;
    let x = (t >> 8) ^ (u64::from(b':' ^ b'0') << 16) | (ZEROS8 << 40);
    let d = x.wrapping_sub(ZEROS8);
    let hh = (d & 0xFF) * 10 + ((d >> 8) & 0xFF);
    let mm = ((d >> 24) & 0xFF) * 10 + ((d >> 32) & 0xFF);
    let ok = ((sign == b'+') | (sign == b'-'))
        & all_digits(x)
        & ((x >> 16) & 0xFF == u64::from(b'0'))
        & (hh < 24)
        & (mm < 60);
    if !ok {
        return None;
    }
    let minutes = (hh * 60 + mm) as i64;
    Some(if sign == b'+' { minutes } else { -minutes })
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1", target_feature = "ssse3"))]
#[cold]
#[inline(never)]
unsafe fn parse_ts_cold(p: *const u8, len: usize, cache: &mut DateCache) -> i64 {
    parse_ts_general(p, len, cache)
}

/// Any `ts` or `ingest_ts` field, one form at a time: epoch milliseconds,
/// `BAD` when it is not a timestamp, `i64::MAX` for a digits-only value too
/// long to matter. This is the path that fills the date cache.
#[inline(always)]
pub unsafe fn parse_ts_general(p: *const u8, len: usize, cache: &mut DateCache) -> i64 {
    if len == 20 {
        if let Some(v) = civil19(p, cache, true) {
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
        let ok = ((sign == b'+') | (sign == b'-'))
            & all_digits(x)
            & ((x >> 16) & 0xFF == u64::from(b'0'))
            & (hh < 24)
            & (mm < 60);
        if let (Some(v), true) = (v, ok) {
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
        if let Some(w) = digit_values(p, len) {
            let v = parse_values(w);
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
    Some(parse_values(digit_values(p, len)?) as u32)
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

    fn fast_single(text: &[u8], cache: &mut DateCache) -> i64 {
        let mut buf = text.to_vec();
        buf.extend_from_slice(&[b'5'; 32]);
        unsafe { parse_ts(buf.as_ptr(), text.len(), cache) }
    }

    fn fast(text: &[u8], cache: &mut DateCache) -> i64 {
        let mut buf = text.to_vec();
        // Hostile padding: digits and separators after the field must not leak in.
        buf.extend_from_slice(b"9Z:59+0101234567890123456789");
        let got = unsafe { parse_ts(buf.as_ptr(), text.len(), cache) };
        // The one-form-at-a-time path (ingest_ts, and the fallback of the
        // other) must say the same, whatever follows the field.
        let mut other = text.to_vec();
        other.extend_from_slice(&[0xFF; 32]);
        assert_eq!(unsafe { parse_ts_general(other.as_ptr(), text.len(), cache) }, got, "{:?}", String::from_utf8_lossy(text));
        // And the ingest_ts reader, which says `i64::MAX` for "not one".
        let ingest = unsafe { parse_ingest(other.as_ptr(), text.len(), cache) };
        assert_eq!(ingest, if got == BAD { i64::MAX } else { got }, "{:?}", String::from_utf8_lossy(text));
        // The pair reader, with this field on either side and a partner
        // that is sometimes fine and sometimes not.
        for partner in [&b"2026-10-03T23:58:10Z"[..], b"2026-10-32T23:58:10Z", b"1791071890000", b"2026-11-03T19:58:10-04:00", b""] {
            let want = fast_single(partner, cache);
            let mut padded = partner.to_vec();
            padded.extend_from_slice(b"7Z:59+0101234567890123456789012345");
            let pair = unsafe { parse_ts_ingest(buf.as_ptr(), text.len(), padded.as_ptr(), partner.len(), cache) };
            assert_eq!(pair, (got, if want == BAD { i64::MAX } else { want }), "{:?}", String::from_utf8_lossy(text));
            let pair = unsafe { parse_ts_ingest(padded.as_ptr(), partner.len(), other.as_ptr(), text.len(), cache) };
            assert_eq!(pair, (want, ingest), "{:?}", String::from_utf8_lossy(text));
        }
        got
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

    /// Thirteen bytes that are digits or nearly digits (the neighbours of
    /// '0' and '9', digits with the high bit set, NUL), and the calendar
    /// forms with every single byte replaced by every byte value: the
    /// table-driven path and the reference agree, on a cold and a warm cache.
    #[test]
    fn near_misses_match_reference() {
        let mut cache = DateCache::new();
        let mut state = 77;
        let odd: &[u8] = b"/:\x00\xb0\xb9 +-\xff";
        for _ in 0..300_000 {
            let mut text: Vec<u8> = (0..13).map(|_| b'0' + (rng(&mut state) % 10) as u8).collect();
            for _ in 0..rng(&mut state) % 3 {
                let at = (rng(&mut state) % 13) as usize;
                text[at] = odd[(rng(&mut state) % odd.len() as u64) as usize];
            }
            check(&text, &mut cache);
        }
        for digits in ["0000000000000", "9999999999999", "0000000000001", "1791071890000", "0001791071890"] {
            assert_eq!(fast(digits.as_bytes(), &mut cache), digits.parse::<i64>().unwrap());
        }
        for base in ["2026-10-03T23:58:10Z", "2026-10-03T19:58:10-04:00", "2028-02-29T00:00:00+13:59", "1791071890000"] {
            for at in 0..base.len() {
                for byte in 0..=255u8 {
                    let mut text = base.as_bytes().to_vec();
                    text[at] = byte;
                    check(&text, &mut cache);
                    check(&text, &mut DateCache::new());
                }
            }
        }
    }

    /// Every day from 1969 to 2101 in F1 (value checked) and F2, then every
    /// day number 0 to 32 of every month 0 to 13 of those years, leap days
    /// and Feb 30 among them: the fast readers and the reference agree.
    #[test]
    fn every_date_1969_to_2101() {
        let mut cache = DateCache::new();
        let mut state = 3;
        for day in days_from_civil(1969, 1, 1)..days_from_civil(2102, 1, 1) {
            let (y, m, d) = civil_from_days(day);
            let s = rng(&mut state) % 86_400;
            let (hh, mi, ss) = (s / 3600, s / 60 % 60, s % 60);
            let f1 = format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mi:02}:{ss:02}Z");
            assert_eq!(fast(f1.as_bytes(), &mut cache), day * DAY_MS + s as i64 * 1000, "{f1}");
            let sign = if rng(&mut state) % 2 == 0 { '+' } else { '-' };
            let f2 = format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mi:02}:{ss:02}{sign}{:02}:{:02}", rng(&mut state) % 24, rng(&mut state) % 60);
            check(f2.as_bytes(), &mut cache);
        }
        for y in 1969..=2101 {
            for m in 0..=13 {
                for d in 0..=32 {
                    check(format!("{y:04}-{m:02}-{d:02}T00:00:00Z").as_bytes(), &mut cache);
                    check(format!("{y:04}-{m:02}-{d:02}T23:59:59-23:59").as_bytes(), &mut cache);
                }
            }
        }
    }

    /// Times of day at and past every limit, fractions of 1 to 12 digits,
    /// offsets of both signs and every width, and every length from 0 to 64
    /// (lengths of 32 or more whose low five bits name a form among them).
    #[test]
    fn times_offsets_and_lengths() {
        let mut cache = DateCache::new();
        for hh in 0..=25 {
            for mm in [0, 1, 9, 10, 58, 59, 60, 61, 99] {
                for ss in [0, 1, 9, 10, 58, 59, 60, 61, 99] {
                    for tail in ["Z", "+00:00", "-00:00", "-23:59", "+23:59", "+24:00", "-00:60", "+12:00", "+1:00", "+001:00", "-10:5", "+10:500"] {
                        check(format!("2028-02-29T{hh:02}:{mm:02}:{ss:02}{tail}").as_bytes(), &mut cache);
                        check(format!("2027-12-31T{hh:02}:{mm:02}:{ss:02}{tail}").as_bytes(), &mut cache);
                    }
                }
            }
        }
        let digits = "123456789012";
        for n in 1..=12 {
            let frac = &digits[..n];
            check(format!("2026-10-03T23:59:59.{frac}Z").as_bytes(), &mut cache);
            check(format!("2026-10-03T23:59:59.{frac}+05:30").as_bytes(), &mut cache);
            check(format!("1791071890000.{frac}").as_bytes(), &mut cache);
            check(format!("{frac}").as_bytes(), &mut cache);
        }
        for base in ["2026-10-03T23:58:10Z", "2026-10-03T19:58:10-04:00", "1791071890000", "0000000000000"] {
            for len in 0..=64 {
                // The form's bytes, cut short or repeated to the length.
                let text: Vec<u8> = base.bytes().cycle().take(len).collect();
                check(&text, &mut cache);
                // The form's bytes behind padding to the length.
                if len >= base.len() {
                    let mut text = vec![b'0'; len - base.len()];
                    text.extend_from_slice(base.as_bytes());
                    check(&text, &mut cache);
                    let mut text = base.as_bytes().to_vec();
                    text.resize(len, b'0');
                    check(&text, &mut cache);
                }
            }
        }
        // 32 leading zeros: a valid F3 value whose length is 13 mod 32.
        let long = format!("{}1791071890000", "0".repeat(32));
        assert_eq!(fast(long.as_bytes(), &mut cache), 1_791_071_890_000);
    }

    /// The pair reader with `ingest_ts` lengths 0 to 40, where the bytes
    /// past the field always read as a valid F1: only a 20-byte field is
    /// one.
    #[test]
    fn pair_reader_ingest_lengths() {
        let mut cache = DateCache::new();
        let ingest = b"2026-10-03T23:58:10Z2026-10-03T23:58:10Z2026-10-03T23:58:10Z";
        let mut ts = b"2026-10-03T23:58:00Z".to_vec();
        ts.extend_from_slice(&[b'0'; 32]);
        for ilen in 0..=40 {
            for start in [0, 1, 20] {
                let field = &ingest[start..start + ilen];
                let want = parse_ts_naive(field).unwrap_or(i64::MAX);
                let pair = unsafe { parse_ts_ingest(ts.as_ptr(), 20, ingest.as_ptr().add(start), ilen, &mut cache) };
                assert_eq!(pair, (1_791_071_880_000, want), "{ilen} {start}");
            }
        }
    }

    /// The month cache: months of one parity in turn evict each other and
    /// still read right, two dates of one month share a slot, a prefix that
    /// is not a month never enters a slot, and the slots without a calendar
    /// never change.
    #[test]
    fn cache_slots() {
        let key = |s: &[u8; 8]| u64::from_le_bytes(*s);
        for m in 1..=12 {
            let next = m % 12 + 1;
            let a = key(format!("2026-{m:02}-").as_bytes().try_into().unwrap());
            let b = key(format!("2026-{next:02}-").as_bytes().try_into().unwrap());
            assert_ne!(slot_off(a), slot_off(b), "{m}");
        }
        let mut cache = DateCache::new();
        let texts = [
            "2026-01-31T10:00:00Z",
            "2026-03-01T10:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-02-28T23:59:59Z",
            "2026-03-31T00:00:00+01:00",
            "2025-11-30T00:00:00Z",
            "2026-01-15T00:00:00Z",
            "2025-12-31T23:59:59Z",
            "2026-10-03T00:00:00Z",
            "2026-10-31T00:00:00Z",
            "2026-10-32T00:00:00Z",
            "2026-11-31T00:00:00Z",
            "2027-02-29T00:00:00Z",
            "2028-02-29T00:00:00Z",
            "2028-02-30T00:00:00Z",
            "1791071890000",
        ];
        for _ in 0..3 {
            for t in texts {
                check(t.as_bytes(), &mut cache);
            }
        }
        // Hit after miss: the slot holds what the miss put there. (Only the
        // single reader here: `fast` also reads other months.)
        let mut cache = DateCache::new();
        assert_eq!(fast_single(b"2026-11-05T00:00:00Z", &mut cache), days_from_civil(2026, 11, 5) * DAY_MS);
        assert_eq!(cache.slots[1].key, key(b"2026-11-"));
        assert_eq!(fast_single(b"2026-11-06T00:00:01Z", &mut cache), days_from_civil(2026, 11, 6) * DAY_MS + 1000);
        // Not months: the slot of their parity keeps its month.
        for bad in ["2026-13-05T00:00:00Z", "2026-1305T00:00:00Z", "2026-00-05T00:00:00Z", "2026-1a-05T00:00:00Z", "2026-21-05T00:00:00Z"] {
            assert_eq!(fast_single(bad.as_bytes(), &mut cache), BAD, "{bad}");
            let mut buf = bad.as_bytes().to_vec();
            buf.extend_from_slice(&[0; 32]);
            assert_eq!(unsafe { parse_ingest(buf.as_ptr(), 20, &mut cache) }, i64::MAX, "{bad}");
        }
        assert_eq!(cache.slots[1].key, key(b"2026-11-"));
        assert_eq!(cache.slots[0].key, key(b"1970-02-"));
        for slot in &cache.slots[2..] {
            assert_eq!((slot.lim, slot.key, slot.base_ms), ([u16::MAX; 8], 0, 0));
        }
    }

    /// The reads stay within their stated bounds: each field is put so that
    /// the page after the bound is unmapped.
    #[cfg(target_os = "linux")]
    #[test]
    fn reads_stay_in_bounds() {
        unsafe {
            let page = 4096;
            let base = libc::mmap(
                std::ptr::null_mut(),
                2 * page,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            ) as *mut u8;
            assert!(!base.is_null() && base as isize != -1);
            assert_eq!(libc::mprotect(base.add(page) as *mut libc::c_void, page, libc::PROT_NONE), 0);
            let end = base.add(page);
            std::ptr::write_bytes(base, b'7', page);
            let mut cache = DateCache::new();
            let texts: [&[u8]; 9] = [
                b"2026-10-03T23:58:10Z",
                b"2026-10-03T19:58:10-04:00",
                b"1791071890000",
                b"",
                b"1",
                b"2026-10-03T23:58:10",
                b"2026-10-03T23:58:10Z0000000000000000000000000000000000",
                b"0000000000000000000000000000000000000000000001791071890000",
                b"20261003",
            ];
            let place = |text: &[u8], room: usize| -> *const u8 {
                let p = end.sub(text.len() + room);
                std::ptr::copy_nonoverlapping(text.as_ptr(), p, text.len());
                p
            };
            for t in texts {
                let want = parse_ts_naive(t).unwrap_or(BAD);
                assert_eq!(parse_ts(place(t, 16), t.len(), &mut cache), want);
                assert_eq!(parse_ts_general(place(t, 16), t.len(), &mut cache), want);
                assert_eq!(parse_ingest(place(t, 16), t.len(), &mut cache), if want == BAD { i64::MAX } else { want });
                for i in texts {
                    let iwant = parse_ts_naive(i).unwrap_or(i64::MAX);
                    // ts with 16 bytes after it, the ingest field last.
                    let mut ts = t.to_vec();
                    ts.extend_from_slice(&[b'7'; 16]);
                    let ip = place(i, 16usize.max(20usize.saturating_sub(i.len())));
                    assert_eq!(parse_ts_ingest(ts.as_ptr(), t.len(), ip, i.len(), &mut cache), (want, iwant));
                    // ingest_ts first, ts last.
                    let mut ing = i.to_vec();
                    ing.extend_from_slice(&[b'7'; 20]);
                    let tp = place(t, 16);
                    assert_eq!(parse_ts_ingest(tp, t.len(), ing.as_ptr(), i.len(), &mut cache), (want, iwant));
                }
            }
            libc::munmap(base as *mut libc::c_void, 2 * page);
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

    #[test]
    fn canonical_numbers_match_reference() {
        let reference = |b: &[u8]| -> Option<u32> {
            if !(1..=7).contains(&b.len()) || b[0] == b'0' || !b.iter().all(u8::is_ascii_digit) {
                return None;
            }
            std::str::from_utf8(b).ok()?.parse().ok()
        };
        // Bytes next to the digits, below and above them, and high bytes
        // whose low seven bits are digits.
        let alphabet = b"0123456789/:\x00\x7f\xb0\xb9\xff.c_p";
        let mut state = 11;
        for _ in 0..300_000 {
            let len = (rng(&mut state) % 10) as usize;
            let digits_only = rng(&mut state) % 2 == 0;
            let case: Vec<u8> = (0..len)
                .map(|_| {
                    let r = rng(&mut state) as usize;
                    if digits_only && r % 16 != 0 { b'0' + (r % 10) as u8 } else { alphabet[r % alphabet.len()] }
                })
                .collect();
            for pad in [&[0u8; 16][..], &[0xFF; 16], b"0000000000000000", b"/:/:/:/:/:/:/:/:"] {
                let mut buf = case.clone();
                buf.extend_from_slice(pad);
                assert_eq!(unsafe { canonical_number(buf.as_ptr(), len) }, reference(&case), "{case:?}");
            }
        }
    }
}
