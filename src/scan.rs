//! Stage 1 of the CSV reader: one pass over a window of bytes that records
//! where the commas and newlines are and which quotes are illegal.
//!
//! The dialect makes this simple: no value holds a comma, quote or newline,
//! so every comma and every `\n` is structural whatever the quoting. A quote
//! is legal only as the first or last byte of a field, so a quote with no
//! delimiter on either side makes its line malformed. Those are rare and
//! go to an exception list; the per-field open/close check in stage 2
//! catches the rest (`"abc`, `abc"`, a lone `"`).

/// 64-byte block to three bitmasks: commas, newlines, quotes.
#[inline(always)]
unsafe fn masks(p: *const u8) -> (u64, u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::*;
        let a = _mm256_loadu_si256(p as *const __m256i);
        let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let m = |x: __m256i, c: i8| -> u64 {
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(x, _mm256_set1_epi8(c))) as u32 as u64
        };
        return (
            m(a, b',' as i8) | (m(b, b',' as i8) << 32),
            m(a, b'\n' as i8) | (m(b, b'\n' as i8) << 32),
            m(a, b'"' as i8) | (m(b, b'"' as i8) << 32),
        );
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        use core::arch::x86_64::*;
        let mut out = (0u64, 0u64, 0u64);
        for i in 0..4 {
            let x = _mm_loadu_si128(p.add(16 * i) as *const __m128i);
            let m = |c: i8| -> u64 { _mm_movemask_epi8(_mm_cmpeq_epi8(x, _mm_set1_epi8(c))) as u32 as u64 };
            out.0 |= m(b',' as i8) << (16 * i);
            out.1 |= m(b'\n' as i8) << (16 * i);
            out.2 |= m(b'"' as i8) << (16 * i);
        }
        return out;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        let x0 = vld1q_u8(p);
        let x1 = vld1q_u8(p.add(16));
        let x2 = vld1q_u8(p.add(32));
        let x3 = vld1q_u8(p.add(48));
        const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let bits = vld1q_u8(BITS.as_ptr());
        let m = |c: u8| -> u64 {
            let k = vdupq_n_u8(c);
            let t0 = vandq_u8(vceqq_u8(x0, k), bits);
            let t1 = vandq_u8(vceqq_u8(x1, k), bits);
            let t2 = vandq_u8(vceqq_u8(x2, k), bits);
            let t3 = vandq_u8(vceqq_u8(x3, k), bits);
            let s0 = vpaddq_u8(t0, t1);
            let s1 = vpaddq_u8(t2, t3);
            let s = vpaddq_u8(s0, s1);
            let s = vpaddq_u8(s, s);
            vgetq_lane_u64(vreinterpretq_u64_u8(s), 0)
        };
        return (m(b','), m(b'\n'), m(b'"'));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        masks_scalar(p)
    }
}

/// Portable reference for `masks`; also what the tests compare against.
#[allow(dead_code)]
pub unsafe fn masks_scalar(p: *const u8) -> (u64, u64, u64) {
    let mut out = (0u64, 0u64, 0u64);
    for i in 0..64 {
        let b = *p.add(i);
        out.0 |= u64::from(b == b',') << i;
        out.1 |= u64::from(b == b'\n') << i;
        out.2 |= u64::from(b == b'"') << i;
    }
    out
}

pub struct Scan {
    /// Positions of commas and newlines, ascending, relative to the window.
    /// Slot 0 stands for the line break before the window (position -1);
    /// the positions start at slot 1. See `delims()`.
    d: Vec<u32>,
    /// One byte per delimiter, laid out like `d`: `CLOSES` when the byte
    /// before the delimiter is a quote, `OPENS` when the byte after it is,
    /// `NEWLINE` when the delimiter is a line break. See `flags()`.
    fl: Vec<u8>,
    /// Positions of quotes with no delimiter on either side.
    pub badq: Vec<u32>,
    pub nd: usize,
}

const PAD: usize = 80;
/// How far ahead of the scan the input is prefetched. Flat from 512 to
/// 4096 bytes on the test box.
const AHEAD: usize = 1024;

/// The field that ends at this delimiter ends with a quote.
const CLOSES: u8 = 1;
/// The field that starts after this delimiter starts with a quote.
const OPENS: u8 = 2;
/// The delimiter is a line break, not a comma.
const NEWLINE: u8 = 4;
const CLOSES8: u64 = 0x0101_0101_0101_0101;
const OPENS8: u64 = 0x0202_0202_0202_0202;
const NEWLINE8: u64 = 0x0404_0404_0404_0404;

/// Flag bytes of up to eight delimiters. `before` and `after` have bit `p`
/// set when the byte before, or after, position `p` is a quote; `nm` when
/// position `p` is a line break. `skip` delimiters of `delim` are passed
/// over first.
#[inline(always)]
fn flag_bytes(before: u64, after: u64, nm: u64, delim: u64, skip: u32) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        use core::arch::x86_64::{_pdep_u64, _pext_u64};
        let c = _pext_u64(before, delim) >> skip;
        let o = _pext_u64(after, delim) >> skip;
        let n = _pext_u64(nm, delim) >> skip;
        _pdep_u64(c, CLOSES8) | _pdep_u64(o, OPENS8) | _pdep_u64(n, NEWLINE8)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let mut m = delim;
        for _ in 0..skip {
            m &= m.wrapping_sub(1);
        }
        let mut out = 0u64;
        let mut i = 0;
        while m != 0 && i < 8 {
            let p = m.trailing_zeros();
            let f = (((before >> p) & 1) * CLOSES as u64)
                | (((after >> p) & 1) * OPENS as u64)
                | (((nm >> p) & 1) * NEWLINE as u64);
            out |= f << (8 * i);
            m &= m - 1;
            i += 1;
        }
        out
    }
}

impl Scan {
    pub fn new(window: usize) -> Self {
        Scan {
            d: vec![u32::MAX; window + 64 + PAD + 1],
            fl: vec![0; window + 64 + PAD + 1],
            badq: vec![u32::MAX; window + 64 + PAD],
            nd: 0,
        }
    }

    /// The delimiter positions. `delims().sub(1)` is readable and holds
    /// `u32::MAX`: the line break before the window.
    #[inline(always)]
    pub fn delims(&self) -> *const u32 {
        unsafe { self.d.as_ptr().add(1) }
    }

    /// The flag byte of each delimiter. `flags().sub(1)` is the flag of the
    /// line break before the window.
    #[inline(always)]
    pub fn flags(&self) -> *const u8 {
        unsafe { self.fl.as_ptr().add(1) }
    }

    /// Scan `len` bytes at `base`. The byte before `base` must be a line
    /// break (or the start of the file), and at least 64 bytes past
    /// `base + len` must be readable.
    pub unsafe fn scan(&mut self, base: *const u8, len: usize) {
        debug_assert!(len + 64 + PAD < self.d.len());
        let d = self.d.as_mut_ptr().add(1);
        let fl = self.fl.as_mut_ptr().add(1);
        let bad = self.badq.as_mut_ptr();
        let (mut nd, mut nbad) = (0usize, 0usize);
        let mut prev_delim = 1u64;
        let mut prev_quote = 0u64;
        let mut pending = 0u64;
        let mut at = 0usize;
        *d.sub(1) = u32::MAX;
        *fl.sub(1) = if *base == b'"' { OPENS } else { 0 };
        while at < len {
            // The input comes straight from the page cache in 4 KiB pages,
            // and the hardware prefetcher stops at every page end. Without
            // this a third of the scan is spent waiting on the two loads.
            crate::sys::prefetch(base.wrapping_add(at + AHEAD));
            let (mut cm, mut nm, mut qm) = masks(base.add(at));
            let left = len - at;
            if left < 64 {
                let keep = (1u64 << left) - 1;
                cm &= keep;
                nm &= keep;
                qm &= keep;
            }
            let delim = cm | nm;
            if (qm | pending) != 0 {
                // A quote that does not follow a delimiter must precede one.
                let nonopen = qm & !((delim << 1) | prev_delim);
                let mut illegal = nonopen & !(delim >> 1) & !(1u64 << 63);
                if pending != 0 && (delim & 1) == 0 {
                    *bad.add(nbad) = at as u32 - 1;
                    nbad += 1;
                }
                pending = nonopen >> 63;
                while illegal != 0 {
                    *bad.add(nbad) = at as u32 + illegal.trailing_zeros();
                    nbad += 1;
                    illegal &= illegal - 1;
                }
            }
            prev_delim = delim >> 63;
            // Quote flags per delimiter, so stage 2 never looks at the
            // bytes around a field. The byte after position 63 is the
            // first of the next block.
            let before = (qm << 1) | prev_quote;
            let after = (qm >> 1) | (u64::from(*base.add(at + 64) == b'"') << 63);
            prev_quote = qm >> 63;

            let base_pos = at as u32;
            let count = delim.count_ones() as usize;
            let mut m = delim;
            let out = d.add(nd);
            let fout = fl.add(nd);
            for i in 0..8 {
                // Volatile keeps these eight plain stores. Left alone the
                // compiler packs them into one vector with seven inserts,
                // which is slower here.
                out.add(i).write_volatile(base_pos + m.trailing_zeros());
                m &= m.wrapping_sub(1);
            }
            (fout as *mut u64).write_unaligned(flag_bytes(before, after, nm, delim, 0).to_le());
            if count > 8 {
                for i in 8..16 {
                    *out.add(i) = base_pos + m.trailing_zeros();
                    m &= m.wrapping_sub(1);
                }
                (fout.add(8) as *mut u64).write_unaligned(flag_bytes(before, after, nm, delim, 8).to_le());
                if count > 16 {
                    let mut i = 16;
                    while m != 0 {
                        *out.add(i) = base_pos + m.trailing_zeros();
                        m &= m - 1;
                        i += 1;
                    }
                    let mut k = 16;
                    while k < count {
                        (fout.add(k) as *mut u64)
                            .write_unaligned(flag_bytes(before, after, nm, delim, k as u32).to_le());
                        k += 8;
                    }
                }
            }
            nd += count;
            at += 64;
        }
        // Sentinels: stage 2 reads up to 8 entries past the last delimiter.
        for i in 0..16 {
            *d.add(nd + i) = u32::MAX;
        }
        *bad.add(nbad) = u32::MAX;
        self.nd = nd;
    }
}

/// Field bounds, quotes stripped, of the `N` fields (at most 7) that the
/// next `N` delimiters end, and whether that is not a well-formed line:
/// one of the first `N - 1` delimiters is a line break, the `N`th is not,
/// or a field has a quote at one end only or is one lone quote. `d` and
/// `fl` point at the `N` delimiters and their flags; the entry before each
/// belongs to the line break before the line. Only the first `N` slots of
/// the result mean anything.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
pub unsafe fn fields<const N: usize>(d: *const u32, fl: *const u8) -> ([u32; 8], [u32; 8], u32) {
    let mut s = [0u32; 8];
    let mut e = [0u32; 8];
    let mut bad = 0u32;
    for i in 0..N {
        let open = u32::from(*fl.add(i).sub(1) & OPENS != 0);
        let close = u32::from(*fl.add(i) & CLOSES != 0);
        let newline = *fl.add(i) & NEWLINE != 0;
        let st = (*d.add(i).sub(1)).wrapping_add(1) + open;
        let en = *d.add(i) - close;
        s[i] = st;
        e[i] = en;
        // A field that is one lone quote opens and closes, and ends
        // before it starts.
        bad |= (open ^ close) | u32::from(en < st) | u32::from(newline != (i == N - 1));
    }
    (s, e, bad)
}

/// The same with all fields at once: starts are the delimiters before plus
/// one plus the open flag, ends the own delimiters minus the close flag.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn fields<const N: usize>(d: *const u32, fl: *const u8) -> ([u32; 8], [u32; 8], u32) {
    use core::arch::x86_64::*;
    debug_assert!(N <= 7);
    // Byte k: the flags of delimiter k - 1 of the line.
    let f = u64::from_le((fl.sub(1) as *const u64).read_unaligned());
    let opens = (f >> 1) & CLOSES8;
    let closes = f & CLOSES8;
    let uneven = (opens ^ (closes >> 8)) & (u64::MAX >> (64 - 8 * N));
    // Bytes 1..=N: commas, then the line break.
    let breaks = (f & (NEWLINE8 >> (56 - 8 * N)) & !0xFF) ^ ((NEWLINE as u64) << (8 * N));
    let v = _mm256_loadu_si256(d.sub(1) as *const __m256i);
    let open = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(opens as i64));
    let close = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(closes as i64));
    let start = _mm256_sub_epi32(_mm256_add_epi32(v, open), _mm256_set1_epi32(-1));
    let end = _mm256_permutevar8x32_epi32(_mm256_sub_epi32(v, close), _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7));
    // A field that is one lone quote opens and closes, and ends before it
    // starts.
    let lone = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_sub_epi32(end, start))) as u32 & ((1 << N) - 1);
    let mut s = [0u32; 8];
    let mut e = [0u32; 8];
    _mm256_storeu_si256(s.as_mut_ptr() as *mut __m256i, start);
    _mm256_storeu_si256(e.as_mut_ptr() as *mut __m256i, end);
    (s, e, lone | u32::from((uneven | breaks) != 0))
}

/// The line whose delimiters start at `d[di]` is not a well-formed line of
/// the expected width, or fewer delimiters are left than a line needs.
/// Skip to the end of that line: the new `(di, bq, line start)`, or `None`
/// when no line break is left in the window.
#[cold]
#[inline(never)]
unsafe fn skip_line(scan: &Scan, mut di: usize, mut bq: usize) -> Option<(usize, usize, u32)> {
    let fl = scan.flags();
    let badq = scan.badq.as_ptr();
    while di < scan.nd && *fl.add(di) & NEWLINE == 0 {
        di += 1;
    }
    if di >= scan.nd {
        return None;
    }
    let p = *scan.delims().add(di);
    while *badq.add(bq) < p {
        bq += 1;
    }
    Some((di + 1, bq, p + 1))
}

/// Walk the complete lines of a scanned window. For each line that has
/// exactly `N` fields and no misplaced quote, `f(starts, ends)` is called.
/// Lines starting at or after `limit` are left alone. Returns the offset of
/// the first unconsumed line.
#[inline(always)]
pub unsafe fn for_each_line<const N: usize>(
    scan: &Scan,
    limit: u32,
    mut f: impl FnMut(&[u32; N], &[u32; N]),
) -> u32 {
    let d = scan.delims();
    let fl = scan.flags();
    let badq = scan.badq.as_ptr();
    let nd = scan.nd;
    let mut di = 0usize;
    let mut bq = 0usize;
    let mut ls = 0u32;
    while ls < limit {
        if di + N <= nd {
            let p = *d.add(di + N - 1);
            let (s, e, bad) = fields::<N>(d.add(di), fl.add(di));
            if (bad == 0) & (*badq.add(bq) >= p) {
                f(&*(s.as_ptr() as *const [u32; N]), &*(e.as_ptr() as *const [u32; N]));
                di += N;
                ls = p + 1;
                continue;
            }
        }
        match skip_line(scan, di, bq) {
            Some(next) => (di, bq, ls) = next,
            None => break,
        }
    }
    ls
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::split_line;

    fn rng(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[test]
    fn masks_match_scalar() {
        let mut state = 1;
        for _ in 0..2000 {
            let mut buf = [0u8; 64];
            for b in buf.iter_mut() {
                *b = match rng(&mut state) % 6 {
                    0 => b',',
                    1 => b'\n',
                    2 => b'"',
                    _ => (rng(&mut state) % 256) as u8,
                };
            }
            unsafe {
                assert_eq!(masks(buf.as_ptr()), masks_scalar(buf.as_ptr()));
            }
        }
    }

    /// Random soup of the structural bytes: the fast line walker must accept
    /// exactly the lines the plain splitter accepts, with the same fields.
    #[test]
    fn lines_match_split_line() {
        let mut state = 7;
        for round in 0..3000 {
            let n = 1 + (rng(&mut state) % 700) as usize;
            let mut text = vec![b'\n'];
            let alphabet: &[u8] = if round % 3 == 0 { b"a,\"\n" } else { b"ab,,\"\n\r 1" };
            for _ in 0..n {
                text.push(alphabet[(rng(&mut state) % alphabet.len() as u64) as usize]);
            }
            text.push(b'\n');
            let body_len = text.len() - 1;
            text.extend_from_slice(&[0u8; 160]);
            for fields_n in [2usize, 4, 7] {
                let mut expect: Vec<Vec<Vec<u8>>> = Vec::new();
                let mut tmp = Vec::new();
                for line in text[1..body_len].split(|&b| b == b'\n') {
                    if split_line(line, fields_n, &mut tmp) {
                        expect.push(tmp.iter().map(|f| f.to_vec()).collect());
                    }
                }
                let mut scan = Scan::new(body_len);
                let mut got: Vec<Vec<Vec<u8>>> = Vec::new();
                unsafe {
                    let base = text.as_ptr().add(1);
                    scan.scan(base, body_len);
                    let mut collect = |s: &[u32], e: &[u32]| {
                        got.push(
                            (0..s.len())
                                .map(|i| text[1 + s[i] as usize..1 + e[i] as usize].to_vec())
                                .collect(),
                        );
                    };
                    let end = match fields_n {
                        2 => for_each_line::<2>(&scan, u32::MAX, |s, e| collect(s, e)),
                        4 => for_each_line::<4>(&scan, u32::MAX, |s, e| collect(s, e)),
                        _ => for_each_line::<7>(&scan, u32::MAX, |s, e| collect(s, e)),
                    };
                    assert_eq!(end as usize, body_len);
                }
                assert_eq!(got, expect, "round {round} fields {fields_n}: {:?}", String::from_utf8_lossy(&text[1..body_len]));
            }
        }
    }

    /// Every delimiter's flag byte says what the bytes around it are, at
    /// every alignment to the 64-byte blocks, with one `Scan` reused.
    #[test]
    fn flags_match_bytes() {
        let mut state = 21;
        let mut scan = Scan::new(4096);
        for round in 0..4000 {
            let n = 1 + (rng(&mut state) % 600) as usize;
            let alphabet: &[u8] = match round % 4 {
                0 => b"\",\n",
                1 => b"ab\"\",\n",
                2 => b"abcdefgh\",",
                _ => b"abcdefghijklmnopqrstuvwxyz0123456789\"\"\"\",,,,,\n",
            };
            let mut text = vec![b'\n'];
            for _ in 0..n {
                text.push(alphabet[(rng(&mut state) % alphabet.len() as u64) as usize]);
            }
            // Hostile bytes past the window must not leak into it.
            text.extend_from_slice(&[b'"'; 160]);
            unsafe {
                let base = text.as_ptr().add(1);
                scan.scan(base, n);
                let body = &text[1..];
                let want: Vec<u32> = (0..n as u32).filter(|&i| matches!(body[i as usize], b',' | b'\n')).collect();
                assert_eq!(scan.nd, want.len(), "round {round}");
                assert_eq!(*scan.delims().sub(1), u32::MAX);
                assert_eq!(*scan.flags().sub(1) & OPENS != 0, body[0] == b'"');
                for (k, &p) in want.iter().enumerate() {
                    let p = p as usize;
                    assert_eq!(*scan.delims().add(k), p as u32, "round {round}");
                    let flag = *scan.flags().add(k);
                    assert_eq!(flag & CLOSES != 0, p > 0 && body[p - 1] == b'"', "round {round} delimiter {k} at {p}");
                    assert_eq!(flag & NEWLINE != 0, body[p] == b'\n', "round {round} delimiter {k} at {p}");
                    // The byte after the window's last byte belongs to no
                    // line of this window.
                    if p + 1 < n {
                        assert_eq!(flag & OPENS != 0, body[p + 1] == b'"', "round {round} delimiter {k} at {p}");
                    }
                }
            }
        }
    }

    /// Lines shaped like the real files (field lengths, 20% quoted, some
    /// malformed) cut into windows of every size: the walker and the plain
    /// splitter agree, also on what is left for the next window.
    #[test]
    fn realistic_lines_match_split_line() {
        let mut state = 33;
        let mut scan = Scan::new(1 << 12);
        for round in 0..1500 {
            let fields_n = [2usize, 4, 7][round % 3];
            let mut text = vec![b'\n'];
            let lines = 1 + rng(&mut state) % 40;
            for _ in 0..lines {
                let width = match rng(&mut state) % 40 {
                    0 => fields_n + 1,
                    1 => fields_n - 1,
                    2 => 0,
                    _ => fields_n,
                };
                for i in 0..width {
                    if i > 0 {
                        text.push(b',');
                    }
                    let len = match rng(&mut state) % 10 {
                        0 => 0,
                        1 => 1,
                        2 => 25 + rng(&mut state) % 50,
                        _ => 3 + rng(&mut state) % 20,
                    };
                    let quoting = rng(&mut state) % 100;
                    if quoting < 20 || quoting == 97 {
                        text.push(b'"');
                    }
                    for _ in 0..len {
                        text.push(b"abcxyz0189 :-TZ\r"[(rng(&mut state) % 16) as usize]);
                    }
                    if quoting == 96 {
                        text.push(b'"');
                        text.push(b'x');
                    }
                    if quoting < 20 || quoting == 98 {
                        text.push(b'"');
                    }
                }
                text.push(b'\n');
            }
            let body_len = text.len() - 1;
            text.extend_from_slice(&[0u8; 160]);
            let mut expect: Vec<Vec<Vec<u8>>> = Vec::new();
            let mut tmp = Vec::new();
            for line in text[1..body_len].split(|&b| b == b'\n') {
                if split_line(line, fields_n, &mut tmp) {
                    expect.push(tmp.iter().map(|f| f.to_vec()).collect());
                }
            }
            // Walk it in windows of a random size, as run_chunk does.
            let window = 100 + (rng(&mut state) % 3000) as usize;
            let mut got: Vec<Vec<Vec<u8>>> = Vec::new();
            let mut pos = 0usize;
            unsafe {
                while pos < body_len {
                    let wlen = window.min(body_len - pos);
                    let base = text.as_ptr().add(1 + pos);
                    scan.scan(base, wlen);
                    let mut collect = |s: &[u32], e: &[u32]| {
                        got.push(
                            (0..s.len())
                                .map(|i| text[1 + pos + s[i] as usize..1 + pos + e[i] as usize].to_vec())
                                .collect(),
                        );
                    };
                    let used = match fields_n {
                        2 => for_each_line::<2>(&scan, u32::MAX, |s, e| collect(s, e)),
                        4 => for_each_line::<4>(&scan, u32::MAX, |s, e| collect(s, e)),
                        _ => for_each_line::<7>(&scan, u32::MAX, |s, e| collect(s, e)),
                    };
                    if used == 0 {
                        // A line longer than the window: take it whole.
                        let end = text[1 + pos..1 + body_len].iter().position(|&b| b == b'\n').unwrap();
                        if split_line(&text[1 + pos..1 + pos + end], fields_n, &mut tmp) {
                            got.push(tmp.iter().map(|f| f.to_vec()).collect());
                        }
                        pos += end + 1;
                    } else {
                        pos += used as usize;
                    }
                }
            }
            assert_eq!(got, expect, "round {round} fields {fields_n} window {window}");
        }
    }
}
