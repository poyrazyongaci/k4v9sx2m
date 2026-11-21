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
    pub d: Vec<u32>,
    /// Positions of newlines.
    pub nl: Vec<u32>,
    /// Positions of quotes with no delimiter on either side.
    pub badq: Vec<u32>,
    pub nd: usize,
    pub nnl: usize,
}

const PAD: usize = 80;

impl Scan {
    pub fn new(window: usize) -> Self {
        Scan {
            d: vec![u32::MAX; window + 64 + PAD],
            nl: vec![u32::MAX; window + 64 + PAD],
            badq: vec![u32::MAX; window + 64 + PAD],
            nd: 0,
            nnl: 0,
        }
    }

    /// Scan `len` bytes at `base`. The byte before `base` must be a line
    /// break (or the start of the file), and at least 64 bytes past
    /// `base + len` must be readable.
    pub unsafe fn scan(&mut self, base: *const u8, len: usize) {
        debug_assert!(len + 64 + PAD <= self.d.len());
        let d = self.d.as_mut_ptr();
        let nl = self.nl.as_mut_ptr();
        let bad = self.badq.as_mut_ptr();
        let (mut nd, mut nnl, mut nbad) = (0usize, 0usize, 0usize);
        let mut prev_delim = 1u64;
        let mut pending = 0u64;
        let mut at = 0usize;
        while at < len {
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

            let base_pos = at as u32;
            let count = delim.count_ones() as usize;
            let mut m = delim;
            let out = d.add(nd);
            for i in 0..8 {
                *out.add(i) = base_pos + m.trailing_zeros();
                m &= m.wrapping_sub(1);
            }
            if count > 8 {
                for i in 8..16 {
                    *out.add(i) = base_pos + m.trailing_zeros();
                    m &= m.wrapping_sub(1);
                }
                if count > 16 {
                    let mut i = 16;
                    while m != 0 {
                        *out.add(i) = base_pos + m.trailing_zeros();
                        m &= m - 1;
                        i += 1;
                    }
                }
            }
            nd += count;

            let ncount = nm.count_ones() as usize;
            let mut m = nm;
            let out = nl.add(nnl);
            *out = base_pos + m.trailing_zeros();
            m &= m.wrapping_sub(1);
            *out.add(1) = base_pos + m.trailing_zeros();
            m &= m.wrapping_sub(1);
            if ncount > 2 {
                let mut i = 2;
                while m != 0 {
                    *out.add(i) = base_pos + m.trailing_zeros();
                    m &= m - 1;
                    i += 1;
                }
            }
            nnl += ncount;
            at += 64;
        }
        // Sentinels: stage 2 reads up to 16 entries past the last delimiter.
        for i in 0..16 {
            *d.add(nd + i) = u32::MAX;
        }
        *nl.add(nnl) = u32::MAX;
        *nl.add(nnl + 1) = u32::MAX;
        *bad.add(nbad) = u32::MAX;
        self.nd = nd;
        self.nnl = nnl;
    }
}

/// Field bounds of one line with `N` fields, quotes stripped, and whether
/// the quoting is malformed. `d` points at the line's `N` delimiters.
#[inline(always)]
pub unsafe fn fields<const N: usize>(base: *const u8, line_start: u32, d: *const u32) -> ([u32; N], [u32; N], u32) {
    let mut s = [0u32; N];
    let mut e = [0u32; N];
    let mut bad = 0u32;
    let mut st = line_start;
    for i in 0..N {
        let en = *d.add(i);
        let open = u32::from(*base.add(st as usize) == b'"');
        let close = u32::from(*base.offset(en as isize - 1) == b'"');
        s[i] = st + open;
        e[i] = en - close;
        bad |= (open ^ close) | u32::from(en - close < st + open);
        st = en + 1;
    }
    (s, e, bad)
}

/// Walk the complete lines of a scanned window. For each line that has
/// exactly `N` fields and no misplaced quote, `f(starts, ends)` is called.
/// Lines starting at or after `limit` are left alone. Returns the offset of
/// the first unconsumed line.
#[inline(always)]
pub unsafe fn for_each_line<const N: usize>(
    base: *const u8,
    scan: &Scan,
    limit: u32,
    mut f: impl FnMut(&[u32; N], &[u32; N]),
) -> u32 {
    let d = scan.d.as_ptr();
    let nl = scan.nl.as_ptr();
    let badq = scan.badq.as_ptr();
    let mut di = 0usize;
    let mut bq = 0usize;
    let mut ls = 0u32;
    for j in 0..scan.nnl {
        if ls >= limit {
            break;
        }
        let p = *nl.add(j);
        if *d.add(di + N - 1) != p || *badq.add(bq) < p {
            while *d.add(di) < p {
                di += 1;
            }
            di += 1;
            while *badq.add(bq) < p {
                bq += 1;
            }
            ls = p + 1;
            continue;
        }
        let (s, e, bad) = fields::<N>(base, ls, d.add(di));
        di += N;
        ls = p + 1;
        if bad == 0 {
            f(&s, &e);
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
                        2 => for_each_line::<2>(base, &scan, u32::MAX, |s, e| collect(s, e)),
                        4 => for_each_line::<4>(base, &scan, u32::MAX, |s, e| collect(s, e)),
                        _ => for_each_line::<7>(base, &scan, u32::MAX, |s, e| collect(s, e)),
                    };
                    assert_eq!(end as usize, body_len);
                }
                assert_eq!(got, expect, "round {round} fields {fields_n}: {:?}", String::from_utf8_lossy(&text[1..body_len]));
            }
        }
    }
}
