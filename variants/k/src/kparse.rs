//! The one-pass line reader of `fuse.rs` with the block classifier behind
//! a trait, so that it can run on AVX-512 mask registers (`ENGINE_K`).
//!
//! The classifier turns the bytes from a line start into comma, line break
//! and quote masks plus the "clean" flag. Everything after it (field
//! bounds, quote checks, the refusal rule) is the code of `fuse.rs`, shared
//! by every classifier, so the variants can only differ in the masks.
//!
//! Classifiers (`Ops`):
//! - `Model`: plain arrays and loops. The reference for the others, and
//!   what the tests run on any machine.
//! - `Avx2`: the current route (compare, `vpmovmskb`, merge halves).
//! - `Vl`: 256-bit compares straight into `k` masks (AVX-512 BW and VL),
//!   the halves joined with `kunpckdq`. No 512-bit instruction, so no
//!   frequency licence.
//! - `Zmm`: one 512-bit load and compare per 64 bytes.
//!
//! Each `Ops` method of a hardware classifier is one intrinsic (or one
//! pair of them) and holds no logic; the logic lives in the generic code
//! and is tested through `Model`.
//!
//! Reads: `event_line` reads the 96 bytes from the line start, `click_line`
//! 64, exactly as `fuse.rs` does; the caller must make them readable.

use crate::fuse::{self, REFUSED};
use crate::scan::Bounds;
use std::sync::atomic::{AtomicU8, Ordering};

/// The masks of the 64 or 96 bytes from a line start. `c`, `n`, `q`: bytes
/// 0..64; `c2`, `n2`, `q2`: bytes 64..96 (zero unless read). `clean`: every
/// one of bytes 0..64 is in 1..=127.
pub struct Masks {
    pub c: u64,
    pub n: u64,
    pub q: u64,
    pub c2: u32,
    pub n2: u32,
    pub q2: u32,
    pub clean: bool,
}

/// The primitive operations of a block classifier.
pub trait Ops {
    /// 64 bytes.
    type Z: Copy;
    /// 32 bytes.
    type Y: Copy;
    unsafe fn load_z(p: *const u8) -> Self::Z;
    unsafe fn load_y(p: *const u8) -> Self::Y;
    /// Bit `i`: byte `i` equals `b`.
    unsafe fn eq_z(z: Self::Z, b: u8) -> u64;
    unsafe fn eq_y(y: Self::Y, b: u8) -> u32;
    /// Every byte is in 1..=127 (signed greater than zero).
    unsafe fn clean_z(z: Self::Z) -> bool;
}

#[inline(always)]
unsafe fn classify<K: Ops, const WIDE: bool>(p: *const u8) -> Masks {
    let z = K::load_z(p);
    let mut m = Masks {
        c: K::eq_z(z, b','),
        n: K::eq_z(z, b'\n'),
        q: K::eq_z(z, b'"'),
        c2: 0,
        n2: 0,
        q2: 0,
        clean: K::clean_z(z),
    };
    if WIDE {
        let y = K::load_y(p.add(64));
        m.c2 = K::eq_y(y, b',');
        m.n2 = K::eq_y(y, b'\n');
        m.q2 = K::eq_y(y, b'"');
    }
    m
}

// ------------------------------------------------------------ classifiers

/// Scalar model of the intrinsics.
#[allow(dead_code)]
pub struct Model;

impl Ops for Model {
    type Z = [u8; 64];
    type Y = [u8; 32];
    #[inline(always)]
    unsafe fn load_z(p: *const u8) -> [u8; 64] {
        std::ptr::read_unaligned(p as *const [u8; 64])
    }
    #[inline(always)]
    unsafe fn load_y(p: *const u8) -> [u8; 32] {
        std::ptr::read_unaligned(p as *const [u8; 32])
    }
    #[inline(always)]
    unsafe fn eq_z(z: [u8; 64], b: u8) -> u64 {
        z.iter().enumerate().fold(0, |m, (i, &x)| m | (u64::from(x == b) << i))
    }
    #[inline(always)]
    unsafe fn eq_y(y: [u8; 32], b: u8) -> u32 {
        y.iter().enumerate().fold(0, |m, (i, &x)| m | (u32::from(x == b) << i))
    }
    #[inline(always)]
    unsafe fn clean_z(z: [u8; 64]) -> bool {
        z.iter().all(|&x| (x as i8) > 0)
    }
}

#[cfg(target_arch = "x86_64")]
pub use hw::*;

#[cfg(target_arch = "x86_64")]
mod hw {
    use super::Ops;
    use core::arch::x86_64::*;

    /// The current AVX2 route.
    pub struct Avx2;

    impl Ops for Avx2 {
        type Z = (__m256i, __m256i);
        type Y = __m256i;
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn load_z(p: *const u8) -> Self::Z {
            (_mm256_loadu_si256(p as *const __m256i), _mm256_loadu_si256(p.add(32) as *const __m256i))
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn load_y(p: *const u8) -> __m256i {
            _mm256_loadu_si256(p as *const __m256i)
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn eq_z(z: Self::Z, b: u8) -> u64 {
            u64::from(Self::eq_y(z.0, b)) | (u64::from(Self::eq_y(z.1, b)) << 32)
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn eq_y(y: __m256i, b: u8) -> u32 {
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(y, _mm256_set1_epi8(b as i8))) as u32
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn clean_z(z: Self::Z) -> bool {
            _mm256_movemask_epi8(_mm256_cmpgt_epi8(_mm256_min_epi8(z.0, z.1), _mm256_setzero_si256())) == -1
        }
    }

    /// 256-bit vectors, compares into `k` masks.
    pub struct Vl;

    impl Ops for Vl {
        type Z = (__m256i, __m256i);
        type Y = __m256i;
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn load_z(p: *const u8) -> Self::Z {
            (_mm256_loadu_si256(p as *const __m256i), _mm256_loadu_si256(p.add(32) as *const __m256i))
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn load_y(p: *const u8) -> __m256i {
            _mm256_loadu_si256(p as *const __m256i)
        }
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
        unsafe fn eq_z(z: Self::Z, b: u8) -> u64 {
            // The empty `asm!` keeps LLVM from joining the two 256-bit
            // compares into one 512-bit compare (AVX-512F is on here and
            // nothing asks it to prefer 256 bits).
            let mut hi = Self::eq_y(z.1, b);
            core::arch::asm!("/* {0} */", inout(reg) hi, options(pure, nomem, nostack, preserves_flags));
            (u64::from(hi) << 32) | u64::from(Self::eq_y(z.0, b))
        }
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
        unsafe fn eq_y(y: __m256i, b: u8) -> u32 {
            _mm256_cmpeq_epi8_mask(y, _mm256_set1_epi8(b as i8))
        }
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
        unsafe fn clean_z(z: Self::Z) -> bool {
            _mm256_cmpgt_epi8_mask(_mm256_min_epi8(z.0, z.1), _mm256_setzero_si256()) == u32::MAX
        }
    }

    /// 512-bit loads and compares.
    pub struct Zmm;

    impl Ops for Zmm {
        type Z = __m512i;
        type Y = __m256i;
        #[inline]
        #[target_feature(enable = "avx512f")]
        unsafe fn load_z(p: *const u8) -> __m512i {
            _mm512_loadu_si512(p as *const _)
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn load_y(p: *const u8) -> __m256i {
            _mm256_loadu_si256(p as *const __m256i)
        }
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw")]
        unsafe fn eq_z(z: __m512i, b: u8) -> u64 {
            _mm512_cmpeq_epi8_mask(z, _mm512_set1_epi8(b as i8))
        }
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
        unsafe fn eq_y(y: __m256i, b: u8) -> u32 {
            _mm256_cmpeq_epi8_mask(y, _mm256_set1_epi8(b as i8))
        }
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw")]
        unsafe fn clean_z(z: __m512i) -> bool {
            _mm512_cmpgt_epi8_mask(z, _mm512_setzero_si512()) == u64::MAX
        }
    }
}

// ------------------------------------------------------------ run-time choice

/// No mask-register reader: `fuse.rs` as before.
pub const K_OFF: u8 = 0;
/// `Vl`: 256-bit compares into `k` masks.
pub const K_VL: u8 = 1;
/// `Zmm`: 512-bit compares.
pub const K_ZMM: u8 = 2;
/// The generic reader on `Avx2`: a control for the restructure itself.
pub const K_AVX2: u8 = 3;

static MODE: AtomicU8 = AtomicU8::new(K_OFF);

#[inline(always)]
pub fn mode() -> u8 {
    MODE.load(Ordering::Relaxed)
}

/// Whether this CPU has what the AVX-512 readers and their callers use.
pub fn cpu_has_512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("lzcnt")
            && std::is_x86_feature_detected!("popcnt")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Pick the reader once at start-up. Off unless `ENGINE_K` asks:
/// `1` or `vl`, `zmm`, `avx2`. An AVX-512 mode the CPU cannot run stays
/// off. Returns the mode.
pub fn init() -> u8 {
    let want = match std::env::var("ENGINE_K").as_deref() {
        Ok("1") | Ok("vl") => K_VL,
        Ok("zmm") => K_ZMM,
        Ok("avx2") if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) => K_AVX2,
        _ => K_OFF,
    };
    let mode = match want {
        K_VL | K_ZMM if !cpu_has_512() => K_OFF,
        m => m,
    };
    MODE.store(mode, Ordering::Relaxed);
    mode
}

// ------------------------------------------------------------ the reader

/// `fuse::event_line` with the masks from `K`: the same verdict, line end,
/// bounds and `clean` for every input.
#[inline(always)]
pub unsafe fn event_line<K: Ops>(p: *const u8, out: &mut Bounds, clean: &mut bool) -> u32 {
    use fuse::{blsr, bzhi, pext, tz};
    let m = classify::<K, true>(p);
    *clean = m.clean;
    let (cf, nf, qf) = (m.c, m.n, m.q);
    let a0 = tz(cf);
    let t = blsr(cf);
    let a1 = tz(t);
    let t = blsr(t);
    let a2 = tz(t);
    let sh = a2 & 63;
    let wide = |lo: u64, hi: u32| ((((u128::from(hi)) << 64) | u128::from(lo)) >> sh) as u64;
    let cb = wide(cf, m.c2);
    let qb = wide(qf, m.q2);
    let nl = std::hint::select_unpredictable(nf != 0, tz(nf), 64 + tz(u64::from(m.n2)));
    let lb = nl.wrapping_sub(a2);
    let cl = bzhi(cb, lb);
    let mut bad = u64::from((cl.count_ones() ^ 4) | (a2 & 64) | (lb & !63));
    let t = blsr(cl);
    let b3 = tz(t);
    let t = blsr(t);
    let b4 = tz(t);
    let t = blsr(t);
    let b5 = tz(t);
    let ef = bzhi(cf, a2 + 1);
    let sf = 1 | (bzhi(cf, a1 + 1) << 1);
    let sb = cl << 1;
    let eb = blsr(cl) | (1u64 << (lb & 63));
    let open = pext(qf, sf) | (pext(qb, sb) << 3);
    let close = pext(qf << 1, ef) | (pext(qb << 1, eb) << 3);
    let stray = (bzhi(qf, a2) & !(sf | (ef >> 1))) | (bzhi(qb, lb) & !(sb | (eb >> 1)));
    bad |= stray | (open ^ close);
    let lone = fuse::bounds::<7>([u32::MAX, a0, a1, a2], [b3, b4, b5, lb], a2, open, out);
    if (bad | u64::from(lone)) != 0 {
        return REFUSED;
    }
    nl
}

/// `fuse::click_line` with the masks from `K`. Reads 64 bytes.
#[inline(always)]
pub unsafe fn click_line<K: Ops>(p: *const u8, out: &mut Bounds, clean: &mut bool) -> u32 {
    use fuse::{blsr, bzhi, pext, tz};
    let m = classify::<K, false>(p);
    *clean = m.clean;
    let (cf, nf, qf) = (m.c, m.n, m.q);
    let l = tz(nf);
    let cl = bzhi(cf, l);
    let mut bad = u64::from((cl.count_ones() ^ 3) | (l & 64));
    let a0 = tz(cl);
    let t = blsr(cl);
    let a1 = tz(t);
    let a2 = tz(blsr(t));
    let s = 1 | (cl << 1);
    let e = cl | (1u64 << (l & 63));
    let open = pext(qf, s);
    let close = pext(qf << 1, e);
    let stray = bzhi(qf, l) & !(s | (e >> 1));
    bad |= stray | (open ^ close);
    let lone = fuse::bounds::<4>([u32::MAX, a0, a1, a2], [0; 4], l, open, out);
    if (bad | u64::from(lone)) != 0 {
        return REFUSED;
    }
    l
}

/// A line reader: what `Fused` calls once per line.
pub trait Reader {
    unsafe fn line<const N: usize>(p: *const u8, out: &mut Bounds, clean: &mut bool) -> u32;
}

/// `fuse.rs` itself.
pub struct Fuse;

impl Reader for Fuse {
    #[inline(always)]
    unsafe fn line<const N: usize>(p: *const u8, out: &mut Bounds, clean: &mut bool) -> u32 {
        if N == 7 {
            fuse::event_line(p, out, clean)
        } else {
            fuse::click_line(p, out, clean)
        }
    }
}

/// The reader of this file on classifier `K`.
pub struct With<K>(std::marker::PhantomData<K>);

impl<K: Ops> Reader for With<K> {
    #[inline(always)]
    unsafe fn line<const N: usize>(p: *const u8, out: &mut Bounds, clean: &mut bool) -> u32 {
        if N == 7 {
            event_line::<K>(p, out, clean)
        } else {
            click_line::<K>(p, out, clean)
        }
    }
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

    /// A dirty field: mostly realistic values, with quotes in every place,
    /// stray commas and line breaks, bytes outside 1..=127 and odd lengths.
    fn field(state: &mut u64, k: usize) -> Vec<u8> {
        const REAL: [&[&str]; 7] = [
            &["e1234567a", "e9", "abcdefghi", "x"],
            &["u12345678", "u1", "U12345678", " u1"],
            &["2026-10-03T23:58:10Z", "2026-10-03T19:58:10-04:00", "1791071890000", "0001791071890000"],
            &["2026-10-03T23:59:10Z"],
            &["view", "cart", "purchase", "Purchase", "cart "],
            &["p123", "p49999", "p0", "P12"],
            &["100", "0250", "1.5", "-5", "10000001"],
        ];
        let r = rng(state);
        let mut f: Vec<u8> = match r % 64 {
            0 => Vec::new(),
            1 => (0..rng(state) % 40).map(|_| b"a,\"\n\x00\x80\xff9Z-:"[(rng(state) % 11) as usize]).collect(),
            2 => (0..rng(state) % 70).map(|_| b'a' + (rng(state) % 26) as u8).collect(),
            3 => b"\"".to_vec(),
            _ => {
                let pool = REAL[k.min(6)];
                pool[(rng(state) % pool.len() as u64) as usize].as_bytes().to_vec()
            }
        };
        match (r >> 8) % 40 {
            0..=7 => {
                f.insert(0, b'"');
                f.push(b'"');
            }
            8 => f.insert(0, b'"'),
            9 => f.push(b'"'),
            10 if !f.is_empty() => {
                let at = (rng(state) % f.len() as u64) as usize;
                f.insert(at, b'"');
            }
            _ => {}
        }
        f
    }

    fn line(state: &mut u64, n: usize) -> Vec<u8> {
        let count = match rng(state) % 20 {
            0 => n - 1,
            1 => n + 1,
            2 => (rng(state) % 10) as usize,
            _ => n,
        };
        let mut l = Vec::new();
        for k in 0..count {
            if k > 0 {
                l.push(b',');
            }
            l.extend(field(state, k));
        }
        if rng(state) % 50 == 0 {
            l.push(b'\r');
        }
        l
    }

    /// Run `R` and `fuse` on the same bytes; every output must agree.
    unsafe fn same<R: Reader, const N: usize>(buf: &[u8]) {
        let (mut b1, mut b2) = (Bounds::new(), Bounds::new());
        let (mut k1, mut k2) = (false, false);
        let want = Fuse::line::<N>(buf.as_ptr(), &mut b1, &mut k1);
        let got = R::line::<N>(buf.as_ptr(), &mut b2, &mut k2);
        let text = String::from_utf8_lossy(&buf[..96.min(buf.len())]);
        assert_eq!(got, want, "verdict, {N} fields: {text:?}");
        assert_eq!(k1, k2, "clean, {N} fields: {text:?}");
        if want != REFUSED {
            for i in 0..N {
                assert_eq!((b1.start(i), b1.len(i)), (b2.start(i), b2.len(i)), "field {i}: {text:?}");
            }
        }
    }

    /// The bytes from a line start: the line, a line break, the next line,
    /// then padding (zeros or junk) to 128 bytes, the reader's reach.
    fn sample(state: &mut u64, n: usize) -> Vec<u8> {
        let mut buf = line(state, n);
        if rng(state) % 30 != 0 {
            buf.push(b'\n');
        }
        buf.extend(line(state, n));
        buf.push(b'\n');
        let junk = rng(state) % 2 == 0;
        while buf.len() < 128 {
            buf.push(if junk { b"a,\"\n"[(rng(state) % 4) as usize] } else { 0 });
        }
        buf
    }

    fn sweep<R: Reader>(lines: usize, seed: u64) -> usize {
        let mut state = seed;
        let mut accepted = 0;
        for _ in 0..lines {
            let n = if rng(&mut state) % 4 == 0 { 4 } else { 7 };
            let buf = sample(&mut state, n);
            unsafe {
                if n == 7 {
                    same::<R, 7>(&buf);
                } else {
                    same::<R, 4>(&buf);
                }
                let mut b = Bounds::new();
                let mut k = false;
                accepted += usize::from(Fuse::line::<7>(buf.as_ptr(), &mut b, &mut k) != REFUSED);
            }
        }
        accepted
    }

    /// The model classifier against `fuse.rs`, on any machine.
    #[test]
    fn model_matches_fuse() {
        let lines = std::env::var("KPARSE_LINES").ok().and_then(|v| v.parse().ok()).unwrap_or(2_000_000);
        let accepted = sweep::<With<Model>>(lines, 7);
        // The sample must exercise the accepting path too.
        assert!(accepted > lines / 5, "{accepted} of {lines}");
    }

    /// The model against itself shifted through every line length near the
    /// block edges (31..33, 63..65, 95..97 bytes).
    #[test]
    fn model_at_block_edges() {
        let base = b"e1234567a,u12345678,2026-10-03T23:58:10Z,2026-10-03T23:59:10Z,view,p123,";
        for extra in 0..60 {
            for quoted in [false, true] {
                let mut l = base.to_vec();
                let tail: Vec<u8> = (0..extra).map(|i| b'0' + (i % 10) as u8).collect();
                if quoted {
                    l.push(b'"');
                    l.extend(&tail);
                    l.push(b'"');
                } else {
                    l.extend(&tail);
                }
                l.push(b'\n');
                l.resize(128, 0);
                unsafe { same::<With<Model>, 7>(&l) };
                let mut c = b"k1,u1,c_12,".to_vec();
                c.extend(&tail);
                c.push(b'\n');
                c.resize(128, 0);
                unsafe { same::<With<Model>, 4>(&c) };
            }
        }
    }

    /// The hardware classifiers against the model, where the CPU has them.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn hardware_matches_model() {
        unsafe fn masks_eq<K: Ops>(buf: &[u8]) {
            let a = classify::<Model, true>(buf.as_ptr());
            let b = classify::<K, true>(buf.as_ptr());
            assert_eq!((a.c, a.n, a.q, a.c2, a.n2, a.q2, a.clean), (b.c, b.n, b.q, b.c2, b.n2, b.q2, b.clean));
        }
        let mut state = 11;
        let have512 = cpu_has_512();
        let have2 = std::is_x86_feature_detected!("avx2");
        for _ in 0..200_000 {
            let buf = sample(&mut state, 7);
            unsafe {
                if have2 {
                    masks_eq::<Avx2>(&buf);
                }
                if have512 {
                    masks_eq::<Vl>(&buf);
                    masks_eq::<Zmm>(&buf);
                }
            }
        }
        if have2 {
            sweep::<With<Avx2>>(500_000, 5);
        }
        if have512 {
            sweep::<With<Vl>>(500_000, 3);
            sweep::<With<Zmm>>(500_000, 4);
        }
    }
}
