//! Probe modes for an on-scorer diagnostic (`ENGINE_PROBE`, unset by
//! default). Each runs the parse phase in a cut-down form, skips the link
//! files, prints the `parse` lap and exits 0; no output is written.
//!
//! - `read`: every event and click line found as the kernels find it (the
//!   same chunks, the same three 32-byte loads and two prefetches per line,
//!   the line break from the masks), a byte summed per line, no field work,
//!   no writes. The input streaming floor.
//! - `nowrite`: the asm event and click kernels in full (`ENGINE_ASM` is
//!   implied), every field parsed and checked, but the entries go to one
//!   small per-thread staging buffer that is overwritten (2 partitions of
//!   255 entries, never moved out) and clicks to a 1024-click ring. Lines
//!   the kernels refuse (about 2%) still go through the old loop and are
//!   written to the real scatter, and purchases are priced and dropped.
//! - `noparse`: the line finding of `read`, then a fixed 32-byte entry per
//!   event line pushed into the real partition (`push_hot`, a multiply of
//!   the line address picks the partition: same staging, flushes, streaming
//!   stores and chunk growth), and a fixed click appended to the real click
//!   blocks. No field parsing.

use super::*;

pub(in crate::engine) const READ: u8 = 1;
pub(in crate::engine) const NOWRITE: u8 = 2;
pub(in crate::engine) const NOPARSE: u8 = 3;

/// The probe mode: 0 (none, the default), `READ`, `NOWRITE` or `NOPARSE`.
pub(in crate::engine) fn probe() -> u8 {
    static P: OnceLock<u8> = OnceLock::new();
    *P.get_or_init(|| match std::env::var("ENGINE_PROBE").as_deref() {
        Ok("read") => READ,
        Ok("nowrite") => NOWRITE,
        Ok("noparse") => NOPARSE,
        _ => 0,
    })
}

/// Where `nowrite` sends the event kernel's entries.
#[repr(C, align(64))]
pub(super) struct Sink {
    pub(super) stage: [u8; 2 * 255 * 32],
    pub(super) fill: [u8; 64],
    /// Two `Part` records with `dst == end`, so a full buffer leaves the
    /// kernel instead of being moved out.
    pub(super) parts: [u64; 8],
    pub(super) clicks: [Click; 1024],
}

impl Sink {
    pub(super) fn new() -> Box<Sink> {
        Box::new(Sink { stage: [0; 2 * 255 * 32], fill: [0; 64], parts: [0; 8], clicks: [Click { key: 0, ckey: 0, tc: 0 }; 1024] })
    }

    /// Points the event kernel's staging at the sink.
    pub(super) fn aim(&mut self, c: &mut Ctx) {
        c.fill = self.fill.as_mut_ptr();
        c.stage = self.stage.as_mut_ptr();
        c.sbytes = 255 * 32;
        c.per = 255;
        c.parts = self.parts.as_mut_ptr() as *mut u8;
        c.shift1 = 63;
    }

    pub(super) fn click_room(&mut self) -> (*mut Click, *mut Click) {
        let a = self.clicks.as_mut_ptr();
        (a, unsafe { a.add(1024) })
    }
}

/// The offset of the first line break in the 96 bytes at `p`, 96 if none:
/// the kernel's three loads.
#[inline(always)]
unsafe fn newline96(p: *const u8) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::*;
        let nl = _mm256_set1_epi8(b'\n' as i8);
        let m = |at: usize| u64::from(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(p.add(at) as *const __m256i), nl)) as u32);
        let lo = m(0) | (m(32) << 32);
        let hi = m(64) | (1 << 32);
        if lo != 0 { u64::from(lo.trailing_zeros()) } else { 64 + u64::from(hi.trailing_zeros()) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        (0..96).find(|&i| *p.add(i) == b'\n').unwrap_or(96) as u64
    }
}

impl<'a> ParseState<'a> {
    /// `Parser::run_chunk` for the event or click lines of `m` that start in
    /// `[a, b)`, in the probe mode `probe()`.
    pub(in crate::engine) unsafe fn probe_chunk(&mut self, scan: &mut Scan, m: Mapped, a: usize, b: usize, kind: Kind) {
        let len = m.len;
        let from = a.saturating_sub(1);
        let Some(first) = find_byte(std::slice::from_raw_parts(m.ptr.add(from), b.min(len) - from), b'\n') else {
            return;
        };
        let mut pos = from + first + 1;
        if probe() == NOWRITE {
            while pos < b {
                pos = match kind {
                    Kind::Events => self.events_hot(m, pos, b, scan),
                    _ => self.clicks_hot(m, pos, b, scan),
                };
                if pos < b {
                    let end = find_byte(std::slice::from_raw_parts(m.ptr.add(pos), len - pos), b'\n').map_or(len, |i| pos + i);
                    self.long_line(std::slice::from_raw_parts(m.ptr.add(pos), end - pos), kind);
                    pos = end + 1;
                }
            }
            return;
        }
        let write = probe() == NOPARSE;
        let base = m.ptr;
        let hot = self.entries.hot();
        let shift1 = self.shift1;
        let clicks: *mut Vec<Vec<Click>> = &mut self.clicks;
        let (mut at, mut stop) = clicks::room(&mut *clicks);
        let mut acc = 0u64;
        while pos < b {
            let p = base.add(pos);
            sys::prefetch(p.wrapping_add(FUSED_AHEAD));
            sys::prefetch(p.wrapping_add(FUSED_AHEAD + 64));
            let nl = newline96(p) as usize;
            let next = if nl < 96 {
                pos + nl + 1
            } else {
                let from = (pos + 96).min(len);
                find_byte(std::slice::from_raw_parts(base.add(from), len - from), b'\n').map_or(len, |i| from + i + 1)
            };
            acc = acc.wrapping_add(u64::from(*p)).wrapping_add(nl as u64);
            if write {
                let key = p as u64;
                if kind == Kind::Events {
                    let part = (key.wrapping_mul(0xD6E8_FEB8_6659_FD93) >> shift1) as usize;
                    self.entries.push_hot(&hot, part, Entry { ekey: key, ukey: acc, ingest: 0, w3: DEAD });
                } else {
                    if at == stop {
                        if let Some(v) = (*clicks).last_mut() {
                            v.set_len(CLICK_BLOCK);
                        }
                        let mut v = Vec::with_capacity(CLICK_BLOCK);
                        at = v.as_mut_ptr();
                        stop = at.add(CLICK_BLOCK);
                        (*clicks).push(v);
                    }
                    at.write(Click { key, ckey: acc, tc: 0 });
                    at = at.add(1);
                }
            }
            pos = next;
        }
        clicks::close(&mut *clicks, at);
        std::hint::black_box(acc);
    }
}
