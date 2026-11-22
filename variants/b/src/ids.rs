//! String IDs as exact 64-bit keys.
//!
//! An ID of at most 9 bytes, all in 1..=127, packs into 63 bits: 7 bits per
//! byte, first byte highest, zero padded. That map is injective (no byte is
//! zero, so the padding cannot be confused with content) and order
//! preserving (integer order equals byte-wise order). Every generated ID
//! used to have this shape. Any other ID (longer, non-ASCII, or holding a
//! NUL) is interned and gets `FALLBACK | address of its copy`; the two key
//! spaces cannot collide and both map back to the exact bytes.

use std::cmp::Ordering;
use std::sync::{Mutex, OnceLock};

use crate::ts::load64;

pub const FALLBACK: u64 = 1 << 63;

// The intern table. A long ID is copied once into memory that is never
// freed or moved, and its key is `FALLBACK | address` of that copy, so a
// key maps back to its bytes with no lock and no lookup. The table that
// finds the copy of a given ID is split by hash into shards, each behind
// its own lock: the parser threads intern millions of long IDs at once and
// one lock would make them queue.

const SHARDS: usize = 1024;
const CHUNK: usize = 64 << 10;
const ADDRESS: u64 = (1 << 48) - 1;

struct Shard {
    /// Open addressing. 0 is empty, else 16 hash bits over a 48-bit address.
    slots: Vec<u64>,
    used: usize,
    free: *mut u8,
    left: usize,
}
unsafe impl Send for Shard {}

static INTERN: [Mutex<Shard>; SHARDS] =
    [const { Mutex::new(Shard { slots: Vec::new(), used: 0, free: std::ptr::null_mut(), left: 0 }) }; SHARDS];

#[inline(always)]
fn fold(a: u64, b: u64) -> u64 {
    let m = u128::from(a) * u128::from(b);
    (m as u64) ^ ((m >> 64) as u64)
}

/// Drawn once a run, so no input can be built to make IDs collide.
fn seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    static SEED: OnceLock<u64> = OnceLock::new();
    *SEED.get_or_init(|| std::collections::hash_map::RandomState::new().build_hasher().finish() | 1)
}

fn hash(bytes: &[u8]) -> u64 {
    let mut h = seed() ^ bytes.len() as u64;
    let mut words = bytes.chunks_exact(8);
    for w in &mut words {
        h = fold(h ^ u64::from_le_bytes(w.try_into().unwrap()), 0xD6E8_FEB8_6659_FD93);
    }
    let rest = words.remainder();
    let mut tail = [0u8; 8];
    tail[..rest.len()].copy_from_slice(rest);
    fold(h ^ u64::from_le_bytes(tail) ^ 0x2545_F491_4F6C_DD1D, 0xA076_1D64_78BD_642F)
}

/// The bytes of the copy at `p`: a length and the low hash half, then the ID.
#[inline(always)]
unsafe fn record(p: *const u8) -> &'static [u8] {
    unsafe { std::slice::from_raw_parts(p.add(8), *(p as *const u32) as usize) }
}

impl Shard {
    #[cold]
    fn grow(&mut self) {
        let size = (self.slots.len() * 2).max(256);
        let mut slots = vec![0u64; size];
        for &e in &self.slots {
            if e != 0 {
                let low = unsafe { *(((e & ADDRESS) as *const u32).add(1)) };
                let mut i = low as usize & (size - 1);
                while slots[i] != 0 {
                    i = (i + 1) & (size - 1);
                }
                slots[i] = e;
            }
        }
        self.slots = slots;
    }
}

#[inline(never)]
fn intern(bytes: &[u8]) -> u64 {
    let h = hash(bytes);
    let tag = (h >> 32) & 0xFFFF;
    let mut guard = INTERN[(h >> 54) as usize].lock().unwrap();
    let s = &mut *guard;
    if s.used * 2 >= s.slots.len() {
        s.grow();
    }
    let mask = s.slots.len() - 1;
    let mut i = h as u32 as usize & mask;
    loop {
        let e = s.slots[i];
        if e == 0 {
            break;
        }
        if e >> 48 == tag && unsafe { record((e & ADDRESS) as *const u8) } == bytes {
            return FALLBACK | (e & ADDRESS);
        }
        i = (i + 1) & mask;
    }
    let need = (8 + bytes.len() + 7) & !7;
    if s.left < need {
        let size = need.max(CHUNK);
        let block: &'static mut [u64] = Vec::leak(vec![0u64; size / 8]);
        s.free = block.as_mut_ptr() as *mut u8;
        s.left = size;
    }
    let p = s.free;
    assert!(p as u64 <= ADDRESS && bytes.len() <= u32::MAX as usize);
    unsafe {
        *(p as *mut u32) = bytes.len() as u32;
        *(p as *mut u32).add(1) = h as u32;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.add(8), bytes.len());
        s.free = p.add(need);
    }
    s.left -= need;
    s.slots[i] = tag << 48 | p as u64;
    s.used += 1;
    FALLBACK | p as u64
}

const LEN_MASK: [u64; 10] = [
    0,
    0xFF,
    0xFFFF,
    0xFF_FFFF,
    0xFFFF_FFFF,
    0xFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF,
    0xFF_FFFF_FFFF_FFFF,
    u64::MAX,
    u64::MAX,
];

/// The key of the `len` bytes at `p`. May read 16 bytes from `p`.
#[inline(always)]
pub unsafe fn pack(p: *const u8, len: usize) -> u64 {
    if len <= 9 {
        let mask = *LEN_MASK.get_unchecked(len);
        let v = load64(p) & mask;
        let ninth = u64::from(*p.add(8)) & (0u64.wrapping_sub(u64::from(len == 9)));
        // Every byte of the ID must be in 1..=127. Bytes past the ID are
        // set to 1 so they pass; a zero byte borrows and sets its high bit.
        let filled = v | (!mask & 0x0101_0101_0101_0101);
        let bad = (filled.wrapping_sub(0x0101_0101_0101_0101) | filled | ninth) & 0x8080_8080_8080_8080;
        if bad == 0 && (len != 9 || ninth != 0) {
            return (squeeze(v.swap_bytes()) << 7) | ninth;
        }
    }
    intern(std::slice::from_raw_parts(p, len))
}

/// `pack` for an ID whose bytes the caller already knows to be in 1..=127
/// (stage 1 checks whole windows): the same key without the byte checks.
#[inline(always)]
pub unsafe fn pack_known(p: *const u8, len: usize) -> u64 {
    if len <= 9 {
        // All nine bytes packed, then the key bits past the ID cleared:
        // the byte swap folds into the load (`movbe`), and one mask does
        // the work of the length mask and the ninth byte select. Bytes 1
        // to 8 go through `squeeze`, which drops their high bits; byte 0
        // is the ID's own (in 1..=127) unless `len` is 0, and then the
        // mask is 0.
        let all = squeeze(load64(p.add(1)).swap_bytes()) | (u64::from(*p) << 56);
        return all & *KEY_MASK.get_unchecked(len);
    }
    intern(std::slice::from_raw_parts(p, len))
}

/// The key bits of the first `len` bytes: 7 a byte from bit 62 down.
const KEY_MASK: [u64; 10] = {
    let mut m = [0u64; 10];
    let mut len = 1;
    while len <= 9 {
        m[len] = ((1u64 << (7 * len)) - 1) << (63 - 7 * len);
        len += 1;
    }
    m
};

/// `pack`, by the short route when `known` says the bytes are in 1..=127.
#[inline(always)]
pub unsafe fn pack_in(p: *const u8, len: usize, known: bool) -> u64 {
    if known {
        pack_known(p, len)
    } else {
        pack(p, len)
    }
}

/// Drop the high bit of each of eight bytes: 64 bits to 56, order kept.
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
fn squeeze(x: u64) -> u64 {
    unsafe { core::arch::x86_64::_pext_u64(x, 0x7F7F_7F7F_7F7F_7F7F) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
#[inline(always)]
fn squeeze(x: u64) -> u64 {
    let x = ((x & 0x7F00_7F00_7F00_7F00) >> 1) | (x & 0x007F_007F_007F_007F);
    let x = ((x & 0x3FFF_0000_3FFF_0000) >> 2) | (x & 0x0000_3FFF_0000_3FFF);
    ((x & 0x0FFF_FFFF_0000_0000) >> 4) | (x & 0x0000_0000_0FFF_FFFF)
}

/// Safe wrapper for tests and cold paths.
pub fn pack_slice(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 16];
    if bytes.len() <= 9 {
        buf[..bytes.len()].copy_from_slice(bytes);
        unsafe { pack(buf.as_ptr(), bytes.len()) }
    } else {
        intern(bytes)
    }
}

/// Bytes of a packed (not fallback) key into `out[..9]`; returns the length.
#[inline(always)]
pub fn unpack_packed(key: u64, out: &mut [u8; 16]) -> usize {
    debug_assert!(key & FALLBACK == 0);
    let ninth = (key & 0x7F) as u8;
    let x = key >> 7;
    let x = ((x & 0x00FF_FFFF_F000_0000) << 4) | (x & 0x0000_0000_0FFF_FFFF);
    let x = ((x & 0x0FFF_C000_0FFF_C000) << 2) | (x & 0x0000_3FFF_0000_3FFF);
    let x = ((x & 0x3F80_3F80_3F80_3F80) << 1) | (x & 0x007F_007F_007F_007F);
    out[..8].copy_from_slice(&x.swap_bytes().to_le_bytes());
    out[8] = ninth;
    if key == 0 {
        0
    } else {
        9 - (key.trailing_zeros() / 7) as usize
    }
}

/// The exact bytes of any key. `buf` holds them when the key is packed.
#[inline(always)]
fn key_bytes(key: u64, buf: &mut [u8; 16]) -> &[u8] {
    if key & FALLBACK == 0 {
        let len = unpack_packed(key, buf);
        &buf[..len]
    } else {
        unsafe { record((key & ADDRESS) as *const u8) }
    }
}

/// Append the exact bytes of any key.
pub fn unpack(key: u64, out: &mut Vec<u8>) {
    let mut buf = [0u8; 16];
    out.extend_from_slice(key_bytes(key, &mut buf));
}

#[inline(never)]
fn cmp_slow(a: u64, b: u64) -> Ordering {
    let (mut x, mut y) = ([0u8; 16], [0u8; 16]);
    key_bytes(a, &mut x).cmp(key_bytes(b, &mut y))
}

/// Byte-wise order of the IDs two keys stand for.
#[inline(always)]
pub fn cmp_ids(a: u64, b: u64) -> Ordering {
    if (a | b) & FALLBACK == 0 {
        a.cmp(&b)
    } else if a == b {
        Ordering::Equal
    } else {
        cmp_slow(a, b)
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

    fn sample(state: &mut u64) -> Vec<u8> {
        let len = (rng(state) % 13) as usize;
        let style = rng(state) % 8;
        (0..len)
            .map(|_| match style {
                0 => (rng(state) % 256) as u8,
                1 => b"\x00\x01\x7f\x80\xff a"[(rng(state) % 7) as usize],
                2 => b"uU"[(rng(state) % 2) as usize],
                _ => b"0123456789abcdefghijklmnopqrstuvwxyzAZ _\t\r"[(rng(state) % 42) as usize],
            })
            .collect()
    }

    #[test]
    fn round_trip_and_order() {
        let mut state = 3;
        let mut ids: Vec<Vec<u8>> = [
            "", "u", "u1", "U1", " u1", "u1 ", "uduffvac", "e42huhoug", "e42huhou", "e42huhougx", "A\tB", "e3\r",
            "user_id_a", "user_id_b", "\u{ff12}", "zzzzzzzzz", "zzzzzzzzzz", "\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f",
            "\x01", "\x01\x01",
        ]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
        ids.push(vec![b'a', 0]);
        ids.push(vec![0]);
        ids.push(vec![b'a', 0, b'b']);
        ids.push(vec![b'a'; 9]);
        ids.push(vec![b'a', b'a', b'a', b'a', b'a', b'a', b'a', b'a', 0]);
        ids.push(vec![b'a', b'a', b'a', b'a', b'a', b'a', b'a', b'a', 0x80]);
        for _ in 0..20_000 {
            ids.push(sample(&mut state));
        }
        let keys: Vec<u64> = ids.iter().map(|id| pack_slice(id)).collect();
        for (id, &key) in ids.iter().zip(&keys) {
            let mut back = Vec::new();
            unpack(key, &mut back);
            assert_eq!(&back, id, "round trip of {id:?}");
            assert_eq!(pack_slice(id), key, "stable key for {id:?}");
            let packable = id.len() <= 9 && id.iter().all(|&b| (1..=127).contains(&b));
            assert_eq!(key & FALLBACK == 0, packable, "{id:?}");
        }
        for i in 0..ids.len() {
            for j in (0..ids.len()).step_by(37) {
                assert_eq!(keys[i] == keys[j], ids[i] == ids[j], "{:?} {:?}", ids[i], ids[j]);
                assert_eq!(cmp_ids(keys[i], keys[j]), ids[i].cmp(&ids[j]), "{:?} {:?}", ids[i], ids[j]);
            }
        }
    }

    /// Eight threads intern the same two million long IDs at once, each in
    /// its own order. Every shard grows its table several times and fills
    /// more than one chunk on the way.
    #[test]
    fn threads_agree_through_growth_and_chunk_changes() {
        const N: usize = 2_000_000;
        const THREADS: usize = 8;
        fn id(i: usize) -> Vec<u8> {
            let (mut a, mut b) = (i as u64, !(i as u64));
            let (x, y) = (rng(&mut a), rng(&mut b));
            format!("u{x:016x}{y:016x}").into_bytes()
        }
        let keys: Vec<Vec<u64>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..THREADS)
                .map(|t| {
                    scope.spawn(move || {
                        let mut keys = vec![0u64; N];
                        for step in 0..N {
                            let i = (t * (N / THREADS) + step) % N;
                            keys[i] = pack_slice(&id(i));
                        }
                        keys
                    })
                })
                .collect();
            workers.into_iter().map(|w| w.join().unwrap()).collect()
        });
        for t in 1..THREADS {
            assert!(keys[t] == keys[0], "thread {t} got other keys than thread 0");
        }
        let keys = &keys[0];
        let mut distinct = keys.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), N, "two IDs share a key");
        {
            let shard = INTERN[0].lock().unwrap();
            assert!(shard.slots.len() >= 2048, "table of shard 0 grew to {}", shard.slots.len());
            assert!(shard.used * 48 > CHUNK, "shard 0 holds {} IDs, one chunk", shard.used);
        }
        let mut back = Vec::new();
        for (i, &key) in keys.iter().enumerate() {
            assert!(key & FALLBACK != 0);
            back.clear();
            unpack(key, &mut back);
            assert_eq!(back, id(i), "round trip of ID {i}");
        }
        for i in (0..N).step_by(997) {
            let j = (i * 31 + 7) % N;
            assert_eq!(cmp_ids(keys[i], keys[j]), id(i).cmp(&id(j)), "order of {i} and {j}");
        }
        // IDs around the size of a chunk, and far over it.
        for len in [CHUNK - 16, CHUNK - 9, CHUNK - 8, CHUNK - 7, CHUNK, 70_000, 200_000] {
            let big: Vec<u8> = (0..len).map(|n| b'a' + (n % 23) as u8).collect();
            let key = pack_slice(&big);
            assert_eq!(pack_slice(&big), key, "stable key at {len} bytes");
            back.clear();
            unpack(key, &mut back);
            assert_eq!(back, big, "round trip at {len} bytes");
        }
        for i in (0..N).step_by(101) {
            assert_eq!(pack_slice(&id(i)), keys[i], "key of ID {i} after the big ones");
        }
    }

    #[test]
    fn pack_ignores_bytes_past_the_field() {
        let mut state = 9;
        for _ in 0..50_000 {
            let id = sample(&mut state);
            if id.len() > 9 {
                continue;
            }
            let mut a = id.clone();
            let mut b = id.clone();
            a.extend_from_slice(&[0xFF; 16]);
            b.extend_from_slice(b",\"\n0123456789abc");
            unsafe {
                assert_eq!(pack(a.as_ptr(), id.len()), pack(b.as_ptr(), id.len()), "{id:?}");
                // The short route gives the same key for every ID it may
                // be used on.
                if id.iter().all(|&c| (1..=127).contains(&c)) {
                    assert_eq!(pack_known(a.as_ptr(), id.len()), pack(a.as_ptr(), id.len()), "{id:?}");
                    assert_eq!(pack_known(b.as_ptr(), id.len()), pack(a.as_ptr(), id.len()), "{id:?}");
                    assert_eq!(pack_in(b.as_ptr(), id.len(), true), pack_in(b.as_ptr(), id.len(), false), "{id:?}");
                }
            }
        }
    }
}
