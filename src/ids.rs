//! String IDs as exact 64-bit keys.
//!
//! An ID of at most 9 bytes, all in 1..=127, packs into 63 bits: 7 bits per
//! byte, first byte highest, zero padded. That map is injective (no byte is
//! zero, so the padding cannot be confused with content) and order
//! preserving (integer order equals byte-wise order). Every generated ID
//! has this shape. Any other ID (longer, non-ASCII, or holding a NUL) is
//! interned in a global table and gets `FALLBACK | index`; the two key
//! spaces cannot collide and both map back to the exact bytes.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::ts::load64;

pub const FALLBACK: u64 = 1 << 63;

struct Interner {
    map: Option<HashMap<Vec<u8>, u32>>,
    list: Vec<Vec<u8>>,
}

static INTERN: Mutex<Interner> = Mutex::new(Interner { map: None, list: Vec::new() });

#[cold]
#[inline(never)]
fn intern(bytes: &[u8]) -> u64 {
    let mut guard = INTERN.lock().unwrap();
    let it = &mut *guard;
    let map = it.map.get_or_insert_with(HashMap::new);
    if let Some(&i) = map.get(bytes) {
        return FALLBACK | u64::from(i);
    }
    let i = it.list.len() as u32;
    it.list.push(bytes.to_vec());
    map.insert(bytes.to_vec(), i);
    FALLBACK | u64::from(i)
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

/// Append the exact bytes of any key.
pub fn unpack(key: u64, out: &mut Vec<u8>) {
    if key & FALLBACK == 0 {
        let mut buf = [0u8; 16];
        let len = unpack_packed(key, &mut buf);
        out.extend_from_slice(&buf[..len]);
    } else {
        let guard = INTERN.lock().unwrap();
        out.extend_from_slice(&guard.list[(key & !FALLBACK) as usize]);
    }
}

#[cold]
#[inline(never)]
fn cmp_slow(a: u64, b: u64) -> Ordering {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    unpack(a, &mut x);
    unpack(b, &mut y);
    x.cmp(&y)
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
            }
        }
    }
}
