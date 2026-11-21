//! IDs as numbers: the codec of the indexed shape (`ENGINE_INDEX=1`).
//!
//! A canonical ID is one prefix byte and the lowercase base 36 of a value
//! `y < 2^40` with no leading zero. `y` is a bijection of a 40-bit number
//! `v` (`scramble` below, keyed). Whatever the key, canonical text, `y`
//! and `v` are in one-to-one correspondence, so a wrong key can only make
//! the numbers large (and send IDs down the general path), never wrong.
//! The key is recovered from the data: pairs of IDs whose numbers are
//! known to be close (two IDs of one link) pin it bit by bit from the
//! bottom.

pub const BITS: u32 = 40;
pub const MASK: u64 = (1 << BITS) - 1;
const M1: u64 = 0x9E37_79B9_7F4B;
const M2: u64 = 0xD6E8_FEB8_6659;

/// The inverse of an odd number modulo 2^64 (Newton).
const fn inverse(a: u64) -> u64 {
    let mut x = a;
    let mut i = 0;
    while i < 6 {
        x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
        i += 1;
    }
    x
}
const M1I: u64 = inverse(M1) & MASK;
const M2I: u64 = inverse(M2) & MASK;

/// The bijection on `0..2^40` that turns a number into the value written.
#[inline(always)]
pub fn scramble(v: u64, key: u64) -> u64 {
    let mut m = (v.wrapping_mul(M1) ^ key) & MASK;
    m ^= m >> 17;
    m.wrapping_mul(M2) & MASK
}

/// The keyless half of the inverse: `y` back to `v * M1 ^ key` (mod 2^40).
#[inline(always)]
pub fn unmix(y: u64) -> u64 {
    let z = y.wrapping_mul(M2I) & MASK;
    z ^ (z >> 17) ^ (z >> 34)
}

/// `v` from `unmix(y)` and the key.
#[inline(always)]
pub fn number(g: u64, key: u64) -> u64 {
    ((g ^ key).wrapping_mul(M1I)) & MASK
}

/// The inverse of `scramble`.
#[inline(always)]
pub fn unscramble(y: u64, key: u64) -> u64 {
    number(unmix(y), key)
}

/// `y` of a packed key (see `ids.rs`: 7 bits a byte, first byte from bit
/// 62 down, zero padded) whose ID is canonical with this prefix, else
/// `None`. Exact: a key is accepted only when it is the packed form of
/// `prefix` followed by the base 36 of the value returned, so the map is
/// injective on what it accepts.
#[inline(always)]
pub fn value_of(key: u64, prefix: u8) -> Option<u64> {
    // Bit 63 (an interned key) or another first byte: not canonical.
    if key >> 56 != u64::from(prefix) {
        return None;
    }
    let mut y = 0u64;
    let mut n = 0u32;
    while n < 8 {
        let c = ((key >> (49 - 7 * n)) & 0x7F) as u8;
        let d = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'z' => c - b'a' + 10,
            0 => break,
            _ => return None,
        };
        if n == 1 && y == 0 {
            // A leading zero.
            return None;
        }
        y = y * 36 + u64::from(d);
        n += 1;
    }
    if n == 0 || y > MASK {
        return None;
    }
    // Everything after the last digit must be padding. With 8 digits the
    // last one ends at bit 0.
    if n < 8 && key & ((1u64 << (56 - 7 * (n + 1) + 7)) - 1) != 0 {
        return None;
    }
    Some(y)
}

/// The packed key of `prefix` and the base 36 of `y` (`y <= MASK`).
pub fn key_of(prefix: u8, y: u64) -> u64 {
    let mut digits = [0u8; 8];
    let mut n = 0;
    let mut rest = y;
    loop {
        let d = (rest % 36) as u8;
        digits[n] = if d < 10 { b'0' + d } else { b'a' + d - 10 };
        n += 1;
        rest /= 36;
        if rest == 0 {
            break;
        }
    }
    let mut key = u64::from(prefix) << 56;
    for i in 0..n {
        key |= u64::from(digits[n - 1 - i]) << (49 - 7 * i);
    }
    key
}

/// Candidate keys for which most pairs of `g` values (`unmix` outputs) are
/// numbers less than `2^spread` apart, built bit by bit from the bottom:
/// the difference of two numbers modulo `2^k` depends only on the low `k`
/// bits of the key. `None` when no key fits or the search grows too wide.
pub fn recover(pairs: &[(u64, u64)], spread: u32) -> Option<Vec<u64>> {
    let pairs = &pairs[..pairs.len().min(64)];
    if pairs.len() < 16 || spread + 2 >= BITS {
        return None;
    }
    let need = pairs.len() * 5 / 8;
    let lim = 1i64 << spread;
    let k0 = spread + 1;
    let mut cands: Vec<u64> = (0..1u64 << k0).collect();
    let mut next: Vec<u64> = Vec::new();
    for k in k0 + 1..=BITS {
        next.clear();
        let mk = (1u64 << k) - 1;
        for &c in &cands {
            for bit in [0, 1u64 << (k - 1)] {
                let kk = c | bit;
                let mut ok = 0;
                for &(a, b) in pairs {
                    let d = ((a ^ kk).wrapping_mul(M1I).wrapping_sub((b ^ kk).wrapping_mul(M1I))) & mk;
                    // Centred: the difference as a signed k-bit number.
                    let d = ((d << (64 - k)) as i64) >> (64 - k);
                    ok += usize::from(d > -lim && d < lim);
                }
                if ok >= need {
                    next.push(kk);
                }
            }
        }
        std::mem::swap(&mut cands, &mut next);
        if cands.is_empty() || cands.len() > 1 << 16 {
            return None;
        }
    }
    Some(cands)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids;

    fn rng(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn base36(v: u64) -> String {
        let digits = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let mut out = Vec::new();
        let mut r = v;
        loop {
            out.push(digits[(r % 36) as usize]);
            r /= 36;
            if r == 0 {
                break;
            }
        }
        out.reverse();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn scramble_inverts() {
        let mut s = 1;
        for _ in 0..100_000 {
            let (v, key) = (rng(&mut s) & MASK, rng(&mut s));
            assert_eq!(unscramble(scramble(v, key), key & MASK), v);
            assert_eq!(unscramble(scramble(v, key), key), v);
        }
    }

    /// `value_of` accepts exactly the canonical IDs, and `key_of` gives
    /// back the packed key of the text.
    #[test]
    fn codec_is_exact() {
        let mut s = 7;
        let mut texts: Vec<String> = vec![
            "u".into(), "u0".into(), "u00".into(), "u01".into(), "u1".into(), "uz".into(), "uZ".into(), "U1".into(),
            "u1 ".into(), " u1".into(), "u-1".into(), "uzzzzzzzz".into(), "u00000000".into(), "u100000000".into(),
            base36(MASK).replace("", "").to_string(), format!("u{}", base36(MASK)), format!("u{}", base36(MASK + 1)),
            "e1".into(), "uA".into(), "u\x7f".into(), "u{".into(), "u/".into(), "u:".into(), "u`".into(),
        ];
        for _ in 0..200_000 {
            let r = rng(&mut s);
            let t = match r % 4 {
                0 => format!("u{}", base36(rng(&mut s) & MASK)),
                1 => format!("u{}", base36(rng(&mut s) % 5000)),
                2 => {
                    let len = (rng(&mut s) % 11) as usize;
                    let set = b"u0123456789abcdefghijklmnopqrstuvwxyzU_ \x01";
                    (0..len).map(|_| set[(rng(&mut s) % set.len() as u64) as usize] as char).collect()
                }
                _ => format!("u0{}", base36(rng(&mut s) % 100000)),
            };
            texts.push(t);
        }
        for t in &texts {
            let key = ids::pack_slice(t.as_bytes());
            let want = {
                let b = t.as_bytes();
                if b.len() >= 2 && b[0] == b'u' && b[1..].iter().all(|c| c.is_ascii_digit() || c.is_ascii_lowercase()) && (b.len() == 2 || b[1] != b'0') {
                    let v = u128::from_str_radix(&t[1..], 36).unwrap();
                    (v <= u128::from(MASK)).then_some(v as u64)
                } else {
                    None
                }
            };
            assert_eq!(value_of(key, b'u'), want, "{t:?}");
            if let Some(y) = want {
                assert_eq!(key_of(b'u', y), key, "{t:?}");
            }
        }
    }

    #[test]
    fn recovers_a_key_from_close_pairs() {
        let mut s = 99;
        for round in 0..8 {
            let key = rng(&mut s) & MASK;
            let pairs: Vec<(u64, u64)> = (0..64)
                .map(|i| {
                    let p = rng(&mut s) % 700_000;
                    let (a, b) = (p * 512 + rng(&mut s) % 8, p * 512 + rng(&mut s) % 8);
                    // A few pairs far apart, as kiosk links are.
                    let b = if i % 16 == 0 { (1 << 39) + 3 } else { b };
                    (unmix(scramble(a, key)), unmix(scramble(b, key)))
                })
                .collect();
            let cands = recover(&pairs, 9).unwrap();
            assert!(cands.contains(&key), "round {round}: {} candidates", cands.len());
        }
    }
}
