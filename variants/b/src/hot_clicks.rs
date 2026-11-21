//! The click loop in assembly (`ENGINE_ASM=1`, off by default): the same
//! job as `ParseState::clicks::<0, _>` over `Fused`, by the method of the
//! event kernel in `hot.rs`.
//!
//! The common line (four fields, quotes only around whole fields, the line
//! break at bytes 32 to 63, bytes 0..32 in 11..=127 and 32..64 in 1..=127,
//! IDs of 1 to 9 bytes, a campaign ID of 1 to 7 bytes, `ts` of 13, 20
//! or 25 bytes whose month is in the date cache) runs through `ckernel`,
//! which drops or appends it. Any other line goes, alone, to the old loop
//! (`slow_line`). A full click block leaves the loop and gets a new one
//! here, as in `clicks`.
//!
//! SAFETY (reads): 64 bytes from a line start (masks); an accepted line
//! ends before byte 64, so each field starts before byte 64, and the field
//! reads (9 bytes from an ID start, 8 from the campaign start, 16 from
//! `tp` or `tp + 4`, 8 from `tp + 17`) end before byte 64 + 25 = 89 from a
//! line start inside the file; a mapping has a readable page after the
//! file (`sys::map_file`), the tests pad with 4096 zero bytes. SAFETY
//! (writes): one 24-byte `Click` at `at`, only when `at != astop`, the end
//! of the last block's capacity, as in `clicks`.

use super::*;

/// Runs click lines from `*p` until `stop` or a line it leaves the loop
/// for; returns why. `c.at` is the append position, in and out.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2", target_feature = "popcnt", target_feature = "movbe"))]
#[inline(never)]
unsafe fn ckernel(c: &mut Ctx, p: &mut *const u8) -> u32 {
    let code: u32;
    core::arch::asm!(
        "push rbx",
        "push rbp",
        "mov rbp, [rdi + {at}]",
        "vmovdqa ymm9, [rdi + {v_sub1}]",
        "vmovdqa ymm10, [rdi + {v_madd}]",
        "vmovdqa ymm11, [rdi + {v_ten}]",
        "vpxor ymm12, ymm12, ymm12",
        "vmovdqa ymm13, [rdi + {v_comma}]",
        "vmovdqa ymm14, [rdi + {v_nl}]",
        "vmovdqa ymm15, [rdi + {v_quote}]",
        "xor eax, eax",
        "cmp rsi, [rdi + {stop}]",
        "jae 99f",
        ".p2align 5",
        "12:",
        "prefetcht0 [rsi + 1024]",
        "vmovdqu ymm0, [rsi]",
        "vmovdqu ymm1, [rsi + 32]",
        "vpcmpgtb ymm3, ymm0, ymm11",
        "vpcmpgtb ymm4, ymm1, ymm12",
        "vpand ymm3, ymm3, ymm4",
        "vpcmpeqb ymm4, ymm0, ymm13",
        "vpcmpeqb ymm5, ymm1, ymm13",
        "vpcmpeqb ymm0, ymm0, ymm15",
        "vpcmpeqb ymm6, ymm1, ymm15",
        "vpcmpeqb ymm1, ymm1, ymm14",
        "vpmovmskb r8d, ymm3",
        "not r8d",
        "vpmovmskb eax, ymm4",
        "vpmovmskb edx, ymm5",
        "shl rdx, 32",
        "or rax, rdx",
        "vpmovmskb r9d, ymm0",
        "vpmovmskb edx, ymm6",
        "shl rdx, 32",
        "or r9, rdx",
        "vpmovmskb edx, ymm1",
        // line break at 32 + rdx; 32 + 32 when there is none
        "tzcnt edx, edx",
        "lea rcx, [rdx + 32]",
        "mov r10d, edx",
        "and r10d, 32",
        "or r8, r10",
        "bzhi rax, rax, rcx",
        "popcnt r10, rax",
        "xor r10, 3",
        "or r8, r10",
        "bts rax, rcx",
        "bzhi r9, r9, rcx",
        // the window: d rax, q r9
        "lea r10, [rax + rax + 1]",
        "mov r11, r10",
        "and r11, r9",
        "lea r15, [r9 + r9]",
        "and r15, rax",
        "lea rbx, [r11 - 1]",
        "sub rbx, rax",
        "and rbx, rax",
        "xor rbx, r15",
        "or r8, rbx",
        "andn rbx, r11, r9",
        "add rbx, rbx",
        "xor rbx, r15",
        "or r8, rbx",
        "add r10, r11",
        "shr r15, 1",
        "sub rax, r15",
        "test r8, r8",
        "jnz 20f",
        // fields 0 and 1: the IDs
        "tzcnt r12, r10",
        "tzcnt r14, rax",
        "blsr r10, r10",
        "blsr rax, rax",
        "tzcnt r15, r10",
        "tzcnt rbx, rax",
        "blsr r10, r10",
        "blsr rax, rax",
        "sub r14, r12",
        "sub rbx, r15",
        "lea r8, [r14 - 1]",
        "cmp r8, 8",
        "ja 20f",
        "lea r8, [rbx - 1]",
        "cmp r8, 8",
        "ja 20f",
        "movbe r8, [rsi + r12 + 1]",
        "pext r8, r8, [rdi + {m7f}]",
        "movzx r11d, byte ptr [rsi + r12]",
        "shl r11, 56",
        "or r8, r11",
        "and r8, [rdi + {km} + r14 * 8]",
        "movbe r9, [rsi + r15 + 1]",
        "pext r9, r9, [rdi + {m7f}]",
        "movzx r11d, byte ptr [rsi + r15]",
        "shl r11, 56",
        "or r9, r11",
        "and r9, [rdi + {km} + rbx * 8]",
        // fields 2 and 3
        "tzcnt r12, r10",
        "tzcnt r14, rax",
        "blsr r10, r10",
        "blsr rax, rax",
        "tzcnt r10, r10",
        "tzcnt rax, rax",
        "sub r14, r12",
        "sub rax, r10",
        // campaign: an ID of 1 to 7 bytes by one probe of `Fast::camp_tab`
        // (r12 start, r14 length); a miss drops the click
        "add r12, rsi",
        "lea r13, [r14 - 1]",
        "cmp r13, 6",
        "ja 20f",
        "lea rcx, [r14 * 8]",
        "bzhi r13, [r12], rcx",
        "shl r14, 56",
        "or r13, r14",
        "mov rcx, r13",
        "imul rcx, [rdi + {camp_mul}]",
        "mov r11, [rdi + {camp_shift}]",
        "shrx rcx, rcx, r11",
        "mov r11, [rdi + {camp_tab}]",
        "50:",
        "mov r15, rcx",
        "shl r15, 4",
        "cmp r13, [r11 + r15]",
        "je 51f",
        "cmp qword ptr [r11 + r15], 0",
        "je 40f",
        "inc rcx",
        "and rcx, [rdi + {camp_mask}]",
        "jmp 50b",
        "51:",
        "mov r13d, [r11 + r15 + 8]",
        // ts: r10 start, rax length
        "movzx eax, word ptr [rdi + {lenf} + rax * 2]",
        "cmp eax, {no_form}",
        "je 20f",
        "add r10, rsi",
        "mov rbx, [r10]",
        "mov r14, rbx",
        "shr r14, 43",
        "and r14d, 32",
        "xor rbx, [rdi + {cache} + 16 + r14]",
        "and rbx, [rdi + {forms} + {f_keymask} + rax]",
        "jnz 20f",
        "add r14, [rdi + {forms} + {f_slot} + rax]",
        "mov rcx, [rdi + {forms} + {f_off} + rax]",
        "vmovdqu xmm0, [r10 + rcx]",
        "vpxor xmm0, xmm0, [rdi + {forms} + {f_expect} + rax]",
        "vpmaxub xmm1, xmm0, [rdi + {forms} + {f_limit} + rax]",
        "vpcmpeqb xmm1, xmm1, [rdi + {forms} + {f_limit} + rax]",
        "vpshufb xmm0, xmm0, [rdi + {forms} + {f_shuf} + rax]",
        "vpmaddubsw xmm0, xmm0, xmm10",
        "vpsubw xmm2, xmm0, xmm9",
        "vpminuw xmm3, xmm2, [rdi + {cache} + r14]",
        "vpcmpeqw xmm3, xmm3, xmm2",
        "vpand xmm1, xmm1, xmm3",
        "vpmovmskb ecx, xmm1",
        "cmp ecx, 0xFFFF",
        "jne 20f",
        "vpmaddwd xmm0, xmm0, [rdi + {forms} + {f_fold4} + rax]",
        "vpackusdw xmm0, xmm0, xmm0",
        "vpmaddwd xmm0, xmm0, [rdi + {forms} + {f_fold2} + rax]",
        "vmovq rcx, xmm0",
        "mov ebx, ecx",
        "shr rcx, 32",
        "imul rbx, [rdi + {forms} + {f_scale} + rax]",
        "add rbx, rcx",
        "add rbx, [rdi + {cache} + 24 + r14]",
        "cmp eax, {f2_at}",
        "je 60f",
        "61:",
        "sub rbx, [rdi + {cws}]",
        "cmp rbx, [rdi + {cwlen}]",
        "jae 40f",
        "shl rbx, 32",
        "or rbx, r13",
        "cmp rbp, [rdi + {astop}]",
        "je 30f",
        "mov [rbp], r9",
        "mov [rbp + 8], r8",
        "mov [rbp + 16], rbx",
        "add rbp, 24",
        "40:",
        "lea rsi, [rsi + rdx + 33]",
        "cmp rsi, [rdi + {stop}]",
        "jb 12b",
        "xor eax, eax",
        "jmp 99f",
        // F2: the offset at bytes 19..25 of the field
        "60:",
        "mov r14, [r10 + 17]",
        "shr r14, 24",
        "mov rax, 0x30303A3030",
        "xor r14, rax",
        "mov rax, 0x7676767676767676",
        "add rax, r14",
        "or rax, r14",
        "test rax, [rdi + {c80}]",
        "jnz 20f",
        "test r14d, 0xFF0000",
        "jnz 20f",
        "imul rax, r14, 10",
        "shr r14, 8",
        "add rax, r14",
        "movzx r14d, al",
        "shr eax, 24",
        "movzx eax, al",
        "cmp r14d, 24",
        "jae 20f",
        "cmp eax, 60",
        "jae 20f",
        "imul r14d, r14d, 60",
        "add r14d, eax",
        "imul r14, r14, 60000",
        "movzx eax, byte ptr [r10 + 19]",
        "mov r15, r14",
        "neg r15",
        "cmp eax, 0x2D",
        "cmove r14, r15",
        "sub eax, 0x2B",
        "test eax, -3",
        "jnz 20f",
        "sub rbx, r14",
        "jmp 61b",
        // block full: the click to Rust, the line done
        "30:",
        "mov [rdi + {o_ekey}], r8",
        "mov [rdi + {o_ukey}], r9",
        "mov [rdi + {o_w3}], rbx",
        "lea rsi, [rsi + rdx + 33]",
        "mov eax, 2",
        "jmp 99f",
        "20:",
        "mov eax, 1",
        "99:",
        "mov [rdi + {at}], rbp",
        "vzeroupper",
        "pop rbp",
        "pop rbx",
        at = const offset_of!(Ctx, at),
        astop = const offset_of!(Ctx, astop),
        v_sub1 = const offset_of!(Ctx, v_sub1),
        v_madd = const offset_of!(Ctx, v_madd),
        v_ten = const offset_of!(Ctx, v_ten),
        v_comma = const offset_of!(Ctx, v_comma),
        v_nl = const offset_of!(Ctx, v_nl),
        v_quote = const offset_of!(Ctx, v_quote),
        stop = const offset_of!(Ctx, stop),
        m7f = const offset_of!(Ctx, m7f),
        km = const offset_of!(Ctx, km),
        c80 = const offset_of!(Ctx, c80),
        camp_tab = const offset_of!(Ctx, camp_tab),
        camp_mask = const offset_of!(Ctx, camp_mask),
        camp_shift = const offset_of!(Ctx, camp_shift),
        camp_mul = const offset_of!(Ctx, camp_mul),
        lenf = const offset_of!(Ctx, lenf),
        no_form = const NO_FORM,
        cache = const offset_of!(Ctx, cache),
        forms = const offset_of!(Ctx, forms),
        f_keymask = const offset_of!(Form, keymask),
        f_slot = const offset_of!(Form, slot),
        f_off = const offset_of!(Form, off),
        f_expect = const offset_of!(Form, expect),
        f_limit = const offset_of!(Form, limit),
        f_shuf = const offset_of!(Form, shuf),
        f_fold4 = const offset_of!(Form, fold4),
        f_fold2 = const offset_of!(Form, fold2),
        f_scale = const offset_of!(Form, scale),
        f2_at = const F2_AT,
        cws = const offset_of!(Ctx, cws),
        cwlen = const offset_of!(Ctx, cwlen),
        o_ekey = const offset_of!(Ctx, ekey),
        o_ukey = const offset_of!(Ctx, ukey),
        o_w3 = const offset_of!(Ctx, w3),
        in("rdi") c as *mut Ctx,
        inout("rsi") *p,
        out("eax") code,
        out("rcx") _, out("rdx") _, out("r8") _, out("r9") _, out("r10") _, out("r11") _,
        out("r12") _, out("r13") _, out("r14") _, out("r15") _,
        out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _, out("ymm4") _, out("ymm5") _,
        out("ymm6") _, out("ymm7") _, out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
        out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
    );
    code
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2", target_feature = "popcnt", target_feature = "movbe")))]
unsafe fn ckernel(c: &mut Ctx, p: &mut *const u8) -> u32 {
    ckernel_twin(c, p)
}

/// `ckernel` in portable Rust, step for step.
unsafe fn ckernel_twin(c: &mut Ctx, pp: &mut *const u8) -> u32 {
    let mut p = *pp;
    while p < c.stop {
        let code = cline_twin(c, p);
        if code == SLOW {
            *pp = p;
            return SLOW;
        }
        p = c.next;
        if code == FULL {
            *pp = p;
            return FULL;
        }
    }
    *pp = p;
    DONE
}

/// One click line: DONE when appended or dropped (`c.next` is the next
/// line), FULL when the block is full (the click in `ekey`, `ukey`, `w3`).
unsafe fn cline_twin(c: &mut Ctx, p: *const u8) -> u32 {
    let m = masks(p);
    let mut bad = u64::from(!m.clean);
    let tzr = u64::from((m.nm as u32).trailing_zeros());
    let nl = tzr + 32;
    bad |= tzr & 32;
    let cl = bzhi(m.c, nl);
    bad |= u64::from(cl.count_ones()) ^ 3;
    let d = cl | (1u64 << (nl & 63));
    let q = bzhi(m.q, nl);
    let (mut s, mut e) = window(d, q, d.wrapping_add(d).wrapping_add(1), &mut bad);
    if bad != 0 {
        return SLOW;
    }
    let mut f = [(0u64, 0u64); 4];
    for slot in &mut f {
        *slot = (tz(s), tz(e));
        s = blsr(s);
        e = blsr(e);
    }
    let (id_len, user_len) = (f[0].1 - f[0].0, f[1].1 - f[1].0);
    if id_len.wrapping_sub(1) > 8 || user_len.wrapping_sub(1) > 8 {
        return SLOW;
    }
    let pack = |at: u64, len: u64| -> u64 {
        let v = pext(load64(p.add(at as usize + 1)).swap_bytes(), c.m7f) | (u64::from(*p.add(at as usize)) << 56);
        v & c.km[len as usize]
    };
    let ckey = pack(f[0].0, id_len);
    let ukey = pack(f[1].0, user_len);
    c.next = p.add(nl as usize + 1);
    // Campaign: an ID of 1 to 7 bytes, one probe of the table.
    let cp = p.add(f[2].0 as usize);
    let clen = f[2].1 - f[2].0;
    if clen.wrapping_sub(1) > 6 {
        return SLOW;
    }
    let key = bzhi(load64(cp), clen * 8) | (clen << 56);
    let mut i = key.wrapping_mul(c.camp_mul) >> c.camp_shift;
    let camp = loop {
        let slot = *c.camp_tab.add(i as usize);
        if slot[0] == key {
            break slot[1] as u32;
        }
        if slot[0] == 0 {
            return DONE;
        }
        i = (i + 1) & c.camp_mask;
    };
    // ts, one lane.
    let (ts_at, tlen) = (f[3].0, f[3].1 - f[3].0);
    let form_at = c.lenf[tlen as usize];
    if form_at == NO_FORM {
        return SLOW;
    }
    let form_at = usize::from(form_at);
    let fm = &*((c.forms.as_ptr() as *const u8).add(form_at) as *const Form);
    let tp = p.add(ts_at as usize);
    let a = load64(tp);
    let sa = (a >> 43) & 32;
    let key = u64::from_le_bytes(c.cache[16 + sa as usize..24 + sa as usize].try_into().unwrap());
    if (a ^ key) & fm.keymask != 0 {
        return SLOW;
    }
    let Some((mut ts, _)) = ts_twin(c, form_at, tp, std::ptr::null(), sa + fm.slot, 0) else {
        return SLOW;
    };
    if form_at == F2_AT {
        let x = (load64(tp.add(17)) >> 24) ^ 0x30_303A_3030;
        if (x.wrapping_add(0x7676_7676_7676_7676) | x) & c.c80 != 0 || x & 0xFF_0000 != 0 {
            return SLOW;
        }
        let s = x.wrapping_mul(10).wrapping_add(x >> 8);
        let (hh, mm) = (s & 0xFF, (s >> 24) & 0xFF);
        if hh >= 24 || mm >= 60 {
            return SLOW;
        }
        let mut minutes = ((hh * 60 + mm) * 60_000) as i64;
        let sign = *tp.add(19);
        if sign == b'-' {
            minutes = -minutes;
        }
        if sign != b'+' && sign != b'-' {
            return SLOW;
        }
        ts -= minutes;
    }
    let rel = ts.wrapping_sub(c.cws) as u64;
    if rel >= c.cwlen {
        return DONE;
    }
    let tc = (rel << 32) | u64::from(camp);
    if c.at == c.astop {
        c.ekey = ckey;
        c.ukey = ukey;
        c.w3 = tc;
        return FULL;
    }
    c.at.write(Click { key: ukey, ckey, tc });
    c.at = c.at.add(1);
    DONE
}

/// A click line the kernel left, alone through the old loop.
#[cold]
#[inline(never)]
unsafe fn slow_line(this: *mut ParseState, m: Mapped, pos: usize, scan: &mut Scan) -> usize {
    (*this).clicks::<0, Fused>(Fused { m, pos, end: pos + 1, scan })
}

/// The free room of the last click block, as `clicks` takes it.
unsafe fn room(clicks: &mut [Vec<Click>]) -> (*mut Click, *mut Click) {
    match clicks.last_mut() {
        Some(v) => (v.as_mut_ptr().add(v.len()), v.as_mut_ptr().add(CLICK_BLOCK)),
        None => (std::ptr::null_mut(), std::ptr::null_mut()),
    }
}

/// The last block's length from the append position.
unsafe fn close(clicks: &mut [Vec<Click>], at: *mut Click) {
    if let Some(v) = clicks.last_mut() {
        v.set_len(at.offset_from(v.as_ptr()) as usize);
    }
}

impl<'a> ParseState<'a> {
    /// `clicks::<0, _>` over `Fused { m, pos, end, scan }`: the same clicks
    /// in the same order, and the same return value.
    #[inline(never)]
    pub(in crate::engine) unsafe fn clicks_hot(&mut self, m: Mapped, pos: usize, end: usize, scan: &mut Scan) -> usize {
        self.clicks_with(m, pos, end, scan, ckernel)
    }

    #[inline(always)]
    unsafe fn clicks_with(
        &mut self,
        m: Mapped,
        pos: usize,
        end: usize,
        scan: &mut Scan,
        run: unsafe fn(&mut Ctx, &mut *const u8) -> u32,
    ) -> usize {
        let this: *mut ParseState<'a> = self;
        let fast = (*this).fast;
        let hot = (*this).entries.hot();
        let base = m.ptr;
        let mut c = Ctx::new(fast, &hot, (*this).shift1, base.add(end));
        c.load_cache(&(*this).cache);
        let clicks: *mut Vec<Vec<Click>> = &mut (*this).clicks;
        (c.at, c.astop) = room(&mut *clicks);
        let mut p = base.add(pos);
        loop {
            match run(&mut c, &mut p) {
                DONE => {
                    close(&mut *clicks, c.at);
                    return p.offset_from(base) as usize;
                }
                FULL => {
                    if let Some(v) = (*clicks).last_mut() {
                        v.set_len(CLICK_BLOCK);
                    }
                    let mut v = Vec::with_capacity(CLICK_BLOCK);
                    c.at = v.as_mut_ptr();
                    c.astop = c.at.add(CLICK_BLOCK);
                    (*clicks).push(v);
                    c.at.write(Click { key: c.ukey, ckey: c.ekey, tc: c.w3 });
                    c.at = c.at.add(1);
                }
                _ => {
                    close(&mut *clicks, c.at);
                    let pos = p.offset_from(base) as usize;
                    let next = slow_line(this, m, pos, scan);
                    if next == pos {
                        return pos;
                    }
                    (c.at, c.astop) = room(&mut *clicks);
                    p = base.add(next);
                    c.load_cache(&(*this).cache);
                }
            }
        }
    }

    #[cfg(test)]
    unsafe fn clicks_twin(&mut self, m: Mapped, pos: usize, end: usize, scan: &mut Scan) -> usize {
        self.clicks_with(m, pos, end, scan, ckernel_twin)
    }
}

#[cfg(test)]
#[path = "hot_clicks_tests.rs"]
mod tests;
