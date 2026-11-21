//! The event loop in assembly (`ENGINE_ASM=1`, off by default).
//!
//! The same job as `ParseState::events::<0, _>` over `Fused`, one line at a
//! time. The common line (seven fields, quotes only around whole fields,
//! IDs of 1 to 9 bytes in 1..=127, `ts` of 13, 20 or 25 bytes and
//! `ingest_ts` of 20 bytes whose months are in the date cache, a non-empty
//! product) runs through one `asm!` loop, `kernel`, that keeps every value
//! in a register. Any other line leaves the loop and goes, alone, to the
//! old loop (`slow_line`), so it is decided exactly as before; the date
//! cache is only filled there, so the kernel reads a copy of it, renewed
//! after each such line. Purchases (3%) leave the loop with their keys,
//! ingest and `ts_rel` and get their price and category here in Rust. A
//! push that fills a staging buffer leaves the loop too; `push_hot` does it.
//!
//! On targets without AVX2 and BMI2 the kernel is `kernel_twin`, the same
//! steps in portable Rust. Tests compare both against the old loop.
//!
//! Per line, with the field splitting from the masks (`D` delimiters, `Q`
//! quotes, `st` field starts of one window, `O = Q & st`, `K = (Q << 1) & D`
//! the delimiters right after a quote): the line is refused unless
//! `(!D + O) & D == K` (each opened field is closed: the carry from an
//! opening quote runs to the delimiter that ends its field) and
//! `2 (Q & !O) == K` (every quote opens or closes a field, none does both).
//! Field starts are `st + O`, ends `D - (K >> 1)`.
//!
//! SAFETY (reads): the kernel reads 96 bytes from a line start (masks), 9
//! bytes from an ID start, 16 bytes from `tp + 4` or `tp`, 16 from `ip + 4`,
//! 8 from `tp + 17` and 4 from the type start. An accepted line has its line
//! break within the first 96 bytes, so every field starts before byte 96
//! and every read ends before byte 96 + 21 = 117 from a line start that is
//! inside the file. A mapping has at least one readable page after the
//! file's end (`sys::map_file`), and the tests pad with 4096 zero bytes.
//! The masks are read for every line start before `stop`, including the
//! last line of a mapping (tested by `noise_and_file_ends` with files cut at
//! every length). SAFETY (writes): a push writes one 32-byte entry at slot
//! `fill[part] < per` of the staging buffer of `part`, `part < 2^(64 -
//! shift1)` partitions, as `Scatter::push_hot` does; a push that would fill
//! the buffer is done by `push_hot` itself.

use super::*;
#[allow(unused_imports)]
use std::mem::offset_of;
use std::sync::OnceLock;

/// `ENGINE_ASM=1` selects this loop for event lines in the default shape.
pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENGINE_ASM").is_ok_and(|v| v == "1"))
}

const DONE: u32 = 0;
const SLOW: u32 = 1;
const FULL: u32 = 2;
const PURCH: u32 = 3;

/// The constants of one timestamp form, as `ts.rs` has them: lane 0 is the
/// `ts` form, lane 1 always F1 (`ingest_ts`).
#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct Form {
    expect: [[u8; 16]; 2],
    limit: [[u8; 16]; 2],
    shuf: [[i8; 16]; 2],
    fold4: [[i16; 8]; 2],
    fold2: [[i16; 8]; 2],
    scale: u64,
    off: u64,
    keymask: u64,
    /// Added to the cache slot offset: 64 moves it to the slots without a
    /// calendar.
    slot: u64,
}

const fn f1_lane(last: u8) -> ([u8; 16], [u8; 16], [i8; 16], [i16; 8], [i16; 8]) {
    (
        [0, 0, 0, 0, b'0', b'0', b'T', b'0', b'0', b':', b'0', b'0', b':', b'0', b'0', b'Z'],
        [255, 255, 255, 255, 9, 9, 0, 9, 9, 0, 9, 9, 0, 9, 9, last],
        [4, 5, 7, 8, 10, 11, 13, 14, -1, -1, -1, -1, -1, -1, -1, -1],
        [24, 1, 60, 1, 0, 0, 0, 0],
        [3600, 1, 0, 0, 0, 0, 0, 0],
    )
}

const fn form(index: usize) -> Form {
    let i = f1_lane(0);
    let t = match index {
        0 => (
            [b'0'; 16],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 255, 255, 255],
            [-1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [100, 1, 100, 1, 100, 1, 100, 1],
            [10000, 1, 10000, 1, 0, 0, 0, 0],
        ),
        1 => f1_lane(0),
        _ => f1_lane(255),
    };
    Form {
        expect: [t.0, i.0],
        limit: [t.1, i.1],
        shuf: [t.2, i.2],
        fold4: [t.3, i.3],
        fold2: [t.4, i.4],
        scale: if index == 0 { 100_000_000 } else { 1000 },
        off: if index == 0 { 0 } else { 4 },
        keymask: if index == 0 { 0 } else { u64::MAX },
        slot: if index == 0 { 64 } else { 0 },
    }
}

const FORM_BYTES: usize = std::mem::size_of::<Form>();
const NO_FORM: u16 = u16::MAX;
/// Byte offset in `Ctx::forms` of the F2 form.
const F2_AT: usize = 2 * FORM_BYTES;

/// Everything the kernel reads, at fixed offsets from one register, and
/// what it hands back when it leaves the loop.
#[repr(C, align(32))]
struct Ctx {
    v_comma: [u8; 32],
    v_nl: [u8; 32],
    v_quote: [u8; 32],
    v_ten: [u8; 32],
    v_sub1: [u16; 16],
    v_madd: [u16; 16],
    forms: [Form; 3],
    /// A copy of the `DateCache` (`#[repr(C)]`: four 32-byte slots, each
    /// `lim` at 0, `key` at 16, `base_ms` at 24; a month's slot is at byte
    /// `(a >> 43) & 32`, the slots without a calendar 64 further on).
    cache: [u8; 128],
    /// `ts` length (below 64) to the offset of its form in `forms`.
    lenf: [u16; 64],
    /// Key bits of an ID of each length, as `ids::pack_known`.
    km: [u64; 16],
    m7f: u64,
    c80: u64,
    emul: u64,
    ws: i64,
    wlen: u64,
    fill: *mut u8,
    stage: *mut u8,
    sbytes: u64,
    per: u64,
    stop: *const u8,
    shift1: u64,
    parts: *mut u8,
    // Out, for FULL and PURCH.
    ekey: u64,
    ukey: u64,
    ingest: i64,
    /// FULL: the entry's `w3`. PURCH: `ts_rel`.
    w3: u64,
    part: u64,
    next: *const u8,
}

const _: () = assert!(std::mem::size_of::<DateCache>() == 128);
const _: () = assert!(FORM_BYTES == 192);
const _: () = assert!(sys::PART_BYTES == 32);

impl Ctx {
    fn new(fast: &Fast, hot: &sys::Hot, shift1: u32, stop: *const u8) -> Box<Ctx> {
        let mut lenf = [NO_FORM; 64];
        lenf[13] = 0;
        lenf[20] = FORM_BYTES as u16;
        lenf[25] = F2_AT as u16;
        let mut km = [0u64; 16];
        for (len, m) in km.iter_mut().enumerate().take(10).skip(1) {
            *m = ((1u64 << (7 * len)) - 1) << (63 - 7 * len);
        }
        let mut sub1 = [0u16; 16];
        sub1[0] = 1;
        sub1[8] = 1;
        Box::new(Ctx {
            v_comma: [b','; 32],
            v_nl: [b'\n'; 32],
            v_quote: [b'"'; 32],
            v_ten: [10; 32],
            v_sub1: sub1,
            v_madd: [0x010A; 16],
            forms: [form(0), form(1), form(2)],
            cache: [0; 128],
            lenf,
            km,
            m7f: 0x7F7F_7F7F_7F7F_7F7F,
            c80: 0x8080_8080_8080_8080,
            emul: 0xD6E8_FEB8_6659_FD93,
            ws: fast.ws,
            wlen: fast.wlen,
            fill: hot.fill,
            stage: hot.stage,
            sbytes: hot.stage_bytes as u64,
            per: hot.per as u64,
            stop,
            shift1: u64::from(shift1),
            parts: hot.parts,
            ekey: 0,
            ukey: 0,
            ingest: 0,
            w3: 0,
            part: 0,
            next: std::ptr::null(),
        })
    }

    fn load_cache(&mut self, cache: &DateCache) {
        unsafe { std::ptr::copy_nonoverlapping(cache as *const DateCache as *const u8, self.cache.as_mut_ptr(), 128) };
    }
}

/// Runs lines from `*p` until `stop` or a line it leaves the loop for (see
/// the top of this file); returns why.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "bmi2", target_feature = "popcnt", target_feature = "movbe"))]
#[inline(never)]
unsafe fn kernel(c: &mut Ctx, p: &mut *const u8) -> u32 {
    let code: u32;
    core::arch::asm!(
        "push rbx",
        "push rbp",
        "mov rbp, [rdi + {shift1}]",
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
        "prefetcht0 [rsi + 1088]",
        "vmovdqu ymm0, [rsi]",
        "vmovdqu ymm1, [rsi + 32]",
        "vmovdqu ymm2, [rsi + 64]",
        // clean: bytes 0..32 in 11..=127, bytes 32..64 in 1..=127
        "vpcmpgtb ymm3, ymm0, ymm11",
        "vpcmpgtb ymm4, ymm1, ymm12",
        "vpand ymm3, ymm3, ymm4",
        "vpcmpeqb ymm4, ymm0, ymm13",
        "vpcmpeqb ymm5, ymm1, ymm13",
        "vpcmpeqb ymm6, ymm2, ymm13",
        "vpcmpeqb ymm0, ymm0, ymm15",
        "vpcmpeqb ymm7, ymm1, ymm15",
        "vpcmpeqb ymm8, ymm2, ymm15",
        "vpcmpeqb ymm1, ymm1, ymm14",
        "vpcmpeqb ymm2, ymm2, ymm14",
        "vpmovmskb r8d, ymm3",
        "not r8d",
        "vpmovmskb eax, ymm4",
        "vpmovmskb edx, ymm5",
        "shl rdx, 32",
        "or rax, rdx",
        "vpmovmskb r9d, ymm0",
        "vpmovmskb edx, ymm7",
        "shl rdx, 32",
        "or r9, rdx",
        "vpmovmskb r10d, ymm6",
        "vpmovmskb r11d, ymm8",
        "vpmovmskb edx, ymm1",
        "vpmovmskb ecx, ymm2",
        "shl rcx, 32",
        "or rdx, rcx",
        // rdx: line break at 32 + rdx (64: none)
        "tzcnt rdx, rdx",
        "blsr rcx, rax",
        "blsr rcx, rcx",
        "tzcnt rcx, rcx",
        // rcx = a2, r12 = lb
        "lea r12, [rdx + 32]",
        "sub r12, rcx",
        "mov r13, rcx",
        "or r13, r12",
        "or r13, rdx",
        "and r13, -64",
        "or r8, r13",
        "lea r13, [rcx + 1]",
        "bzhi r13, rax, r13",
        "bzhi r14, r9, rcx",
        "shrd rax, r10, cl",
        "shrd r9, r11, cl",
        "bzhi rax, rax, r12",
        "popcnt r10, rax",
        "xor r10, 4",
        "or r8, r10",
        "bts rax, r12",
        "bzhi r9, r9, r12",
        // window 1: d r13, q r14
        "lea r10, [r13 + r13 + 1]",
        "mov r11, r10",
        "and r11, r14",
        "lea r15, [r14 + r14]",
        "and r15, r13",
        "lea rbx, [r11 - 1]",
        "sub rbx, r13",
        "and rbx, r13",
        "xor rbx, r15",
        "or r8, rbx",
        "andn rbx, r11, r14",
        "add rbx, rbx",
        "xor rbx, r15",
        "or r8, rbx",
        "add r10, r11",
        "shr r15, 1",
        "sub r13, r15",
        // window 2: d rax, q r9
        "lea r11, [rax + rax]",
        "mov r14, r11",
        "and r14, r9",
        "lea r15, [r9 + r9]",
        "and r15, rax",
        "lea rbx, [r14 - 1]",
        "sub rbx, rax",
        "and rbx, rax",
        "xor rbx, r15",
        "or r8, rbx",
        "andn rbx, r14, r9",
        "add rbx, rbx",
        "xor rbx, r15",
        "or r8, rbx",
        "add r11, r14",
        "shr r15, 1",
        "blsr rax, rax",
        "sub rax, r15",
        "test r8, r8",
        "jnz 20f",
        // fields 0 and 1
        "tzcnt r12, r10",
        "tzcnt r14, r13",
        "blsr r10, r10",
        "blsr r13, r13",
        "tzcnt r15, r10",
        "tzcnt rbx, r13",
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
        "movzx r9d, byte ptr [rsi + r12]",
        "shl r9, 56",
        "or r8, r9",
        "and r8, [rdi + {km} + r14 * 8]",
        "movbe r9, [rsi + r15 + 1]",
        "pext r9, r9, [rdi + {m7f}]",
        "movzx r12d, byte ptr [rsi + r15]",
        "shl r12, 56",
        "or r9, r12",
        "and r9, [rdi + {km} + rbx * 8]",
        // field 2
        "blsr r10, r10",
        "blsr r13, r13",
        "tzcnt r10, r10",
        "tzcnt r13, r13",
        "sub r13, r10",
        "add rcx, rsi",
        // fields 3, 4, 5
        "tzcnt r12, r11",
        "tzcnt r14, rax",
        "sub r14, r12",
        "cmp r14, 20",
        "jne 20f",
        "blsr r11, r11",
        "blsr rax, rax",
        "tzcnt r15, r11",
        "tzcnt rbx, rax",
        "sub rbx, r15",
        "blsr r11, r11",
        "blsr rax, rax",
        "tzcnt r11, r11",
        "tzcnt rax, rax",
        "cmp r11, rax",
        "je 20f",
        "mov eax, [rcx + r15]",
        "mov r11d, eax",
        "xor r11d, {view}",
        "xor eax, {cart}",
        "imul r11, rax",
        "sub rbx, 4",
        "or r11, rbx",
        "neg r11",
        "sbb r11, r11",
        // timestamps: r10 tp, r12 ip, r13 form
        "movzx r13d, word ptr [rdi + {lenf} + r13 * 2]",
        "cmp r13d, {no_form}",
        "je 20f",
        "add r10, rsi",
        "add r12, rcx",
        "mov rax, [r10]",
        "mov r14, rax",
        "shr r14, 43",
        "and r14d, 32",
        "xor rax, [rdi + {cache} + 16 + r14]",
        "and rax, [rdi + {forms} + {f_keymask} + r13]",
        "mov rbx, [r12]",
        "mov r15, rbx",
        "shr r15, 43",
        "and r15d, 32",
        "xor rbx, [rdi + {cache} + 16 + r15]",
        "or rax, rbx",
        "jnz 20f",
        "add r14, [rdi + {forms} + {f_slot} + r13]",
        "mov rax, [rdi + {forms} + {f_off} + r13]",
        "vmovdqu xmm0, [r10 + rax]",
        "vinserti128 ymm0, ymm0, [r12 + 4], 1",
        "vpxor ymm0, ymm0, [rdi + {forms} + {f_expect} + r13]",
        "vpmaxub ymm1, ymm0, [rdi + {forms} + {f_limit} + r13]",
        "vpcmpeqb ymm1, ymm1, [rdi + {forms} + {f_limit} + r13]",
        "vpshufb ymm0, ymm0, [rdi + {forms} + {f_shuf} + r13]",
        "vpmaddubsw ymm0, ymm0, ymm10",
        "vpsubw ymm2, ymm0, ymm9",
        "vmovdqa xmm3, [rdi + {cache} + r14]",
        "vinserti128 ymm3, ymm3, [rdi + {cache} + r15], 1",
        "vpminuw ymm3, ymm3, ymm2",
        "vpcmpeqw ymm3, ymm3, ymm2",
        "vpand ymm1, ymm1, ymm3",
        "vpmovmskb eax, ymm1",
        "inc eax",
        "jnz 20f",
        "vpmaddwd ymm0, ymm0, [rdi + {forms} + {f_fold4} + r13]",
        "vpackusdw ymm0, ymm0, ymm0",
        "vpmaddwd ymm0, ymm0, [rdi + {forms} + {f_fold2} + r13]",
        "vmovq rax, xmm0",
        "vextracti128 xmm0, ymm0, 1",
        "vmovq rbx, xmm0",
        "imul rbx, rbx, 1000",
        "add rbx, [rdi + {cache} + 24 + r15]",
        "mov r12d, eax",
        "shr rax, 32",
        "imul r12, [rdi + {forms} + {f_scale} + r13]",
        "add r12, rax",
        "add r12, [rdi + {cache} + 24 + r14]",
        "cmp r13d, {f2_at}",
        "je 60f",
        "61:",
        // rules: rbx ingest, r12 ts
        "mov rax, r12",
        "sub rax, [rdi + {ws}]",
        "cmp rax, [rdi + {wlen}]",
        "jae 50f",
        "lea r14, [r12 + {late}]",
        "cmp rbx, r14",
        "jg 50f",
        "test r11, r11",
        "jnz 40f",
        "shl rax, 32",
        "mov r11, rax",
        "70:",
        // push: r8 ekey, r9 ukey, rbx ingest, r11 w3
        "mov rax, r8",
        "imul rax, [rdi + {emul}]",
        "shrx rax, rax, rbp",
        "mov r14, [rdi + {fill}]",
        "movzx r15d, byte ptr [r14 + rax]",
        "lea r10, [r15 + 1]",
        "cmp r10, [rdi + {per}]",
        "je 30f",
        "mov [r14 + rax], r10b",
        "imul rax, [rdi + {sbytes}]",
        "add rax, [rdi + {stage}]",
        "shl r15, 5",
        "mov [rax + r15], r8",
        "mov [rax + r15 + 8], r9",
        "mov [rax + r15 + 16], rbx",
        "mov [rax + r15 + 24], r11",
        "13:",
        "lea rsi, [rsi + rdx + 33]",
        "cmp rsi, [rdi + {stop}]",
        "jb 12b",
        "xor eax, eax",
        "jmp 99f",
        // not sound: DEAD
        "50:",
        "mov r11, -1",
        "jmp 70b",
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
        "sub r12, r14",
        "jmp 61b",
        // staging buffer full: write the entry, move the buffer to its
        // chunk as `Scatter::write_out` does, unless the chunk is full
        "30:",
        "mov r10, rax",
        "shl r10, 5",
        "add r10, [rdi + {parts}]",
        "mov r12, [r10]",
        "cmp r12, [r10 + 8]",
        "je 31f",
        "mov byte ptr [r14 + rax], 0",
        "imul rax, [rdi + {sbytes}]",
        "add rax, [rdi + {stage}]",
        "shl r15, 5",
        "mov [rax + r15], r8",
        "mov [rax + r15 + 8], r9",
        "mov [rax + r15 + 16], rbx",
        "mov [rax + r15 + 24], r11",
        "mov r13, [rdi + {sbytes}]",
        "add r13, rax",
        "32:",
        "vmovdqa ymm0, [rax]",
        "vmovdqa ymm1, [rax + 32]",
        "vmovntdq [r12], ymm0",
        "vmovntdq [r12 + 32], ymm1",
        "add rax, 64",
        "add r12, 64",
        "cmp rax, r13",
        "jb 32b",
        "mov [r10], r12",
        "jmp 13b",
        "31:",
        "mov [rdi + {o_ekey}], r8",
        "mov [rdi + {o_ukey}], r9",
        "mov [rdi + {o_ingest}], rbx",
        "mov [rdi + {o_w3}], r11",
        "mov [rdi + {o_part}], rax",
        "lea rsi, [rsi + rdx + 33]",
        "mov eax, 2",
        "jmp 99f",
        // purchase or other type, sound
        "40:",
        "mov [rdi + {o_ekey}], r8",
        "mov [rdi + {o_ukey}], r9",
        "mov [rdi + {o_ingest}], rbx",
        "mov [rdi + {o_w3}], rax",
        "lea rax, [rsi + rdx + 33]",
        "mov [rdi + {o_next}], rax",
        "mov eax, 3",
        "jmp 99f",
        "20:",
        "mov eax, 1",
        "99:",
        "vzeroupper",
        "pop rbp",
        "pop rbx",
        shift1 = const offset_of!(Ctx, shift1),
        v_sub1 = const offset_of!(Ctx, v_sub1),
        v_madd = const offset_of!(Ctx, v_madd),
        v_ten = const offset_of!(Ctx, v_ten),
        v_comma = const offset_of!(Ctx, v_comma),
        v_nl = const offset_of!(Ctx, v_nl),
        v_quote = const offset_of!(Ctx, v_quote),
        stop = const offset_of!(Ctx, stop),
        m7f = const offset_of!(Ctx, m7f),
        km = const offset_of!(Ctx, km),
        view = const VIEW,
        cart = const CART,
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
        ws = const offset_of!(Ctx, ws),
        wlen = const offset_of!(Ctx, wlen),
        late = const LATE_MS,
        emul = const offset_of!(Ctx, emul),
        fill = const offset_of!(Ctx, fill),
        per = const offset_of!(Ctx, per),
        sbytes = const offset_of!(Ctx, sbytes),
        stage = const offset_of!(Ctx, stage),
        c80 = const offset_of!(Ctx, c80),
        parts = const offset_of!(Ctx, parts),
        o_ekey = const offset_of!(Ctx, ekey),
        o_ukey = const offset_of!(Ctx, ukey),
        o_ingest = const offset_of!(Ctx, ingest),
        o_w3 = const offset_of!(Ctx, w3),
        o_part = const offset_of!(Ctx, part),
        o_next = const offset_of!(Ctx, next),
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
unsafe fn kernel(c: &mut Ctx, p: &mut *const u8) -> u32 {
    kernel_twin(c, p)
}

// ------------------------------------------------------------ the twin

fn tz(x: u64) -> u64 {
    u64::from(x.trailing_zeros())
}
fn blsr(x: u64) -> u64 {
    x & x.wrapping_sub(1)
}
/// The instruction: index from the low byte of `n`, `x` itself from 64 up.
fn bzhi(x: u64, n: u64) -> u64 {
    let n = n & 0xFF;
    if n >= 64 {
        x
    } else {
        x & ((1u64 << n) - 1)
    }
}
/// `shrd lo, hi, n` (count mod 64).
fn shrd(lo: u64, hi: u64, n: u64) -> u64 {
    let n = n & 63;
    if n == 0 {
        lo
    } else {
        (lo >> n) | (hi << (64 - n))
    }
}
fn pext(x: u64, m: u64) -> u64 {
    let (mut m, mut out, mut k) = (m, 0u64, 0);
    while m != 0 {
        out |= ((x >> m.trailing_zeros()) & 1) << k;
        k += 1;
        m &= m - 1;
    }
    out
}

/// The masks of the 96 bytes at `p`: commas and quotes of bytes 0..64 and
/// 64..96, line breaks of bytes 32..96, and whether bytes 0..32 are in
/// 11..=127 and bytes 32..64 in 1..=127.
#[derive(Clone, Copy)]
struct Masks {
    c: u64,
    q: u64,
    c2: u64,
    q2: u64,
    nm: u64,
    clean: bool,
}

#[inline(always)]
unsafe fn masks(p: *const u8) -> Masks {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::*;
        let a = _mm256_loadu_si256(p as *const __m256i);
        let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let x = _mm256_loadu_si256(p.add(64) as *const __m256i);
        let m = |v: __m256i, k: u8| u64::from(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(k as i8))) as u32);
        let ok = _mm256_and_si256(_mm256_cmpgt_epi8(a, _mm256_set1_epi8(10)), _mm256_cmpgt_epi8(b, _mm256_setzero_si256()));
        Masks {
            c: m(a, b',') | (m(b, b',') << 32),
            q: m(a, b'"') | (m(b, b'"') << 32),
            c2: m(x, b','),
            q2: m(x, b'"'),
            nm: m(b, b'\n') | (m(x, b'\n') << 32),
            clean: _mm256_movemask_epi8(ok) == -1,
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        masks_scalar(p)
    }
}

#[allow(dead_code)]
unsafe fn masks_scalar(p: *const u8) -> Masks {
    let mut m = Masks { c: 0, q: 0, c2: 0, q2: 0, nm: 0, clean: true };
    for i in 0..96 {
        let b = *p.add(i);
        if i < 32 && !(11..=127).contains(&b) || (32..64).contains(&i) && !(1..=127).contains(&b) {
            m.clean = false;
        }
        let (c, q) = if i < 64 { (&mut m.c, &mut m.q) } else { (&mut m.c2, &mut m.q2) };
        if b == b',' {
            *c |= 1 << (i % 64);
        }
        if b == b'"' {
            *q |= 1 << (i % 64);
        }
        if i >= 32 && b == b'\n' {
            m.nm |= 1 << (i - 32);
        }
    }
    m
}

/// One line split as the kernel splits it: `bad != 0` when refused;
/// otherwise the starts and ends of fields 0 to 2 from the line start and
/// of fields 3 to 6 from `a2`, as bit masks.
struct Split {
    bad: u64,
    tzr: u64,
    a2: u64,
    s1: u64,
    e1: u64,
    s2: u64,
    e2: u64,
}

/// One window's quote check: adds to `bad`, returns the start and end masks.
fn window(d: u64, q: u64, st: u64, bad: &mut u64) -> (u64, u64) {
    let o = st & q;
    let k = q.wrapping_add(q) & d;
    *bad |= (o.wrapping_sub(1).wrapping_sub(d) & d) ^ k;
    *bad |= ((!o & q).wrapping_add(!o & q)) ^ k;
    (st.wrapping_add(o), d.wrapping_sub(k >> 1))
}

#[inline(always)]
unsafe fn split(p: *const u8) -> Split {
    let m = masks(p);
    let mut bad = u64::from(!m.clean);
    let tzr = tz(m.nm);
    let a2 = tz(blsr(blsr(m.c)));
    let lb = (tzr + 32).wrapping_sub(a2);
    bad |= (a2 | lb | tzr) & !63;
    let d1 = bzhi(m.c, a2 + 1);
    let q1 = bzhi(m.q, a2);
    let cw = shrd(m.c, m.c2, a2);
    let qw = shrd(m.q, m.q2, a2);
    let cl = bzhi(cw, lb);
    bad |= u64::from(cl.count_ones()) ^ 4;
    let d2 = cl | (1u64 << (lb & 63));
    let q2 = bzhi(qw, lb);
    let (s1, e1) = window(d1, q1, d1.wrapping_add(d1).wrapping_add(1), &mut bad);
    let (s2, e2) = window(d2, q2, d2.wrapping_add(d2), &mut bad);
    // The ends of fields 3 to 6: `blsr(d2) - k2`, without the third comma.
    let e2 = e2.wrapping_sub(d2 & d2.wrapping_neg());
    Split { bad, tzr, a2, s1, e1, s2, e2 }
}

/// The `[DD - 1, HH, MM, SS]`-style vector step of `parse_ts_ingest`, lane
/// by lane as the instructions do it. Returns `(ts, ingest)` or `None`.
unsafe fn ts_twin(c: &Ctx, form_at: usize, tp: *const u8, ip: *const u8, sa: u64, sb: u64) -> Option<(i64, i64)> {
    let f = &*((c.forms.as_ptr() as *const u8).add(form_at) as *const Form);
    let mut v = [0u8; 32];
    std::ptr::copy_nonoverlapping(tp.add(f.off as usize), v.as_mut_ptr(), 16);
    std::ptr::copy_nonoverlapping(ip.add(4), v.as_mut_ptr().add(16), 16);
    let mut ok = true;
    let mut digits = [0u8; 32];
    for lane in 0..2 {
        let mut x = [0u8; 16];
        for i in 0..16 {
            x[i] = v[lane * 16 + i] ^ f.expect[lane][i];
            ok &= x[i].max(f.limit[lane][i]) == f.limit[lane][i];
        }
        for i in 0..16 {
            let s = f.shuf[lane][i];
            digits[lane * 16 + i] = if s < 0 { 0 } else { x[(s & 15) as usize] };
        }
    }
    let lim = |slot: u64, i: usize| -> u16 {
        let at = slot as usize + 2 * i;
        u16::from_le_bytes([c.cache[at], c.cache[at + 1]])
    };
    let mut pairs = [0i16; 16];
    for i in 0..16 {
        // maddubs: unsigned digit times signed weight (10, 1), saturated.
        let s = i32::from(digits[2 * i]) * 10 + i32::from(digits[2 * i + 1]);
        pairs[i] = s.clamp(-32768, 32767) as i16;
        let shifted = (pairs[i] as u16).wrapping_sub(if i % 8 == 0 { 1 } else { 0 });
        let l = lim(if i < 8 { sa } else { sb }, i % 8);
        ok &= shifted.min(l) == shifted;
    }
    if !ok {
        return None;
    }
    let mut fours = [0i32; 8];
    for i in 0..8 {
        let lane = i / 4;
        let w = &f.fold4[lane];
        let j = i % 4;
        fours[i] = i32::from(pairs[2 * i]) * i32::from(w[2 * j]) + i32::from(pairs[2 * i + 1]) * i32::from(w[2 * j + 1]);
    }
    // packus_epi32(fours, fours) per lane: 4 from a, 4 from b (the same).
    let pk = |x: i32| x.clamp(0, 65535) as u16 as i16;
    let mut packed = [0i16; 16];
    for lane in 0..2 {
        for j in 0..4 {
            packed[lane * 8 + j] = pk(fours[lane * 4 + j]);
            packed[lane * 8 + 4 + j] = pk(fours[lane * 4 + j]);
        }
    }
    let mut halves = [0i32; 8];
    for i in 0..8 {
        let lane = i / 4;
        let w = &f.fold2[lane];
        let j = i % 4;
        halves[i] = i32::from(packed[2 * i]).wrapping_mul(i32::from(w[2 * j])).wrapping_add(i32::from(packed[2 * i + 1]).wrapping_mul(i32::from(w[2 * j + 1])));
    }
    let base = |slot: u64| i64::from_le_bytes(c.cache[24 + slot as usize..32 + slot as usize].try_into().unwrap());
    let seconds = (halves[4] as u32 as u64) | ((halves[5] as u32 as u64) << 32);
    let ingest = (seconds.wrapping_mul(1000) as i64).wrapping_add(base(sb));
    let lo = halves[0] as u32 as u64;
    let hi = halves[1] as u32 as u64;
    let ts = (lo.wrapping_mul(f.scale).wrapping_add(hi) as i64).wrapping_add(base(sa));
    Some((ts, ingest))
}

/// `kernel` in portable Rust, step for step.
unsafe fn kernel_twin(c: &mut Ctx, pp: &mut *const u8) -> u32 {
    let mut p = *pp;
    while p < c.stop {
        let code = line_twin(c, p);
        if code != DONE {
            if code == FULL {
                p = c.next;
            }
            *pp = p;
            return code;
        }
        p = c.next;
    }
    *pp = p;
    DONE
}

/// One line: DONE when pushed (`c.next` is the next line), else why not.
unsafe fn line_twin(c: &mut Ctx, p: *const u8) -> u32 {
    let sp = split(p);
    if sp.bad != 0 {
        return SLOW;
    }
    let (mut s1m, mut e1m) = (sp.s1, sp.e1);
    let (s0, e0) = (tz(s1m), tz(e1m));
    s1m = blsr(s1m);
    e1m = blsr(e1m);
    let (s1, e1) = (tz(s1m), tz(e1m));
    let (id_len, user_len) = (e0 - s0, e1 - s1);
    if id_len.wrapping_sub(1) > 8 || user_len.wrapping_sub(1) > 8 {
        return SLOW;
    }
    let pack = |at: u64, len: u64| -> u64 {
        let v = pext(load64(p.add(at as usize + 1)).swap_bytes(), c.m7f) | (u64::from(*p.add(at as usize)) << 56);
        v & c.km[len as usize]
    };
    let ekey = pack(s0, id_len);
    let ukey = pack(s1, user_len);
    s1m = blsr(s1m);
    e1m = blsr(e1m);
    let s2 = tz(s1m);
    let tlen = tz(e1m) - s2;
    let q = p.add(sp.a2 as usize);
    let (mut s2m, mut e2m) = (sp.s2, sp.e2);
    let s3 = tz(s2m);
    if tz(e2m) - s3 != 20 {
        return SLOW;
    }
    s2m = blsr(s2m);
    e2m = blsr(e2m);
    let s4 = tz(s2m);
    let type_len = tz(e2m) - s4;
    s2m = blsr(s2m);
    e2m = blsr(e2m);
    if tz(s2m) == tz(e2m) {
        return SLOW;
    }
    let w = load32(q.add(s4 as usize));
    let plain = ((u64::from(w ^ VIEW) * u64::from(w ^ CART)) | type_len.wrapping_sub(4)) == 0;
    let form_at = c.lenf[tlen as usize];
    if form_at == NO_FORM {
        return SLOW;
    }
    let form_at = usize::from(form_at);
    let f = &*((c.forms.as_ptr() as *const u8).add(form_at) as *const Form);
    let (tp, ip) = (p.add(s2 as usize), q.add(s3 as usize));
    let key = |at: u64| u64::from_le_bytes(c.cache[16 + at as usize..24 + at as usize].try_into().unwrap());
    let a = load64(tp);
    let sa = (a >> 43) & 32;
    let b = load64(ip);
    let sb = (b >> 43) & 32;
    if ((a ^ key(sa)) & f.keymask) | (b ^ key(sb)) != 0 {
        return SLOW;
    }
    let Some((mut ts, ingest)) = ts_twin(c, form_at, tp, ip, sa + f.slot, sb) else {
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
    let ts_rel = ts.wrapping_sub(c.ws) as u64;
    let next = p.add(sp.tzr as usize + 33);
    let w3 = if ts_rel >= c.wlen || ingest > ts.wrapping_add(LATE_MS) {
        DEAD
    } else if !plain {
        c.ekey = ekey;
        c.ukey = ukey;
        c.ingest = ingest;
        c.w3 = ts_rel;
        c.next = next;
        return PURCH;
    } else {
        ts_rel << 32
    };
    let part = ekey.wrapping_mul(c.emul) >> c.shift1;
    let fill = c.fill.add(part as usize);
    let n = u64::from(*fill);
    c.next = next;
    if n + 1 == c.per {
        let part_rec = c.parts.add(part as usize * sys::PART_BYTES) as *mut *mut u8;
        if *part_rec != *part_rec.add(1) {
            *fill = 0;
            let buf = c.stage.add((part * c.sbytes) as usize);
            (buf.add(n as usize * 32) as *mut Entry).write(Entry { ekey, ukey, ingest, w3 });
            std::ptr::copy_nonoverlapping(buf, *part_rec, c.sbytes as usize);
            *part_rec = (*part_rec).add(c.sbytes as usize);
            return DONE;
        }
        c.ekey = ekey;
        c.ukey = ukey;
        c.ingest = ingest;
        c.w3 = w3;
        c.part = part;
        return FULL;
    }
    *fill = (n + 1) as u8;
    let at = c.stage.add((part * c.sbytes + n * 32) as usize) as *mut Entry;
    at.write(Entry { ekey, ukey, ingest, w3 });
    DONE
}

/// The `w3` of a sound line whose type is not `view` or `cart` of four
/// bytes: a purchase with a good price, or `DEAD`, as in `events`.
#[inline(never)]
unsafe fn purchase(fast: &Fast, p: *const u8, ts_rel: u64) -> u64 {
    let sp = split(p);
    let q = p.add(sp.a2 as usize);
    let (mut s, mut e) = (sp.s2, sp.e2);
    let mut f = [(0u64, 0u64); 4];
    for slot in &mut f {
        *slot = (tz(s), tz(e));
        s = blsr(s);
        e = blsr(e);
    }
    let (s4, e4) = f[1];
    let tp = q.add(s4 as usize);
    if e4 - s4 != 8 || load64(tp) != PURCHASE {
        return DEAD;
    }
    let (s6, e6) = f[3];
    let price = parse_price(q.add(s6 as usize), (e6 - s6) as usize);
    if price == 0 {
        return DEAD;
    }
    let (s5, e5) = f[2];
    let cat = fast.category(q.add(s5 as usize), (e5 - s5) as usize);
    (ts_rel << 32) | (price << 8) | u64::from(cat)
}

/// A line the kernel left, alone through the old loop. Returns where the
/// next line starts, or `pos` when only `long_line` can take this one.
#[cold]
#[inline(never)]
unsafe fn slow_line(this: *mut ParseState, m: Mapped, pos: usize, scan: &mut Scan) -> usize {
    (*this).events::<0, Fused>(Fused { m, pos, end: pos + 1, scan })
}

impl<'a> ParseState<'a> {
    /// `events::<0, _>` over `Fused { m, pos, end, scan }`: the same entries
    /// in the same order, and the same return value.
    #[inline(never)]
    pub(super) unsafe fn events_hot(&mut self, m: Mapped, pos: usize, end: usize, scan: &mut Scan) -> usize {
        self.events_with(m, pos, end, scan, kernel)
    }

    #[inline(always)]
    unsafe fn events_with(
        &mut self,
        m: Mapped,
        pos: usize,
        end: usize,
        scan: &mut Scan,
        run: unsafe fn(&mut Ctx, &mut *const u8) -> u32,
    ) -> usize {
        let this: *mut ParseState<'a> = self;
        let fast = (*this).fast;
        let shift1 = (*this).shift1;
        let entries: *mut Scatter<Entry> = &mut (*this).entries;
        let hot = (*entries).hot();
        let base = m.ptr;
        let mut c = Ctx::new(fast, &hot, shift1, base.add(end));
        c.load_cache(&(*this).cache);
        let mut p = base.add(pos);
        loop {
            match run(&mut c, &mut p) {
                DONE => return p.offset_from(base) as usize,
                FULL => {
                    let e = Entry { ekey: c.ekey, ukey: c.ukey, ingest: c.ingest, w3: c.w3 };
                    (*entries).push_hot(&hot, c.part as usize, e);
                }
                PURCH => {
                    let w3 = purchase(fast, p, c.w3);
                    let e = Entry { ekey: c.ekey, ukey: c.ukey, ingest: c.ingest, w3 };
                    (*entries).push_hot(&hot, event_part(c.ekey, shift1), e);
                    p = c.next;
                }
                _ => {
                    let pos = p.offset_from(base) as usize;
                    let next = slow_line(this, m, pos, scan);
                    if next == pos {
                        return pos;
                    }
                    p = base.add(next);
                    c.load_cache(&(*this).cache);
                }
            }
        }
    }

    #[cfg(test)]
    unsafe fn events_twin(&mut self, m: Mapped, pos: usize, end: usize, scan: &mut Scan) -> usize {
        self.events_with(m, pos, end, scan, kernel_twin)
    }
}

#[cfg(test)]
#[path = "hot_tests.rs"]
mod tests;
