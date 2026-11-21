//! OS glue: file mappings with readable padding, bump arenas over lazily
//! committed anonymous memory, chunked append-only lists, thread CPU clocks.

use std::ffi::CString;
use std::marker::PhantomData;
use std::path::Path;

use crate::common::die;

pub fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// A read-only file mapping followed by at least one zero page, so SIMD and
/// 8-byte loads may read past the end of the file.
#[derive(Clone, Copy)]
pub struct Mapped {
    pub ptr: *const u8,
    pub len: usize,
}
unsafe impl Send for Mapped {}
unsafe impl Sync for Mapped {}

static ZEROS: [u8; 256] = [0; 256];

pub fn map_file(path: &Path) -> Mapped {
    let c = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    unsafe {
        let fd = libc::open(c.as_ptr(), libc::O_RDONLY);
        if fd < 0 {
            die(&format!("{}: {}", path.display(), std::io::Error::last_os_error()));
        }
        let mut st: libc::stat = std::mem::zeroed();
        if libc::fstat(fd, &mut st) != 0 {
            die(&format!("{}: {}", path.display(), std::io::Error::last_os_error()));
        }
        let len = st.st_size as usize;
        if len == 0 {
            libc::close(fd);
            return Mapped { ptr: ZEROS.as_ptr().add(64), len: 0 };
        }
        let page = page_size();
        let rounded = (len + page - 1) & !(page - 1);
        // Reserve the file span plus one anonymous page, then map the file
        // over the front. The tail page stays anonymous zeros.
        let base = libc::mmap(
            std::ptr::null_mut(),
            rounded + page,
            libc::PROT_READ,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        );
        if base == libc::MAP_FAILED {
            die(&format!("mmap reserve: {}", std::io::Error::last_os_error()));
        }
        let got = libc::mmap(base, len, libc::PROT_READ, libc::MAP_PRIVATE | libc::MAP_FIXED, fd, 0);
        if got == libc::MAP_FAILED {
            die(&format!("{}: mmap: {}", path.display(), std::io::Error::last_os_error()));
        }
        libc::close(fd);
        Mapped { ptr: base as *const u8, len }
    }
}

/// Ask for transparent huge pages on the big anonymous arenas. On unless
/// `ENGINE_THP=0`.
#[cfg(target_os = "linux")]
fn huge_pages() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENGINE_THP").map_or(true, |v| v != "0"))
}

pub fn unmap_file(m: Mapped) {
    if m.len > 0 {
        let page = page_size();
        let rounded = (m.len + page - 1) & !(page - 1);
        unsafe { libc::munmap(m.ptr as *mut libc::c_void, rounded + page) };
    }
}

/// Unmap the whole pages of `[start, end)` of a parsed file chunk. A
/// neighbouring chunk may still be reading the first and last partial
/// lines, so the range is shrunk by a page at both ends; if a later read
/// does touch a dropped page it simply faults back in from the page cache.
pub fn drop_pages(m: Mapped, start: usize, end: usize, on: bool) {
    if !on {
        return;
    }
    let page = page_size();
    let from = (start + page - 1) & !(page - 1);
    let to = end & !(page - 1);
    if to > from + 2 * page {
        unsafe { libc::madvise(m.ptr.add(from + page) as *mut libc::c_void, to - from - 2 * page, libc::MADV_DONTNEED) };
    }
}

use std::sync::atomic::AtomicUsize;

/// Every anonymous region handed out, so they can be returned in parallel
/// before exit. A fixed table: it must not live in the memory it lists.
static REGIONS: [(AtomicUsize, AtomicUsize); 1024] = [const { (AtomicUsize::new(0), AtomicUsize::new(0)) }; 1024];
static REGIONS_USED: AtomicUsize = AtomicUsize::new(0);

/// Unmap all anonymous regions with `threads` threads. Call only when
/// nothing in them is needed any more, right before exit.
///
/// Why: when the process exits, the kernel frees its memory on one CPU,
/// inside the exit, and under a memory cgroup (the scorer's container)
/// that costs about 0.13 ms per MB of 4 KiB pages. The container is not
/// done until that finishes. Unmapping from all threads does the same work
/// in parallel while the clock is still ours.
pub fn release_all(threads: usize) {
    use std::sync::atomic::Ordering;
    const PIECE: usize = 32 << 20;
    let used = REGIONS_USED.load(Ordering::Acquire).min(REGIONS.len());
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    for region in &REGIONS[..used] {
        let (base, bytes) = (region.0.load(Ordering::Acquire), region.1.load(Ordering::Acquire));
        let mut at = 0;
        while at < bytes {
            let len = PIECE.min(bytes - at);
            pieces.push((base + at, len));
            at += len;
        }
    }
    // Small enough to live in the system allocator, not in a region.
    debug_assert!(pieces.capacity() * 16 < (64 << 10));
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(base, len)) = pieces.get(i) else {
                    break;
                };
                unsafe { libc::munmap(base as *mut libc::c_void, len) };
            });
        }
    });
}

pub fn anon(bytes: usize) -> *mut u8 {
    unsafe {
        #[cfg(target_os = "linux")]
        let flags = libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE;
        #[cfg(not(target_os = "linux"))]
        let flags = libc::MAP_PRIVATE | libc::MAP_ANON;
        let p = libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0);
        if p == libc::MAP_FAILED {
            die(&format!("mmap {bytes} bytes: {}", std::io::Error::last_os_error()));
        }
        #[cfg(target_os = "linux")]
        if huge_pages() {
            libc::madvise(p, bytes, libc::MADV_HUGEPAGE);
        }
        let slot = REGIONS_USED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if slot < REGIONS.len() {
            REGIONS[slot].0.store(p as usize, std::sync::atomic::Ordering::Relaxed);
            REGIONS[slot].1.store(bytes, std::sync::atomic::Ordering::Release);
        }
        p as *mut u8
    }
}

/// Bump allocator over big anonymous slabs. Nothing is ever freed: the
/// process exits without teardown.
pub struct Arena {
    cur: *mut u8,
    end: *mut u8,
}

const SLAB: usize = 1 << 28;

impl Arena {
    pub fn new() -> Self {
        Arena { cur: std::ptr::null_mut(), end: std::ptr::null_mut() }
    }
    #[inline]
    pub fn alloc(&mut self, bytes: usize) -> *mut u8 {
        let bytes = (bytes + 63) & !63;
        if (self.end as usize) - (self.cur as usize) < bytes {
            let slab = SLAB.max(bytes);
            self.cur = anon(slab);
            self.end = unsafe { self.cur.add(slab) };
        }
        let p = self.cur;
        self.cur = unsafe { self.cur.add(bytes) };
        p
    }
}

#[repr(C, align(64))]
pub struct ChunkHdr {
    next: *mut ChunkHdr,
    len: u32,
}

const HDR: usize = 64;
const STAGE_BUDGET: usize = 256 << 10;

/// One finished list of chunks: the output of a scatter for one partition.
pub struct List<T> {
    head: *mut ChunkHdr,
    pub chunks: u32,
    _t: PhantomData<T>,
}
impl<T> Clone for List<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for List<T> {}
unsafe impl<T> Send for List<T> {}
unsafe impl<T> Sync for List<T> {}

impl<T> List<T> {
    /// Visit every chunk as a mutable slice.
    #[inline]
    pub unsafe fn for_each_chunk(&self, mut f: impl FnMut(&mut [T])) {
        let mut at = self.head;
        while !at.is_null() {
            let len = (*at).len as usize;
            let data = (at as *mut u8).add(HDR) as *mut T;
            f(std::slice::from_raw_parts_mut(data, len));
            at = (*at).next;
        }
    }
    /// Append this list's chunks to `out` as `(data, len)`, oldest first:
    /// the order the records were pushed in.
    pub unsafe fn chunks_oldest_first(&self, out: &mut Vec<(*const T, usize)>) {
        let start = out.len();
        let mut at = self.head;
        while !at.is_null() {
            out.push(((at as *const u8).add(HDR) as *const T, (*at).len as usize));
            at = (*at).next;
        }
        out[start..].reverse();
    }
    /// Hand every chunk's memory to `f`. The list must not be used again.
    pub unsafe fn release(&self, mut f: impl FnMut(*mut u8)) {
        let mut at = self.head;
        while !at.is_null() {
            let next = (*at).next;
            f(at as *mut u8);
            at = next;
        }
    }
    pub unsafe fn count(&self) -> usize {
        let mut n = 0;
        let mut at = self.head;
        while !at.is_null() {
            n += (*at).len as usize;
            at = (*at).next;
        }
        n
    }
}

#[derive(Clone, Copy)]
struct Part {
    /// Next line-aligned write position in the current chunk, and its end.
    dst: *mut u8,
    end: *mut u8,
    head: *mut ChunkHdr,
    chunks: u32,
}

/// Copy `lines` cache lines from an aligned staging buffer to an aligned
/// destination with non-temporal stores: no read-for-ownership of the
/// destination and no cache pollution.
#[inline(always)]
unsafe fn stream_lines(src: *const u8, dst: *mut u8, lines: usize) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::*;
        for i in 0..lines * 2 {
            let v = _mm256_load_si256(src.add(32 * i) as *const __m256i);
            _mm256_stream_si256(dst.add(32 * i) as *mut __m256i, v);
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    std::ptr::copy_nonoverlapping(src, dst, lines * 64);
}

/// Append-only scatter into `parts` partitions. Records first land in a
/// small per-partition staging buffer (a whole number of cache lines) and
/// move to the partition's current chunk one full buffer at a time, so the
/// scatter touches one staging line per record, not a chunk tail and a TLB
/// entry per partition. Chunks are cut from a thread-local arena.
impl Scatter<u64> {
    /// Push a run of words to one partition, in order.
    #[inline(always)]
    pub unsafe fn push_words(&mut self, p: usize, src: &[u64]) {
        let mut from = src.as_ptr();
        let mut len = src.len();
        let mut n = *self.fill.get_unchecked(p) as usize;
        let buf = self.stage.add(p * self.stage_bytes);
        while len > 0 {
            let take = (self.per - n).min(len);
            copy_words(from, (buf as *mut u64).add(n), take);
            n += take;
            from = from.add(take);
            len -= take;
            if n == self.per {
                n = 0;
                let mut part = self.parts.as_mut_ptr().add(p);
                if (*part).dst == (*part).end {
                    self.grow(p);
                    part = self.parts.as_mut_ptr().add(p);
                }
                stream_lines(buf, (*part).dst, self.stage_bytes / 64);
                (*part).dst = (*part).dst.add(self.stage_bytes);
            }
        }
        *self.fill.get_unchecked_mut(p) = n as u8;
    }
}

pub struct Scatter<T> {
    parts: Vec<Part>,
    /// Records waiting in each staging buffer.
    fill: Vec<u8>,
    stage: *mut u8,
    /// Bytes and records per staging buffer.
    stage_bytes: usize,
    per: usize,
    arena: Arena,
    /// Chunks handed back by a consumer, reused before the arena grows.
    free: Vec<*mut u8>,
    chunk_bytes: usize,
    /// Records per chunk; a multiple of `per`.
    cap: usize,
    _t: PhantomData<T>,
}
unsafe impl<T> Send for Scatter<T> {}

impl<T: Copy> Scatter<T> {
    pub fn new(parts: usize, chunk_bytes: usize) -> Self {
        let size = std::mem::size_of::<T>();
        let mut per = 1;
        while (per * size) % 64 != 0 {
            per += 1;
        }
        // A longer staging buffer flushes less often, which mostly saves
        // mispredicted flush branches; keep all buffers within ~256 KB so
        // they stay in L2 next to the parser's own data.
        let lines = (STAGE_BUDGET / (parts * per * size)).clamp(1, 4);
        per *= lines;
        assert!(per <= 255);
        let stage_bytes = per * size;
        let chunk_bytes = chunk_bytes.max(HDR + stage_bytes);
        let cap = (chunk_bytes - HDR) / stage_bytes * per;
        let mut arena = Arena::new();
        let stage = arena.alloc(parts * stage_bytes);
        let empty = Part { dst: std::ptr::null_mut(), end: std::ptr::null_mut(), head: std::ptr::null_mut(), chunks: 0 };
        Scatter {
            parts: vec![empty; parts],
            fill: vec![0; parts],
            stage,
            stage_bytes,
            per,
            arena,
            free: Vec::new(),
            chunk_bytes,
            cap,
            _t: PhantomData,
        }
    }

    /// Give a dead chunk of exactly `bytes` bytes to this scatter to reuse.
    #[inline]
    pub fn recycle(&mut self, chunk: *mut u8, bytes: usize) {
        if bytes == self.chunk_bytes {
            self.free.push(chunk);
        }
    }

    pub fn spare(&self) -> usize {
        self.free.len()
    }

    pub fn chunk_bytes(&self) -> usize {
        self.chunk_bytes
    }

    #[inline(always)]
    pub unsafe fn push(&mut self, p: usize, value: T) {
        let n = *self.fill.get_unchecked(p) as usize;
        let buf = self.stage.add(p * self.stage_bytes);
        (buf as *mut T).add(n).write(value);
        if n + 1 == self.per {
            *self.fill.get_unchecked_mut(p) = 0;
            let mut part = self.parts.as_mut_ptr().add(p);
            if (*part).dst == (*part).end {
                self.grow(p);
                part = self.parts.as_mut_ptr().add(p);
            }
            stream_lines(buf, (*part).dst, self.stage_bytes / 64);
            (*part).dst = (*part).dst.add(self.stage_bytes);
        } else {
            *self.fill.get_unchecked_mut(p) = (n + 1) as u8;
        }
    }

    #[cold]
    #[inline(never)]
    unsafe fn grow(&mut self, p: usize) {
        let size = std::mem::size_of::<T>();
        let part = &mut self.parts[p];
        if !part.head.is_null() {
            (*part.head).len = self.cap as u32;
        }
        let chunk = match self.free.pop() {
            Some(chunk) => chunk as *mut ChunkHdr,
            None => self.arena.alloc(self.chunk_bytes) as *mut ChunkHdr,
        };
        (*chunk).next = part.head;
        (*chunk).len = 0;
        part.head = chunk;
        part.chunks += 1;
        part.dst = (chunk as *mut u8).add(HDR);
        part.end = part.dst.add(self.cap * size);
    }

    pub fn finish(mut self) -> Vec<List<T>> {
        let size = std::mem::size_of::<T>();
        unsafe {
            for p in 0..self.parts.len() {
                let n = self.fill[p] as usize;
                if n > 0 {
                    if self.parts[p].dst == self.parts[p].end {
                        self.grow(p);
                    }
                    let part = &mut self.parts[p];
                    std::ptr::copy_nonoverlapping(self.stage.add(p * self.stage_bytes), part.dst, n * size);
                    part.dst = part.dst.add(n * size);
                }
            }
            #[cfg(target_arch = "x86_64")]
            core::arch::x86_64::_mm_sfence();
        }
        self.parts
            .iter()
            .map(|part| {
                if !part.head.is_null() {
                    let start = unsafe { (part.head as *mut u8).add(HDR) };
                    let used = (part.dst as usize - start as usize) / size;
                    unsafe { (*part.head).len = used as u32 };
                }
                List { head: part.head, chunks: part.chunks, _t: PhantomData }
            })
            .collect()
    }
}

/// Copy `n` 8-byte words. Inline, so that a short run pays no call into
/// `memcpy`.
#[inline(always)]
pub unsafe fn copy_words(mut src: *const u64, mut dst: *mut u64, mut n: usize) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::*;
        while n >= 4 {
            _mm256_storeu_si256(dst as *mut __m256i, _mm256_loadu_si256(src as *const __m256i));
            src = src.add(4);
            dst = dst.add(4);
            n -= 4;
        }
        if n & 2 != 0 {
            _mm_storeu_si128(dst as *mut __m128i, _mm_loadu_si128(src as *const __m128i));
            src = src.add(2);
            dst = dst.add(2);
        }
        if n & 1 != 0 {
            dst.write(src.read());
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    std::ptr::copy_nonoverlapping(src, dst, n);
}

#[inline(always)]
pub fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags, readonly));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = p;
}

/// Sub-phase CPU accounting, summed over threads. Off (one predictable
/// branch per call) unless `ENGINE_TIMING` is set.
pub mod prof {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    pub const NAMES: [&str; 17] = [
        "parse.scan",
        "parse.fields",
        "links.parse",
        "links.build",
        "resolve.dedup",
        "resolve.emit",
        "resolve.clicks",
        "person.group",
        "person.sessions",
        "person.flush",
        "raw.group",
        "raw.lookup",
        "raw.order",
        "raw.clicks",
        "raw.sessions",
        "raw.forward",
        "raw.recycle",
    ];
    pub static ON: AtomicBool = AtomicBool::new(false);
    static NANOS: [AtomicU64; 17] = [const { AtomicU64::new(0) }; 17];

    #[inline(always)]
    pub fn start() -> f64 {
        if ON.load(Ordering::Relaxed) {
            super::thread_cpu()
        } else {
            0.0
        }
    }

    /// Charge the time since `since` to `slot`; returns the new start.
    #[inline(always)]
    pub fn add(slot: usize, since: f64) -> f64 {
        if ON.load(Ordering::Relaxed) {
            let now = super::thread_cpu();
            NANOS[slot].fetch_add(((now - since) * 1e9) as u64, Ordering::Relaxed);
            now
        } else {
            0.0
        }
    }

    pub fn report() {
        for (name, nanos) in NAMES.iter().zip(&NANOS) {
            eprintln!("[engine]   {name:<16} cpu {:7.1} ms", nanos.load(Ordering::Relaxed) as f64 / 1e6);
        }
    }
}

/// CPU seconds used by the calling thread.
pub fn thread_cpu() -> f64 {
    unsafe {
        let mut ts: libc::timespec = std::mem::zeroed();
        libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts);
        ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
    }
}

pub fn threads() -> usize {
    let from_env = std::env::var("NCPU").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0);
    let n = from_env.unwrap_or_else(|| std::thread::available_parallelism().map_or(8, usize::from));
    n.clamp(1, 256)
}

pub fn exit_now(code: i32) -> ! {
    unsafe { libc::_exit(code) }
}
