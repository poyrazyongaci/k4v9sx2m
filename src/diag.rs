//! Diagnostic build only (cargo feature `diag`). Reports facts about the
//! machine the run lands on: static probes on stderr before the pipeline,
//! micro-benchmarks after it, and the most important facts again as marker
//! rows in user_daily (user ids `~diag00`..), so the dev comparison fails
//! on purpose and shows them. Every probe reports its own error and none
//! may panic: the release profile aborts on panic.

use std::ffi::{CStr, CString};
use std::hint::black_box;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Barrier, Mutex};
use std::time::Instant;

use crate::common::Lookups;
use crate::output::USER_SCHEMA_COMPACT;
use crate::parquet::{as_bytes, def_levels, plain_bytes, Page, PqFile};

static PHASES: Mutex<Vec<(String, f64, f64)>> = Mutex::new(Vec::new());
static FACTS: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
static LARGEST: Mutex<Option<(PathBuf, u64)>> = Mutex::new(None);

pub fn phase(name: &str, wall_ms: f64, cpu_ms: f64) {
    if let Ok(mut p) = PHASES.lock() {
        p.push((name.to_string(), wall_ms, cpu_ms));
    }
}

fn fact(key: &str, val: &str) {
    let val: String = val.trim().chars().map(|c| if c == ';' || c == '=' || c.is_control() { ',' } else { c }).collect();
    if let Ok(mut f) = FACTS.lock() {
        match f.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = val,
            None => f.push((key.to_string(), val)),
        }
    }
}

// ---- output

fn emit(line: &str) {
    let mut b = Vec::with_capacity(line.len() + 1);
    b.extend_from_slice(line.as_bytes());
    b.push(b'\n');
    let mut off = 0;
    while off < b.len() {
        let n = unsafe { libc::write(2, b[off..].as_ptr().cast(), b.len() - off) };
        if n < 0 {
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return;
        }
        if n == 0 {
            return;
        }
        off += n as usize;
    }
}

fn chunks(s: &str, max: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = s;
    while rest.len() > max {
        let mut cut = max;
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        out.push(&rest[..cut]);
        rest = &rest[cut..];
    }
    out.push(rest);
    out
}

/// One fact: the lines of `text` joined with " | ", split over numbered
/// lines of at most 440 bytes of content.
fn put(name: &str, text: &str) {
    let flat: String = text
        .trim_end()
        .lines()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join(" | ")
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let parts = chunks(&flat, 440);
    if parts.len() == 1 {
        emit(&format!("DIAG {name}: {}", parts[0]));
    } else {
        for (i, p) in parts.iter().enumerate() {
            emit(&format!("DIAG {name} [{}/{}]: {p}", i + 1, parts.len()));
        }
    }
}

fn read(path: &str) -> std::io::Result<String> {
    let f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.take(8 << 20).read_to_end(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read1(path: &str) -> String {
    match read(path) {
        Ok(s) => s.trim().to_string(),
        Err(e) => format!("ERR {e}"),
    }
}

fn show(name: &str, path: &str) -> Option<String> {
    match read(path) {
        Ok(s) => {
            put(name, if s.trim().is_empty() { "(empty)" } else { &s });
            Some(s)
        }
        Err(e) => {
            put(name, &format!("ERR {e}"));
            None
        }
    }
}

fn last_err() -> String {
    format!("ERR {}", std::io::Error::last_os_error())
}

fn list_dir(dir: &str) -> std::io::Result<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    Ok(names)
}

fn ranges(v: &[usize]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1] == v[j] + 1 {
            j += 1;
        }
        out.push(if i == j { format!("{}", v[i]) } else { format!("{}-{}", v[i], v[j]) });
        i = j + 1;
    }
    out.join(",")
}

fn bracketed(s: &str) -> String {
    match (s.find('['), s.find(']')) {
        (Some(a), Some(b)) if b > a => s[a + 1..b].to_string(),
        _ => s.trim().to_string(),
    }
}

fn kv_lines<'a>(text: &'a str, keys: &[&str]) -> Vec<&'a str> {
    text.lines().filter(|l| keys.iter().any(|k| l.split(':').next().map(str::trim) == Some(*k))).collect()
}

fn field(text: &str, key: &str) -> Option<String> {
    text.lines().find_map(|l| {
        let (a, b) = l.split_once(':').or_else(|| l.split_once(' '))?;
        (a.trim() == key).then(|| b.trim().to_string())
    })
}

// ---- clocks and counters

fn clock(id: libc::clockid_t) -> (i64, i64) {
    let mut ts: libc::timespec = unsafe { std::mem::zeroed() };
    unsafe { libc::clock_gettime(id, &mut ts) };
    (ts.tv_sec as i64, ts.tv_nsec as i64)
}

fn now_ns() -> u64 {
    let (s, n) = clock(libc::CLOCK_MONOTONIC);
    s as u64 * 1_000_000_000 + n as u64
}

fn rusage(who: i32) -> libc::rusage {
    let mut r: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(who, &mut r) };
    r
}

const RUSAGE_THREAD: i32 = 1;

fn minflt_thread() -> i64 {
    rusage(RUSAGE_THREAD).ru_minflt as i64
}

fn affinity() -> Result<Vec<usize>, String> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Err(last_err());
        }
        Ok((0..libc::CPU_SETSIZE as usize).filter(|&c| libc::CPU_ISSET(c, &set)).collect())
    }
}

fn paths() -> (String, String, String) {
    let args: Vec<String> = std::env::args().collect();
    let env = |n: &str, d: &str| std::env::var(n).unwrap_or_else(|_| d.to_string());
    let (data, out) = if args.len() >= 3 {
        (args[args.len() - 2].clone(), args[args.len() - 1].clone())
    } else {
        (env("DATA_DIR", "/data"), env("OUTPUT_DIR", "/output"))
    };
    (data, out, env("SCRATCH_DIR", "/scratch"))
}

// ---- static probes

pub fn before() {
    let t_entry = now_ns();
    let mono = clock(libc::CLOCK_MONOTONIC);
    let real = clock(libc::CLOCK_REALTIME);
    emit("DIAG begin: diagnostic build, sections A to I, then the pipeline, then benches");
    let (data, out, scratch) = paths();
    identity();
    cpu();
    memory_policy();
    cgroup_static();
    mounts(&data, &out, &scratch);
    input_state(&data);
    environment();
    limits();
    startup(mono, real);
    put("probes_ms", &format!("{:.3}", (now_ns() - t_entry) as f64 / 1e6));
    emit("DIAG pipeline: starting the normal engine with phase timers on");
}

fn identity() {
    unsafe {
        let mut u: libc::utsname = std::mem::zeroed();
        if libc::uname(&mut u) == 0 {
            let f = |p: &[libc::c_char]| CStr::from_ptr(p.as_ptr()).to_string_lossy().into_owned();
            let release = f(&u.release);
            put(
                "A.uname",
                &format!("sysname={} release={} version={} machine={}", f(&u.sysname), release, f(&u.version), f(&u.machine)),
            );
            fact("kernel", &release);
        } else {
            put("A.uname", &last_err());
        }
    }
    show("A.proc_version", "/proc/version");
    show("A.proc_cmdline", "/proc/cmdline");
}

fn cpu() {
    match read("/proc/cpuinfo") {
        Ok(s) => {
            let nproc = s.lines().filter(|l| l.split(':').next().map(str::trim) == Some("processor")).count();
            let first = s.split("\n\n").next().unwrap_or("");
            for k in ["model name", "cpu family", "model", "stepping", "microcode", "cpu MHz", "cache size", "bogomips"] {
                let v = field(first, k).unwrap_or_else(|| "absent".to_string());
                put(&format!("B.cpuinfo.{}", k.replace(' ', "_")), &v);
                if k == "model name" {
                    fact("cpu", &v);
                }
            }
            put("B.cpuinfo.flags", &field(first, "flags").unwrap_or_else(|| "absent".to_string()));
            put("B.cpuinfo.processors", &nproc.to_string());
            fact("nproc", &nproc.to_string());
        }
        Err(e) => put("B.cpuinfo", &format!("ERR {e}")),
    }
    for f in ["online", "possible", "present"] {
        show(&format!("B.cpu.{f}"), &format!("/sys/devices/system/cpu/{f}"));
    }
    let dir = "/sys/devices/system/cpu/cpu0/cache";
    match list_dir(dir) {
        Ok(names) => {
            for n in names.iter().filter(|n| n.starts_with("index")) {
                let get = |f: &str| read1(&format!("{dir}/{n}/{f}"));
                let (level, ty, size) = (get("level"), get("type"), get("size"));
                put(
                    &format!("B.cache.{n}"),
                    &format!(
                        "level={level} type={ty} size={size} shared_cpu_list={} ways={}",
                        get("shared_cpu_list"),
                        get("ways_of_associativity")
                    ),
                );
                if ty != "Instruction" && (level == "2" || level == "3") {
                    fact(&format!("L{level}"), &size);
                }
            }
        }
        Err(e) => put("B.cache", &format!("ERR {e}")),
    }
    match affinity() {
        Ok(aff) => {
            put("B.affinity", &format!("count={} cpus={}", aff.len(), ranges(&aff)));
            fact("aff", &format!("{}:{}", aff.len(), ranges(&aff)));
            let mut topo = Vec::new();
            let mut sibs: Vec<String> = Vec::new();
            for c in &aff {
                let base = format!("/sys/devices/system/cpu/cpu{c}/topology");
                let sib = read1(&format!("{base}/thread_siblings_list"));
                topo.push(format!("cpu{c}:core={},sib={}", read1(&format!("{base}/core_id")), sib));
                if !sibs.contains(&sib) {
                    sibs.push(sib);
                }
            }
            put("B.topology", &topo.join(" "));
            fact("sib", &sibs.join(" "));
        }
        Err(e) => put("B.affinity", &e),
    }
    match read("/proc/self/status") {
        Ok(s) => put(
            "B.status",
            &kv_lines(&s, &["Cpus_allowed_list", "Mems_allowed_list", "Threads", "voluntary_ctxt_switches"]).join("\n"),
        ),
        Err(e) => put("B.status", &format!("ERR {e}")),
    }
    let vdir = "/sys/devices/system/cpu/vulnerabilities";
    match list_dir(vdir) {
        Ok(names) => {
            for n in names {
                put(&format!("B.vuln.{n}"), &read1(&format!("{vdir}/{n}")));
            }
        }
        Err(e) => put("B.vuln", &format!("ERR {e}")),
    }
    show("B.cpufreq.governor", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    show("B.cpufreq.cur_freq", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
    show("B.node.online", "/sys/devices/system/node/online");
    match read("/sys/devices/system/node/node0/meminfo") {
        Ok(s) => put("B.node0.meminfo", &s.lines().take(4).collect::<Vec<_>>().join("\n")),
        Err(e) => put("B.node0.meminfo", &format!("ERR {e}")),
    }
}

fn memory_policy() {
    let t = "/sys/kernel/mm/transparent_hugepage";
    for (name, file, key) in [
        ("enabled", "enabled", "thp"),
        ("defrag", "defrag", "thpdefrag"),
        ("shmem_enabled", "shmem_enabled", "thpshmem"),
        ("hpage_pmd_size", "hpage_pmd_size", ""),
        ("khugepaged_defrag", "khugepaged/defrag", ""),
        ("use_zero_page", "use_zero_page", ""),
    ] {
        if let Some(s) = show(&format!("C.thp.{name}"), &format!("{t}/{file}")) {
            if !key.is_empty() {
                fact(key, &bracketed(&s));
            }
        }
    }
    show("C.meminfo", "/proc/meminfo");
    for f in [
        "overcommit_memory",
        "swappiness",
        "dirty_ratio",
        "dirty_background_ratio",
        "dirty_expire_centisecs",
        "dirty_writeback_centisecs",
        "max_map_count",
        "zone_reclaim_mode",
    ] {
        show(&format!("C.vm.{f}"), &format!("/proc/sys/vm/{f}"));
    }
    for f in ["perf_event_paranoid", "randomize_va_space", "numa_balancing"] {
        show(&format!("C.kernel.{f}"), &format!("/proc/sys/kernel/{f}"));
    }
}

fn cgroup_static() {
    show("D.proc_cgroup", "/proc/self/cgroup");
    for f in [
        "cgroup.controllers",
        "cpuset.cpus.effective",
        "cpuset.mems.effective",
        "cpu.max",
        "cpu.weight",
        "cpu.stat",
        "memory.max",
        "memory.high",
        "memory.swap.max",
        "memory.current",
        "memory.peak",
        "memory.stat",
        "pids.max",
        "io.max",
    ] {
        let got = show(&format!("D.cg.{f}"), &format!("/sys/fs/cgroup/{f}"));
        let key = match f {
            "cpu.max" => "cpumax",
            "cpuset.cpus.effective" => "cpuset",
            "memory.max" => "memmax",
            _ => "",
        };
        if let (Some(s), false) = (got, key.is_empty()) {
            fact(key, &s.trim().replace(' ', "/"));
        }
    }
}

fn fs_name(t: u64) -> &'static str {
    match t {
        0x0102_1994 => "tmpfs",
        0xEF53 => "ext4",
        0x794c_7630 => "overlayfs",
        0x5846_5342 => "xfs",
        0x9123_683E => "btrfs",
        0x6969 => "nfs",
        0x6573_5546 => "fuse",
        0x7371_7368 => "squashfs",
        0x2fc1_2fc1 => "zfs",
        0x0102_1997 => "9p",
        0x6a65_6a63 => "virtiofs",
        _ => "other",
    }
}

fn mounts(data: &str, out: &str, scratch: &str) {
    show("E.mountinfo", "/proc/self/mountinfo");
    let events = format!("{data}/events");
    let list = [("data", data), ("data_events", events.as_str()), ("output", out), ("scratch", scratch), ("tmp", "/tmp"), ("root", "/")];
    let mut devs: Vec<(&str, Option<u64>)> = Vec::new();
    for (tag, p) in list {
        let Ok(c) = CString::new(p) else {
            put(&format!("E.statfs.{tag}"), "ERR bad path");
            continue;
        };
        unsafe {
            let mut s: libc::statfs = std::mem::zeroed();
            if libc::statfs(c.as_ptr(), &mut s) == 0 {
                let t = (s.f_type as u64) & 0xffff_ffff;
                let bs = s.f_bsize as u64;
                put(
                    &format!("E.statfs.{tag}"),
                    &format!(
                        "path={p} type=0x{t:x} ({}) bsize={bs} total_mb={} free_mb={}",
                        fs_name(t),
                        s.f_blocks as u64 * bs >> 20,
                        s.f_bavail as u64 * bs >> 20
                    ),
                );
                if matches!(tag, "data" | "output" | "scratch") {
                    fact(&format!("fs_{tag}"), fs_name(t));
                }
                if tag == "data" {
                    fact("data_tmpfs", if t == 0x0102_1994 { "yes" } else { "no" });
                }
            } else {
                put(&format!("E.statfs.{tag}"), &format!("path={p} {}", last_err()));
            }
            let mut st: libc::stat = std::mem::zeroed();
            devs.push((tag, (libc::stat(c.as_ptr(), &mut st) == 0).then_some(st.st_dev as u64)));
        }
    }
    let dev = |t: &str| devs.iter().find(|d| d.0 == t).and_then(|d| d.1);
    let same = |a: &str, b: &str| match (dev(a), dev(b)) {
        (Some(x), Some(y)) => (x == y).to_string(),
        _ => "unknown".to_string(),
    };
    let all: Vec<String> = devs.iter().map(|(t, d)| format!("{t}={}", d.map_or("ERR".to_string(), |d| format!("0x{d:x}")))).collect();
    put(
        "E.st_dev",
        &format!(
            "{} output==scratch:{} output==data:{} scratch==data:{}",
            all.join(" "),
            same("output", "scratch"),
            same("output", "data"),
            same("scratch", "data")
        ),
    );
}

fn meminfo_pick() -> String {
    match read("/proc/meminfo") {
        Ok(s) => kv_lines(&s, &["FileHugePages", "ShmemHugePages", "Shmem", "Cached", "Mapped"])
            .iter()
            .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
            .collect::<Vec<_>>()
            .join(", "),
        Err(e) => format!("ERR {e}"),
    }
}

/// The smaps block of the mapping that contains `addr`.
fn smaps_entry(addr: usize) -> Result<String, String> {
    let s = read("/proc/self/smaps").map_err(|e| format!("ERR {e}"))?;
    let mut lines = s.lines();
    while let Some(l) = lines.next() {
        let Some(range) = l.split_whitespace().next() else { continue };
        let Some((a, b)) = range.split_once('-') else { continue };
        if range.ends_with(':') {
            continue;
        }
        let (Ok(a), Ok(b)) = (usize::from_str_radix(a, 16), usize::from_str_radix(b, 16)) else { continue };
        if a <= addr && addr < b {
            let body: Vec<&str> = lines.by_ref().take_while(|l| l.split_whitespace().next().is_some_and(|t| t.ends_with(':'))).collect();
            return Ok(body.join("\n"));
        }
    }
    Err("ERR mapping not found".to_string())
}

fn smaps_pick(body: &str, keys: &[&str]) -> String {
    kv_lines(body, keys).iter().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).collect::<Vec<_>>().join(", ")
}

fn input_state(data: &str) {
    let mut largest: Option<(PathBuf, u64)> = None;
    for sub in ["events", "clicks", "links"] {
        let dir = format!("{data}/{sub}");
        match std::fs::read_dir(&dir) {
            Ok(rd) => {
                let mut sizes: Vec<(PathBuf, u64)> = Vec::new();
                for e in rd.filter_map(|e| e.ok()) {
                    if let Ok(m) = e.metadata() {
                        if m.is_file() {
                            sizes.push((e.path(), m.len()));
                        }
                    }
                }
                let total: u64 = sizes.iter().map(|s| s.1).sum();
                let max = sizes.iter().map(|s| s.1).max().unwrap_or(0);
                let min = sizes.iter().map(|s| s.1).min().unwrap_or(0);
                put(&format!("F.files.{sub}"), &format!("count={} total_bytes={total} largest={max} smallest={min}", sizes.len()));
                if sub == "events" {
                    largest = sizes.iter().max_by_key(|s| s.1).cloned();
                }
            }
            Err(e) => put(&format!("F.files.{sub}"), &format!("ERR {e}")),
        }
    }
    let Some((path, len)) = largest else {
        put("F.largest", "ERR no events file");
        return;
    };
    if let Ok(mut l) = LARGEST.lock() {
        *l = Some((path.clone(), len));
    }
    put("F.largest", &format!("path={} bytes={len}", path.display()));
    put("F.meminfo_before", &meminfo_pick());
    if len == 0 {
        put("F.mincore", "ERR empty file");
        return;
    }
    let Ok(c) = CString::new(path.as_os_str().as_encoded_bytes()) else { return };
    unsafe {
        let fd = libc::open(c.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC);
        if fd < 0 {
            put("F.mincore", &last_err());
            return;
        }
        let len = len as usize;
        let page = 4096usize;
        let pages = len.div_ceil(page);
        let p = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_PRIVATE, fd, 0);
        if p == libc::MAP_FAILED {
            put("F.mincore", &format!("mmap {}", last_err()));
        } else {
            let mut vec = vec![0u8; pages];
            if libc::mincore(p, len, vec.as_mut_ptr().cast()) == 0 {
                let res = vec.iter().filter(|&&b| b & 1 != 0).count();
                let frac = res as f64 / pages as f64;
                put("F.mincore", &format!("resident_pages={res} of {pages} fraction={frac:.4}"));
                fact("mincore", &format!("{frac:.3}"));
            } else {
                put("F.mincore", &last_err());
            }
            libc::munmap(p, len);
        }
        let f0 = minflt_thread();
        let t0 = now_ns();
        let p = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_PRIVATE, fd, 0);
        if p == libc::MAP_FAILED {
            put("F.touch", &format!("mmap {}", last_err()));
        } else {
            let base = p as *const u8;
            let mut sum = 0u64;
            let mut off = 0;
            while off < len {
                sum = sum.wrapping_add(std::ptr::read_volatile(base.add(off)) as u64);
                off += page;
            }
            let t1 = now_ns();
            let f1 = minflt_thread();
            black_box(sum);
            put(
                "F.touch",
                &format!(
                    "pages={pages} wall_ns={} ns_per_page={:.1} minflt={}",
                    t1 - t0,
                    (t1 - t0) as f64 / pages as f64,
                    f1 - f0
                ),
            );
            match smaps_entry(p as usize) {
                Ok(body) => put(
                    "F.smaps",
                    &smaps_pick(
                        &body,
                        &["Rss", "FilePmdMapped", "ShmemPmdMapped", "AnonHugePages", "KernelPageSize", "MMUPageSize", "THPeligible"],
                    ),
                ),
                Err(e) => put("F.smaps", &e),
            }
            libc::munmap(p, len);
        }
        libc::close(fd);
    }
    put("F.meminfo_after", &meminfo_pick());
}

fn environment() {
    let wanted = ["DATA_DIR", "OUTPUT_DIR", "SCRATCH_DIR", "HOME", "NCPU", "MEM_GB", "PATH", "HOSTNAME"];
    let mut set = Vec::new();
    let mut others = Vec::new();
    for (k, v) in std::env::vars_os() {
        let k = k.to_string_lossy().into_owned();
        if wanted.contains(&k.as_str()) {
            set.push(format!("{k}={}", v.to_string_lossy()));
        } else {
            others.push(k);
        }
    }
    set.sort();
    others.sort();
    put("G.env", &set.join(" "));
    let names = if others.is_empty() { "(none)".to_string() } else { others.join(" ") };
    put("G.env_other_names", &names);
}

fn limits() {
    show("H.limits", "/proc/self/limits");
    let mut parts = Vec::new();
    for (name, res) in
        [("NOFILE", libc::RLIMIT_NOFILE), ("AS", libc::RLIMIT_AS), ("MEMLOCK", libc::RLIMIT_MEMLOCK), ("STACK", libc::RLIMIT_STACK)]
    {
        let mut r: libc::rlimit = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrlimit(res, &mut r) } == 0 {
            let f = |v: libc::rlim_t| if v == libc::RLIM_INFINITY { "inf".to_string() } else { v.to_string() };
            parts.push(format!("{name}={}/{}", f(r.rlim_cur), f(r.rlim_max)));
        } else {
            parts.push(format!("{name}={}", last_err()));
        }
    }
    put("H.getrlimit", &parts.join(" "));
}

fn startup(mono: (i64, i64), real: (i64, i64)) {
    put("I.clock_entry", &format!("monotonic={}.{:09} realtime={}.{:09}", mono.0, mono.1, real.0, real.1));
    let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    let start_ticks = read("/proc/self/stat").ok().and_then(|s| {
        let after = &s[s.rfind(')')? + 1..];
        after.split_whitespace().nth(19)?.parse::<u64>().ok()
    });
    let uptime = read("/proc/uptime").ok().and_then(|s| s.split_whitespace().next()?.parse::<f64>().ok());
    let delay = match (start_ticks, uptime, hz > 0) {
        (Some(st), Some(up), true) => format!("{:.0}", (up - st as f64 / hz as f64) * 1e3),
        _ => "unknown".to_string(),
    };
    put(
        "I.start",
        &format!(
            "starttime_ticks={} clk_tck={hz} uptime_s={} start_to_probe_ms~{delay} monotonic_s={}",
            start_ticks.map_or("ERR".to_string(), |v| v.to_string()),
            uptime.map_or("ERR".to_string(), |v| v.to_string()),
            mono.0
        ),
    );
    show("I.loadavg", "/proc/loadavg");
    match read("/proc/stat") {
        Ok(s) => put(
            "I.proc_stat",
            &s.lines().filter(|l| l.starts_with("cpu ") || l.starts_with("procs_")).collect::<Vec<_>>().join("\n"),
        ),
        Err(e) => put("I.proc_stat", &format!("ERR {e}")),
    }
    for f in ["cpu", "memory", "io"] {
        show(&format!("I.pressure.{f}"), &format!("/proc/pressure/{f}"));
    }
}

// ---- benchmarks

const MB: usize = 1 << 20;

struct Region {
    base: *mut libc::c_void,
    maplen: usize,
    ptr: *mut u8,
}

impl Region {
    /// Anonymous memory, 2 MiB aligned; `Some(true)` asks for huge pages,
    /// `Some(false)` forbids them.
    fn new(len: usize, huge: Option<bool>) -> Result<Region, String> {
        let maplen = len + 2 * MB;
        unsafe {
            let base = libc::mmap(
                std::ptr::null_mut(),
                maplen,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            if base == libc::MAP_FAILED {
                return Err(format!("mmap {}", last_err()));
            }
            let ptr = ((base as usize + 2 * MB - 1) & !(2 * MB - 1)) as *mut u8;
            match huge {
                Some(true) => libc::madvise(ptr.cast(), len, libc::MADV_HUGEPAGE),
                Some(false) => libc::madvise(ptr.cast(), len, libc::MADV_NOHUGEPAGE),
                None => 0,
            };
            Ok(Region { base, maplen, ptr })
        }
    }

    fn u64s(&self) -> *mut u64 {
        self.ptr.cast()
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.base, self.maplen) };
    }
}

fn xorshift(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

fn fill(p: *mut u64, n: usize, seed: u64) {
    let s = unsafe { std::slice::from_raw_parts_mut(p, n) };
    for (i, v) in s.iter_mut().enumerate() {
        *v = (i as u64 ^ seed) | 1;
    }
}

fn sum(p: *const u64, n: usize) -> u64 {
    let s = unsafe { std::slice::from_raw_parts(p, n) };
    s.iter().fold(0u64, |a, &x| a.wrapping_add(x))
}

fn bench(name: &str, r: Result<String, String>) {
    match r {
        Ok(s) => put(&format!("bench {name}"), &s),
        Err(e) => put(&format!("bench {name}"), &format!("ERR {e}")),
    }
}

/// From the start and end times each thread took itself: the earliest
/// start and the latest end, or `None` if any thread failed. Main's own
/// clock is not used, since main is a ninth thread on eight CPUs.
fn span(spans: &[Option<(u64, u64)>]) -> Option<(u64, u64)> {
    let all: Option<Vec<(u64, u64)>> = spans.iter().copied().collect();
    let all = all.filter(|v| !v.is_empty())?;
    Some((all.iter().map(|s| s.0).min()?, all.iter().map(|s| s.1).max()?))
}

fn b_seq() {
    let r = (|| {
        let len = 512 * MB;
        let reg = Region::new(len, Some(true))?;
        let n = len / 8;
        let t0 = now_ns();
        fill(reg.u64s(), n, 0x5555);
        let t1 = now_ns();
        let s = black_box(sum(reg.u64s(), n));
        let t2 = now_ns();
        let fill_ns = (t1 - t0) as f64 / len as f64;
        let gbs = len as f64 / (t2 - t1) as f64;
        fact("seq1", &format!("{gbs:.2}"));
        fact("fill_nspb", &format!("{fill_ns:.4}"));
        Ok(format!("1 thread 512MB hugepage: fill_first_touch_ns_per_byte={fill_ns:.4} sum_GBps={gbs:.3} (check {})", s & 0xff))
    })();
    bench("seq1", r);
    let threads = 8;
    let barrier = Barrier::new(threads + 1);
    let spans = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let barrier = &barrier;
                sc.spawn(move || {
                    let len = 256 * MB;
                    let reg = Region::new(len, Some(true));
                    if let Ok(r) = &reg {
                        fill(r.u64s(), len / 8, t as u64);
                    }
                    barrier.wait();
                    let t0 = now_ns();
                    let s = reg.as_ref().map(|r| black_box(sum(r.u64s(), len / 8)));
                    let t1 = now_ns();
                    barrier.wait();
                    s.ok().map(|_| (t0, t1))
                })
            })
            .collect();
        barrier.wait();
        barrier.wait();
        let spans: Vec<Option<(u64, u64)>> = hs.into_iter().map(|h| h.join().ok().flatten()).collect();
        span(&spans)
    });
    let r = if let Some((a, b)) = spans {
        let gbs = (8 * 256 * MB) as f64 / (b - a) as f64;
        fact("seq8", &format!("{gbs:.2}"));
        Ok(format!("8 threads x 256MB hugepage, private slices: aggregate_sum_GBps={gbs:.3} wall_ms={:.2}", (b - a) as f64 / 1e6))
    } else {
        Err("a thread could not map its slice".to_string())
    };
    bench("seq8", r);
}

fn b_faults() {
    for huge in [true, false] {
        let r = (|| {
            let len = 256 * MB;
            let reg = Region::new(len, Some(huge))?;
            let f0 = minflt_thread();
            let t0 = now_ns();
            let mut off = 0;
            while off < len {
                unsafe { std::ptr::write_volatile(reg.ptr.add(off), 1u8) };
                off += 4096;
            }
            let t1 = now_ns();
            let f1 = minflt_thread();
            let ms = (t1 - t0) as f64 / 1e6;
            let ahp = smaps_entry(reg.ptr as usize).map_or_else(|e| e, |b| smaps_pick(&b, &["AnonHugePages", "Rss"]));
            fact(if huge { "fault_huge_ms" } else { "fault_small_ms" }, &format!("{ms:.1}"));
            if huge {
                let kb = smaps_entry(reg.ptr as usize).ok().and_then(|b| field(&b, "AnonHugePages")).unwrap_or_else(|| "?".into());
                fact("ahp", &kb);
            }
            Ok(format!("first touch 256MB {}: ms={ms:.2} minflt={} smaps: {ahp}", if huge { "MADV_HUGEPAGE" } else { "MADV_NOHUGEPAGE" }, f1 - f0))
        })();
        bench(if huge { "fault_huge" } else { "fault_small" }, r);
    }
}

/// A single random cycle through the cache lines of `bytes`: slot 0 of
/// each line holds the u64 index of the next line's slot 0. Also returns
/// the visiting order, so chains can start evenly spaced on the cycle.
fn cycle(bytes: usize, seed: u64) -> Result<(Region, Region, usize), String> {
    let n = bytes / 64;
    let arr = Region::new(bytes, Some(true))?;
    let ord = Region::new(n * 4, Some(true))?;
    let order = unsafe { std::slice::from_raw_parts_mut(ord.ptr as *mut u32, n) };
    for (i, o) in order.iter_mut().enumerate() {
        *o = i as u32;
    }
    let mut x = seed | 1;
    for i in (1..n).rev() {
        x = xorshift(x);
        let j = (x % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let a = arr.u64s();
    for i in 0..n {
        let next = order[(i + 1) % n] as u64 * 8;
        unsafe { *a.add(order[i] as usize * 8) = next };
    }
    Ok((arr, ord, n))
}

fn start_of(ord: &Region, n: usize, k: usize, chains: usize) -> usize {
    unsafe { *(ord.ptr as *const u32).add(k * n / chains) as usize * 8 }
}

fn chase(a: *const u64, start: usize, hops: usize) -> f64 {
    let mut p = start;
    let t0 = now_ns();
    for _ in 0..hops {
        p = unsafe { *a.add(p) } as usize;
    }
    let t1 = now_ns();
    black_box(p);
    (t1 - t0) as f64 / hops as f64
}

fn chase_k<const K: usize>(a: *const u64, starts: [usize; K], accesses: usize) -> f64 {
    let mut ps = starts;
    let rounds = accesses / K;
    let t0 = now_ns();
    for _ in 0..rounds {
        for p in ps.iter_mut() {
            *p = unsafe { *a.add(*p) } as usize;
        }
    }
    let t1 = now_ns();
    black_box(ps);
    (t1 - t0) as f64 / (rounds * K) as f64
}

fn b_chase() {
    const HOPS: usize = 2_000_000;
    let r = (|| {
        let (arr, ord, n) = cycle(256 * MB, 0x9E37_79B9)?;
        let one = chase(arr.u64s(), start_of(&ord, n, 0, 1), HOPS);
        let mut s8 = [0usize; 8];
        for (k, s) in s8.iter_mut().enumerate() {
            *s = start_of(&ord, n, k, 8);
        }
        let mut s16 = [0usize; 16];
        for (k, s) in s16.iter_mut().enumerate() {
            *s = start_of(&ord, n, k, 16);
        }
        let c8 = chase_k(arr.u64s(), s8, HOPS);
        let c16 = chase_k(arr.u64s(), s16, HOPS);
        fact("dram1ns", &format!("{one:.1}"));
        fact("mlp8", &format!("{c8:.2}"));
        fact("mlp16", &format!("{c16:.2}"));
        Ok(format!(
            "256MB hugepage, one line per hop, {HOPS} accesses: chain1_ns_per_hop={one:.2} chains8_ns_per_access={c8:.3} chains16_ns_per_access={c16:.3}"
        ))
    })();
    bench("chase_dram", r);
    let threads = 8;
    let barrier = Barrier::new(threads);
    let res: Vec<Option<f64>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let barrier = &barrier;
                sc.spawn(move || {
                    let c = cycle(64 * MB, 0x1234_5678 + t as u64);
                    barrier.wait();
                    c.ok().map(|(arr, ord, n)| chase(arr.u64s(), start_of(&ord, n, 0, 1), HOPS))
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().ok().flatten()).collect()
    });
    let r = if res.iter().all(Option::is_some) {
        let v: Vec<f64> = res.into_iter().flatten().collect();
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let max = v.iter().cloned().fold(0.0, f64::max);
        fact("dram8ns", &format!("{mean:.1}"));
        Ok(format!("8 threads, private 64MB hugepage arrays, {HOPS} hops each: mean_ns_per_hop={mean:.2} max={max:.2}"))
    } else {
        Err("a thread failed to build its array".to_string())
    };
    bench("chase_dram8", r);
    for (bytes, name, key) in [(16 * MB, "chase_16MB", "l3ns"), (4 * MB, "chase_4MB", "l3ns4"), (512 << 10, "chase_512KB", "l2ns")] {
        let r = (|| {
            let (arr, ord, n) = cycle(bytes, 0xABCD)?;
            let a = arr.u64s();
            let s = start_of(&ord, n, 0, 1);
            chase(a, s, HOPS / 4);
            let ns = chase(a, s, HOPS);
            fact(key, &format!("{ns:.2}"));
            Ok(format!("{} KB array, {n} lines, {HOPS} hops after warm-up: ns_per_hop={ns:.3}", bytes >> 10))
        })();
        bench(name, r);
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn flush_nt(src: *const u8, dst: *mut u8, bytes: usize) {
    use std::arch::x86_64::{__m128i, _mm_load_si128, _mm_stream_si128};
    let mut o = 0;
    while o < bytes {
        _mm_stream_si128(dst.add(o) as *mut __m128i, _mm_load_si128(src.add(o) as *const __m128i));
        o += 16;
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn flush_nt(src: *const u8, dst: *mut u8, bytes: usize) {
    std::ptr::copy_nonoverlapping(src, dst, bytes);
}

fn sfence() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence()
    };
}

const NT: u8 = 0;
const PLAIN: u8 = 1;
const DIRECT: u8 = 2;

fn mode_name(mode: u8) -> &'static str {
    match mode {
        NT => "nt",
        PLAIN => "plain",
        _ => "direct",
    }
}

/// Scatter `records` 16-byte records to 512 destinations on pre-faulted
/// huge pages. `NT` and `PLAIN` stage 12 records per destination in a
/// 192-byte buffer and flush it with non-temporal or ordinary stores;
/// `DIRECT` stores each record straight to its destination. Returns the
/// timed nanoseconds.
fn scatter<const MODE: u8>(records: usize, seed: u64, sync: Option<&Barrier>) -> Result<(u64, u64), String> {
    const D: usize = 512;
    const B: usize = 192;
    let setup = (|| {
        let avg = records / D;
        let cap = (avg + avg / 8 + 1024) * 16 / B * B + B;
        let dst = Region::new(D * cap, Some(true))?;
        unsafe { std::ptr::write_bytes(dst.ptr, 0, D * cap) };
        let buf = Region::new(D * B, None)?;
        unsafe { std::ptr::write_bytes(buf.ptr, 0, D * B) };
        Ok::<_, String>((dst, buf, cap))
    })();
    if let Some(b) = sync {
        b.wait();
    }
    let result = setup.map(|(dst, buf, cap)| {
        let mut fill = [0usize; D];
        let mut pos = [0usize; D];
        let mut x = seed | 1;
        let t0 = now_ns();
        for i in 0..records {
            x = xorshift(x);
            let d = (x >> 55) as usize;
            if MODE == DIRECT {
                unsafe {
                    let slot = dst.ptr.add(d * cap + pos[d]) as *mut u64;
                    *slot = x;
                    *slot.add(1) = i as u64;
                }
                pos[d] += 16;
                if pos[d] + 16 > cap {
                    pos[d] = 0;
                }
                continue;
            }
            unsafe {
                let slot = buf.ptr.add(d * B + fill[d] * 16) as *mut u64;
                *slot = x;
                *slot.add(1) = i as u64;
            }
            fill[d] += 1;
            if fill[d] == B / 16 {
                unsafe {
                    let (src, to) = (buf.ptr.add(d * B), dst.ptr.add(d * cap + pos[d]));
                    if MODE == NT {
                        flush_nt(src, to, B);
                    } else {
                        std::ptr::copy_nonoverlapping(src, to, B);
                    }
                }
                pos[d] += B;
                if pos[d] + B > cap {
                    pos[d] = 0;
                }
                fill[d] = 0;
            }
        }
        sfence();
        let t1 = now_ns();
        black_box(pos);
        black_box(dst.ptr);
        (t0, t1)
    });
    if let Some(b) = sync {
        b.wait();
    }
    result
}

fn scatter_name(threads: usize, mode: u8) -> String {
    if mode == NT {
        format!("scatter{threads}")
    } else {
        format!("scatter{threads}_{}", mode_name(mode))
    }
}

fn scatter_pair<const MODE: u8>(records: usize) -> (Option<f64>, Option<f64>) {
    let how = match MODE {
        NT => "via 192B buffers, nt stores",
        PLAIN => "via 192B buffers, plain stores",
        _ => "direct 16B stores, no buffer",
    };
    let one = scatter::<MODE>(records, 77, None).map(|(t0, t1)| (t1 - t0) as f64 / records as f64);
    let r = one.clone().map(|per| {
        format!("1 thread, {records} records of 16B to 512 dests {how}, dest pre-faulted hugepage: ns_per_record={per:.3}")
    });
    bench(&scatter_name(1, MODE), r);
    let threads = 8;
    let barrier = Barrier::new(threads + 1);
    let res = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let barrier = &barrier;
                sc.spawn(move || scatter::<MODE>(records / threads, 1000 + t as u64, Some(barrier)))
            })
            .collect();
        barrier.wait();
        barrier.wait();
        hs.into_iter().map(|h| h.join().unwrap_or(Err("join".into()))).collect::<Vec<_>>()
    });
    let spans: Vec<Option<(u64, u64)>> = res.iter().map(|r| r.as_ref().ok().copied()).collect();
    let eight = if let Some((a, b)) = span(&spans) {
        let per_thread: f64 =
            spans.iter().flatten().map(|&(t0, t1)| (t1 - t0) as f64).sum::<f64>() / threads as f64 / (records / threads) as f64;
        let agg = (b - a) as f64 / records as f64;
        bench(
            &scatter_name(8, MODE),
            Ok(format!(
                "8 threads x {} records, own dests, {how}: per_thread_ns_per_record={per_thread:.3} aggregate_ns_per_record={agg:.3}",
                records / threads
            )),
        );
        Some(per_thread)
    } else {
        bench(&scatter_name(8, MODE), Err("a thread failed".to_string()));
        None
    };
    (one.ok(), eight)
}

fn b_scatter() {
    let records = 400 * MB / 16;
    let nt = scatter_pair::<NT>(records);
    let plain = scatter_pair::<PLAIN>(records);
    let direct = scatter_pair::<DIRECT>(records);
    let f = |v: Option<f64>| v.map_or("ERR".to_string(), |v| format!("{v:.2}"));
    if let Some(v) = nt.0 {
        fact("nt1", &format!("{v:.2}"));
    }
    if let Some(v) = nt.1 {
        fact("nt8", &format!("{v:.2}"));
    }
    fact("sc1_nt/plain/direct", &format!("{}/{}/{}", f(nt.0), f(plain.0), f(direct.0)));
    fact("sc8_nt/plain/direct", &format!("{}/{}/{}", f(nt.1), f(plain.1), f(direct.1)));
}

/// Batched random lookups into the table at `table` (`slots` units of
/// `stride` bytes): per block of 128, a pass that computes indices and
/// prefetches, then a pass that reads one u64 per lookup. Returns ns per
/// lookup.
fn lookups(table: usize, slots: usize, stride: usize, count: usize, seed: u64) -> f64 {
    const BLK: usize = 128;
    let base = table as *const u8;
    let mut idx = [0usize; BLK];
    let mut x = seed | 1;
    let mut acc = 0u64;
    let t0 = now_ns();
    for _ in 0..count / BLK {
        for i in idx.iter_mut() {
            x = xorshift(x);
            *i = ((x as u128 * slots as u128) >> 64) as usize * stride;
            crate::sys::prefetch(unsafe { base.add(*i) });
        }
        for &i in idx.iter() {
            acc = acc.wrapping_add(unsafe { std::ptr::read(base.add(i) as *const u64) });
        }
    }
    let t1 = now_ns();
    black_box(acc);
    (t1 - t0) as f64 / (count / BLK * BLK) as f64
}

fn lookups8(table: usize, slots: usize, stride: usize, count: usize) -> Option<(f64, f64)> {
    let threads = 8;
    let barrier = Barrier::new(threads);
    let v: Vec<f64> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let barrier = &barrier;
                sc.spawn(move || {
                    barrier.wait();
                    lookups(table, slots, stride, count, 500 + t as u64)
                })
            })
            .collect();
        hs.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    if v.len() != threads {
        return None;
    }
    Some((v.iter().sum::<f64>() / threads as f64, v.iter().cloned().fold(0.0, f64::max)))
}

fn b_lookup() {
    const N: usize = 2_000_000;
    let mut keys = Vec::new();
    for mb in [32usize, 128] {
        let r = (|| {
            let len = mb * MB;
            let reg = Region::new(len, Some(true))?;
            fill(reg.u64s(), len / 8, 17);
            let slots = len / 64;
            let one = lookups(reg.ptr as usize, slots, 64, N, 3);
            let (mean, max) = lookups8(reg.ptr as usize, slots, 64, N).ok_or("a thread failed")?;
            keys.push(format!("{one:.1}/{mean:.1}"));
            Ok(format!(
                "{mb}MB table of 64B buckets, hugepage, {N} lookups per thread in blocks of 128 (prefetch pass, read pass): 1thread_ns_per_lookup={one:.3} 8threads_mean_ns_per_lookup={mean:.3} max={max:.3}"
            ))
        })();
        if r.is_err() {
            keys.push("ERR".to_string());
        }
        bench(&format!("lookup_{mb}MB"), r);
    }
    fact("lk32_1t/8t", keys.first().map_or("ERR", |s| s.as_str()));
    fact("lk128_1t/8t", keys.get(1).map_or("ERR", |s| s.as_str()));
    let mut bl = Vec::new();
    for mb in [1usize, 4] {
        let r = (|| {
            let len = mb * MB;
            let reg = Region::new(len, Some(true))?;
            fill(reg.u64s(), len / 8, 23);
            let (mean, max) = lookups8(reg.ptr as usize, len / 8, 8, N).ok_or("a thread failed")?;
            bl.push(format!("{mean:.2}"));
            Ok(format!(
                "{mb}MB bit array, one random u64 word per probe, {N} probes per thread in blocks of 128 (prefetch pass, read pass), 8 threads: mean_ns_per_probe={mean:.3} max={max:.3}"
            ))
        })();
        if r.is_err() {
            bl.push("ERR".to_string());
        }
        bench(&format!("bloom_{mb}MB"), r);
    }
    fact("bloom1/4MB", &bl.join("/"));
}

const CHUNK: usize = 16 << 10;

/// Write every chunk of `order` (chunk numbers in `base`) in list order
/// with non-temporal 16-byte stores.
fn write_chunks(base: *mut u8, order: &[u32], seed: u64) {
    for (k, &c) in order.iter().enumerate() {
        let p = unsafe { base.add(c as usize * CHUNK) };
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{__m128i, _mm_set_epi64x, _mm_stream_si128};
            let mut o = 0;
            while o < CHUNK {
                _mm_stream_si128(p.add(o) as *mut __m128i, _mm_set_epi64x((k * CHUNK + o) as i64, seed as i64));
                o += 16;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        unsafe {
            let q = p as *mut u64;
            for j in 0..CHUNK / 8 {
                *q.add(j) = (k * CHUNK + j) as u64 ^ seed;
            }
        }
    }
    sfence();
}

/// Sum the chunks in list order, one 64-byte line at a time; with `PF`,
/// prefetch the line 1 KiB ahead, crossing into the next chunk at the end.
fn read_chunks<const PF: bool>(base: *const u8, order: &[u32]) -> u64 {
    let mut acc = 0u64;
    for k in 0..order.len() {
        let p = unsafe { base.add(order[k] as usize * CHUNK) };
        let next = order.get(k + 1).map(|&c| unsafe { base.add(c as usize * CHUNK) });
        let mut o = 0;
        while o < CHUNK {
            if PF {
                let ahead = o + 1024;
                if ahead < CHUNK {
                    crate::sys::prefetch(unsafe { p.add(ahead) });
                } else if let Some(n) = next {
                    crate::sys::prefetch(unsafe { n.add(ahead - CHUNK) });
                }
            }
            let line = unsafe { std::slice::from_raw_parts(p.add(o) as *const u64, 8) };
            acc = acc.wrapping_add(line.iter().fold(0u64, |a, &v| a.wrapping_add(v)));
            o += 64;
        }
    }
    acc
}

/// A list of 16 KB chunks over `bytes` of pre-faulted huge-page memory,
/// in a random order.
fn chunk_list(bytes: usize, seed: u64) -> Result<(Region, Vec<u32>), String> {
    let reg = Region::new(bytes, Some(true))?;
    unsafe { std::ptr::write_bytes(reg.ptr, 0, bytes) };
    let n = bytes / CHUNK;
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut x = seed | 1;
    for i in (1..n).rev() {
        x = xorshift(x);
        order.swap(i, (x % (i as u64 + 1)) as usize);
    }
    Ok((reg, order))
}

/// Write the list, then time its sequential re-read, twice: without and
/// with prefetch. Returns the two read times in ns. With `sync`, each
/// write and each read sits between barriers so main can time the reads
/// of all threads together.
fn reread(bytes: usize, seed: u64, sync: Option<&Barrier>) -> Result<[(u64, u64); 2], String> {
    let wait = || {
        if let Some(b) = sync {
            b.wait();
        }
    };
    let list = chunk_list(bytes, seed);
    let mut out = [(0u64, 0u64); 2];
    for (pass, slot) in out.iter_mut().enumerate() {
        wait();
        if let Ok((reg, order)) = &list {
            write_chunks(reg.ptr, order, seed + pass as u64);
        }
        wait();
        if let Ok((reg, order)) = &list {
            let t0 = now_ns();
            black_box(if pass == 0 { read_chunks::<false>(reg.ptr, order) } else { read_chunks::<true>(reg.ptr, order) });
            *slot = (t0, now_ns());
        }
        wait();
    }
    list.map(|_| out)
}

fn b_reread() {
    let len = 256 * MB;
    let r = reread(len, 41, None).map(|[a, b]| {
        let (g0, g1) = (len as f64 / (a.1 - a.0) as f64, len as f64 / (b.1 - b.0) as f64);
        format!("1 thread, 256MB as 16KB chunks in random order, nt-written then summed: GBps_no_prefetch={g0:.3} GBps_prefetch_1KiB={g1:.3}")
    });
    bench("reread1", r);
    let threads = 8;
    let per = 128 * MB;
    let barrier = Barrier::new(threads + 1);
    let res = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let barrier = &barrier;
                sc.spawn(move || reread(per, 60 + t as u64, Some(barrier)))
            })
            .collect();
        for _ in 0..6 {
            barrier.wait();
        }
        hs.into_iter().map(|h| h.join().unwrap_or(Err("join".into()))).collect::<Vec<_>>()
    });
    let pass = |k: usize| -> Vec<Option<(u64, u64)>> { res.iter().map(|r| r.as_ref().ok().map(|v| v[k])).collect() };
    let r = if let (Some(w0), Some(w1)) = (span(&pass(0)), span(&pass(1))) {
        let total = (threads * per) as f64;
        let (g0, g1) = (total / (w0.1 - w0.0) as f64, total / (w1.1 - w1.0) as f64);
        let rate = |k: usize| pass(k).iter().flatten().map(|v| per as f64 / (v.1 - v.0) as f64).sum::<f64>() / threads as f64;
        let (p0, p1) = (rate(0), rate(1));
        fact("rr8_nopf/pf_GBps", &format!("{g0:.1}/{g1:.1}"));
        Ok(format!(
            "8 threads, private 128MB lists of 16KB chunks: aggregate_GBps_no_prefetch={g0:.3} aggregate_GBps_prefetch_1KiB={g1:.3} per_thread_GBps={p0:.3}/{p1:.3}"
        ))
    } else {
        Err("a thread failed".to_string())
    };
    bench("reread8", r);
}

fn spin(iters: u64, seed: u64) -> u64 {
    let mut x = seed | 1;
    for _ in 0..iters {
        x = xorshift(x).wrapping_mul(0x2545_F491_4F6C_DD1D);
    }
    x
}

fn b_compute() {
    const IT: u64 = 50_000_000;
    let t0 = now_ns();
    black_box(spin(black_box(IT), black_box(3)));
    let one = (now_ns() - t0) as f64 / IT as f64;
    fact("cpu1", &format!("{one:.3}"));
    bench("compute1", Ok(format!("xorshift64*multiply, {IT} dependent iterations, 1 thread: ns_per_iter={one:.4}")));
    let threads = 8;
    let barrier = Barrier::new(threads);
    let v: Vec<f64> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let barrier = &barrier;
                sc.spawn(move || {
                    barrier.wait();
                    let t0 = now_ns();
                    black_box(spin(black_box(IT), black_box(t as u64 + 5)));
                    (now_ns() - t0) as f64 / IT as f64
                })
            })
            .collect();
        hs.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    let max = v.iter().cloned().fold(0.0, f64::max);
    fact("cpu8", &format!("{mean:.3}"));
    bench("compute8", Ok(format!("same, 8 threads at once ({} joined): mean_ns_per_iter={mean:.4} max={max:.4}", v.len())));
    let r = (|| {
        let len = 64 * MB;
        let src = Region::new(len, Some(true))?;
        let dst = Region::new(len, Some(true))?;
        fill(src.u64s(), len / 8, 9);
        fill(dst.u64s(), len / 8, 10);
        let reps = 4;
        let t0 = now_ns();
        for _ in 0..reps {
            unsafe { std::ptr::copy_nonoverlapping(black_box(src.ptr as *const u8), black_box(dst.ptr), len) };
        }
        let big = (reps * len) as f64 / (now_ns() - t0) as f64;
        let small_len = 16 << 10;
        let small_reps = 65_536;
        let t0 = now_ns();
        for i in 0..small_reps {
            let o = (i & 1) * small_len;
            unsafe { std::ptr::copy_nonoverlapping(black_box(src.ptr.add(o) as *const u8), black_box(dst.ptr.add(o)), small_len) };
        }
        let small = (small_reps * small_len) as f64 / (now_ns() - t0) as f64;
        fact("memcpy", &format!("{big:.1}/{small:.1}"));
        Ok(format!("memcpy 64MB blocks x{reps}: GBps={big:.3}; 16KB blocks x{small_reps}: GBps={small:.3}"))
    })();
    bench("memcpy", r);
}

fn b_syscalls() {
    let n = 1_000_000;
    let t0 = now_ns();
    for _ in 0..n {
        black_box(clock(libc::CLOCK_MONOTONIC));
    }
    let cg = (now_ns() - t0) as f64 / n as f64;
    let n2 = 200_000;
    let t0 = now_ns();
    for _ in 0..n2 {
        black_box(unsafe { libc::syscall(libc::SYS_getpid) });
    }
    let gp = (now_ns() - t0) as f64 / n2 as f64;
    fact("sys", &format!("{cg:.1}/{gp:.1}"));
    bench("syscall", Ok(format!("clock_gettime_ns={cg:.2} ({n} calls) getpid_raw_ns={gp:.2} ({n2} calls)")));
    let largest = LARGEST.lock().ok().and_then(|l| l.clone());
    let r = (|| {
        let (path, len) = largest.ok_or("no events file")?;
        let len = len as usize;
        if len < 8192 {
            return Err("file too small".to_string());
        }
        let c = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| "bad path")?;
        let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
        if fd < 0 {
            return Err(last_err());
        }
        let mut buf = [0u8; 4096];
        let reads = 20_000;
        let pages = (len - 4096) / 4096;
        let mut x = 99u64;
        let mut bad = 0;
        let t0 = now_ns();
        for _ in 0..reads {
            x = xorshift(x);
            let off = (x as usize % (pages + 1)) * 4096;
            let got = unsafe { libc::pread(fd, buf.as_mut_ptr().cast(), 4096, off as libc::off_t) };
            bad += usize::from(got != 4096);
        }
        let ns = (now_ns() - t0) as f64 / reads as f64;
        unsafe { libc::close(fd) };
        fact("pread", &format!("{ns:.0}"));
        Ok(format!("pread 4KiB random offsets x{reads}: ns_per_call={ns:.1} short_reads={bad}"))
    })();
    bench("pread", r);
    let r = (|| {
        let reg = Region::new(2 * MB, Some(false))?;
        let reps = 20_000;
        unsafe { std::ptr::write_volatile(reg.ptr, 1u8) };
        let mut fails = 0;
        let f0 = minflt_thread();
        let t0 = now_ns();
        for i in 0..reps {
            unsafe {
                fails += usize::from(libc::madvise(reg.ptr.cast(), 4096, libc::MADV_DONTNEED) != 0);
                std::ptr::write_volatile(reg.ptr, i as u8);
            }
        }
        let ns = (now_ns() - t0) as f64 / reps as f64;
        let f = minflt_thread() - f0;
        fact("dontneed", &format!("{ns:.0}"));
        Ok(format!("madvise(DONTNEED)+refault 4KiB x{reps}: ns_per_pair={ns:.1} minflt={f} madvise_fails={fails}"))
    })();
    bench("dontneed", r);
}

fn b_threads() {
    let mut us = Vec::new();
    for _ in 0..10 {
        let t0 = now_ns();
        let hs: Vec<_> = (0..8).map(|i| std::thread::Builder::new().spawn(move || black_box(i))).collect();
        let mut ok = true;
        for h in hs {
            ok &= h.map(|h| h.join().is_ok()).unwrap_or(false);
        }
        if ok {
            us.push((now_ns() - t0) as f64 / 1e3);
        }
    }
    us.sort_by(|a, b| a.total_cmp(b));
    let r = if us.is_empty() {
        Err("spawn failed".to_string())
    } else {
        let med = us[us.len() / 2];
        fact("spawn8", &format!("{med:.0}"));
        Ok(format!(
            "create+join 8 threads, {} rounds: median_us={med:.1} min={:.1} max={:.1}",
            us.len(),
            us[0],
            us[us.len() - 1]
        ))
    };
    bench("threads", r);
}

// ---- after the pipeline

pub fn after(out: &Path, lk: &Lookups, started: Instant) {
    let total_ms = started.elapsed().as_secs_f64() * 1e3;
    let ru = rusage(libc::RUSAGE_SELF);
    put("pipeline", &format!("total_ms_incl_probes={total_ms:.1} minflt={} majflt={} maxrss_kb={}", ru.ru_minflt, ru.ru_majflt, ru.ru_maxrss));
    if let Ok(p) = PHASES.lock() {
        let s: Vec<String> = p.iter().map(|(n, w, c)| format!("{n}:{w:.1}/{c:.0}")).collect();
        fact("phase_wall/cpu_ms", &s.join(" "));
    }
    fact("total_ms", &format!("{total_ms:.1}"));
    fact("minflt_pipe", &ru.ru_minflt.to_string());
    let t0 = now_ns();
    b_seq();
    b_faults();
    b_chase();
    b_scatter();
    b_lookup();
    b_reread();
    b_compute();
    b_syscalls();
    b_threads();
    let bench_ms = (now_ns() - t0) as f64 / 1e6;
    put("bench_total_ms", &format!("{bench_ms:.1}"));
    fact("bench_ms", &format!("{bench_ms:.0}"));
    if let Some(s) = show("end.cg.cpu.stat", "/sys/fs/cgroup/cpu.stat") {
        let get = |k: &str| field(&s, k).unwrap_or_else(|| "?".to_string());
        fact("throttle", &format!("{}/{}", get("nr_throttled"), get("throttled_usec")));
    }
    show("end.cg.memory.stat", "/sys/fs/cgroup/memory.stat");
    show("end.cg.memory.current", "/sys/fs/cgroup/memory.current");
    if let Some(s) = show("end.cg.memory.peak", "/sys/fs/cgroup/memory.peak") {
        fact("mempeak_mb", &format!("{}", s.trim().parse::<u64>().map_or(0, |v| v >> 20)));
    }
    let ru = rusage(libc::RUSAGE_SELF);
    put("end.rusage", &format!("minflt={} majflt={} maxrss_kb={} nvcsw={} nivcsw={}", ru.ru_minflt, ru.ru_majflt, ru.ru_maxrss, ru.ru_nvcsw, ru.ru_nivcsw));
    fact("minflt_all", &ru.ru_minflt.to_string());
    let rows = pack_rows();
    write_rows(out, lk, &rows);
    put("rows", &format!("wrote {} marker rows to user_daily/part-diag.parquet", rows.len()));
    put("done_ms", &format!("{:.1}", started.elapsed().as_secs_f64() * 1e3));
}

const ORDER: &[&str] = &[
    "cpu", "nproc", "aff", "sib", "L2", "L3", "thp", "thpdefrag", "thpshmem", "fs_data", "fs_output", "fs_scratch",
    "data_tmpfs", "mincore", "cpumax", "cpuset", "memmax", "kernel", "phase_wall/cpu_ms", "total_ms", "dram1ns", "dram8ns",
    "l3ns", "l3ns4", "l2ns", "mlp8", "mlp16", "seq1", "seq8", "fault_huge_ms", "fault_small_ms", "ahp", "cpu1", "cpu8", "throttle",
    "minflt_pipe", "minflt_all", "mempeak_mb", "nt1", "nt8", "fill_nspb", "memcpy", "sys", "pread", "dontneed",
    "spawn8", "bench_ms", "sc1_nt/plain/direct", "sc8_nt/plain/direct", "lk32_1t/8t", "lk128_1t/8t",
    "rr8_nopf/pf_GBps", "bloom1/4MB",
];

fn pack_rows() -> Vec<String> {
    let facts = FACTS.lock().map(|f| f.clone()).unwrap_or_default();
    let mut items: Vec<String> = Vec::new();
    for k in ORDER {
        if let Some((_, v)) = facts.iter().find(|(f, _)| f == k) {
            items.push(format!("{k}={v}"));
        }
    }
    for (k, v) in &facts {
        if !ORDER.contains(&k.as_str()) {
            items.push(format!("{k}={v}"));
        }
    }
    let mut rows: Vec<String> = Vec::new();
    let mut cur = String::new();
    for it in items {
        let it: String = if it.len() > 390 { chunks(&it, 390)[0].to_string() } else { it };
        if !cur.is_empty() && cur.len() + 1 + it.len() > 400 {
            if rows.len() == 11 {
                break;
            }
            rows.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(';');
        }
        cur.push_str(&it);
    }
    if !cur.is_empty() && rows.len() < 12 {
        rows.push(cur);
    }
    rows
}

fn write_rows(out: &Path, lk: &Lookups, rows: &[String]) {
    let n = rows.len();
    if n == 0 {
        return;
    }
    let day = lk.window_start.div_euclid(86_400_000) as i32;
    let mut ids = Vec::new();
    let mut tops = Vec::new();
    for i in 0..n {
        plain_bytes(format!("~diag{i:02}").as_bytes(), &mut ids);
    }
    def_levels(rows.iter().map(|_| true), n, &mut tops);
    for r in rows {
        plain_bytes(r.as_bytes(), &mut tops);
    }
    let days: Vec<i32> = vec![day; n];
    let sessions: Vec<i32> = (1..=n as i32).collect();
    let events: Vec<i32> = vec![n as i32; n];
    let revenue: Vec<i32> = vec![0; n];
    let mut file = PqFile::create(&out.join("user_daily/part-diag.parquet"), &USER_SCHEMA_COMPACT);
    file.write_row_group(
        n,
        &[
            vec![Page::plain(n, vec![&ids])],
            vec![Page::plain(n, vec![as_bytes(&days)])],
            vec![Page::plain(n, vec![as_bytes(&sessions)])],
            vec![Page::plain(n, vec![as_bytes(&events)])],
            vec![Page::plain(n, vec![as_bytes(&revenue)])],
            vec![Page::plain(n, vec![&tops])],
        ],
    );
    file.finish();
}
