//! Diagnostic build only (cargo feature `diag`), version 3. Reports facts about the
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
    for (bytes, name, key) in [(4 * MB, "chase_4MB", "l3ns4")] {
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
    for mb in [32usize, 128] {
        let r = (|| {
            let len = mb * MB;
            let reg = Region::new(len, Some(true))?;
            fill(reg.u64s(), len / 8, 17);
            let (mean, max) = lookups8(reg.ptr as usize, len / 64, 64, N).ok_or("a thread failed")?;
            fact(&format!("lk{mb}_8t"), &format!("{mean:.1}"));
            Ok(format!(
                "{mb}MB table of 64B buckets, hugepage, {N} lookups per thread in blocks of 128 (prefetch pass, read pass), 8 threads: mean_ns_per_lookup={mean:.3} max={max:.3}"
            ))
        })();
        bench(&format!("lookup_{mb}MB"), r);
    }
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
}

// ---- start, phase and end blocks

/// The vmstat counters sampled at every phase boundary.
const VM_PHASE: [&str; 7] =
    ["compact_stall", "compact_fail", "compact_success", "thp_fault_alloc", "thp_fault_fallback", "thp_fault_fallback_charge", "pgfault"];

const VM_PREFIXES: [&str; 14] = [
    "compact_",
    "thp_",
    "pgfault",
    "pgmajfault",
    "allocstall",
    "pgscan",
    "pgsteal",
    "pgalloc",
    "nr_free_pages",
    "nr_anon_transparent_hugepages",
    "nr_shmem",
    "nr_file_pages",
    "kswapd",
    "numa_",
];

/// Process CPU and faults plus the host-wide vmstat counters at one moment.
/// `vm` holds `VM_PHASE` in order, then the sum of all `allocstall_*`.
#[derive(Clone, Copy, Default)]
struct Snap {
    user_us: i64,
    sys_us: i64,
    minflt: i64,
    vm_ok: bool,
    vm: [u64; 8],
}

static BASE: Mutex<Option<Snap>> = Mutex::new(None);
static LAST: Mutex<Option<Snap>> = Mutex::new(None);
static PHASE_SNAPS: Mutex<Vec<(String, Snap)>> = Mutex::new(Vec::new());

fn vmstat() -> Result<Vec<(String, u64)>, String> {
    let s = read("/proc/vmstat").map_err(|e| format!("ERR {e}"))?;
    Ok(s.lines()
        .filter_map(|l| {
            let (k, v) = l.split_once(' ')?;
            Some((k.to_string(), v.trim().parse::<u64>().ok()?))
        })
        .collect())
}

fn snap() -> Snap {
    let ru = rusage(libc::RUSAGE_SELF);
    let us = |t: libc::timeval| t.tv_sec as i64 * 1_000_000 + t.tv_usec as i64;
    let mut s = Snap { user_us: us(ru.ru_utime), sys_us: us(ru.ru_stime), minflt: ru.ru_minflt as i64, ..Snap::default() };
    if let Ok(vm) = vmstat() {
        s.vm_ok = true;
        for (i, k) in VM_PHASE.iter().enumerate() {
            s.vm[i] = vm.iter().find(|(n, _)| n == k).map_or(0, |(_, v)| *v);
        }
        s.vm[7] = vm.iter().filter(|(n, _)| n.starts_with("allocstall")).map(|(_, v)| *v).sum();
    }
    s
}

fn delta(a: &Snap, b: &Snap) -> Snap {
    let mut d = Snap {
        user_us: b.user_us - a.user_us,
        sys_us: b.sys_us - a.sys_us,
        minflt: b.minflt - a.minflt,
        vm_ok: a.vm_ok && b.vm_ok,
        vm: [0; 8],
    };
    for i in 0..8 {
        d.vm[i] = b.vm[i].wrapping_sub(a.vm[i]);
    }
    d
}

/// Compact text of a delta: user and system ms, minor faults, then
/// compact_stall, thp_fault_alloc, thp_fault_fallback.
fn short(d: &Snap) -> String {
    let vm = |i: usize| if d.vm_ok { d.vm[i].to_string() } else { "?".to_string() };
    format!("u{:.1}/s{:.1}/f{}/cs{}/ta{}/tf{}", d.user_us as f64 / 1e3, d.sys_us as f64 / 1e3, d.minflt, vm(0), vm(3), vm(4))
}

fn long(d: &Snap) -> String {
    let mut s = format!("user_ms={:.1} sys_ms={:.1} minflt={}", d.user_us as f64 / 1e3, d.sys_us as f64 / 1e3, d.minflt);
    if d.vm_ok {
        for (i, k) in VM_PHASE.iter().enumerate() {
            s.push_str(&format!(" {k}={}", d.vm[i]));
        }
        s.push_str(&format!(" allocstall_sum={}", d.vm[7]));
    } else {
        s.push_str(" vmstat=ERR");
    }
    s
}

/// Called at each engine phase boundary, where the engine prints its
/// timing line.
pub fn phase(name: &str, wall_ms: f64, cpu_ms: f64) {
    let now = snap();
    let prev = LAST.lock().ok().and_then(|mut l| l.replace(now));
    let d = prev.map_or(Snap::default(), |p| delta(&p, &now));
    put(&format!("phase.{name}"), &format!("wall_ms={wall_ms:.1} cpu_ms={cpu_ms:.1} {}", long(&d)));
    if let Ok(mut p) = PHASES.lock() {
        p.push((name.to_string(), wall_ms, cpu_ms));
    }
    if let Ok(mut p) = PHASE_SNAPS.lock() {
        p.push((name.to_string(), d));
    }
}

/// The baseline for phase deltas: called right after the engine's start
/// instant is taken.
pub fn mark() {
    let s = snap();
    if let Ok(mut b) = BASE.lock() {
        *b = Some(s);
    }
    if let Ok(mut l) = LAST.lock() {
        *l = Some(s);
    }
}

fn burst() -> f64 {
    let t0 = now_ns();
    let mut it = 0u64;
    let mut x = 1u64;
    while now_ns() - t0 < 10_000_000 {
        x = spin(black_box(20_000), x);
        it += 20_000;
    }
    black_box(x);
    (now_ns() - t0) as f64 / it as f64
}

fn compute_block(tag: &str) {
    let one = burst();
    let threads = 8;
    let barrier = Barrier::new(threads);
    let v: Vec<f64> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                let barrier = &barrier;
                sc.spawn(move || {
                    barrier.wait();
                    burst()
                })
            })
            .collect();
        hs.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    let max = v.iter().cloned().fold(0.0, f64::max);
    put(
        &format!("{tag}.compute"),
        &format!("10ms bursts of the compute loop: 1thread_ns_per_iter={one:.4} 8threads_mean={mean:.4} max={max:.4} ({} joined)", v.len()),
    );
    fact(&format!("c1.{}", &tag[..1]), &format!("{one:.3}"));
    fact(&format!("c8.{}", &tag[..1]), &format!("{mean:.3}"));
    fact(&format!("c8max.{}", &tag[..1]), &format!("{max:.3}"));
}

fn mem_block(tag: &str) {
    match vmstat() {
        Ok(vm) => {
            let lines: Vec<String> = vm
                .iter()
                .filter(|(k, _)| VM_PREFIXES.iter().any(|p| k.starts_with(p)))
                .map(|(k, v)| format!("{k} {v}"))
                .collect();
            put(&format!("{tag}.vmstat"), &lines.join(", "));
        }
        Err(e) => put(&format!("{tag}.vmstat"), &e),
    }
    match read("/proc/buddyinfo") {
        Ok(s) => {
            put(&format!("{tag}.buddyinfo"), &s);
            let (mut o9, mut o10) = (0u64, 0u64);
            let mut zones = Vec::new();
            for l in s.lines() {
                let t: Vec<&str> = l.split_whitespace().collect();
                let at = |i: usize| t.get(i).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
                if t.len() >= 15 {
                    o9 += at(13);
                    o10 += at(14);
                    zones.push(format!("{}:{}/{}", t[3], at(13), at(14)));
                }
            }
            put(&format!("{tag}.buddy_o9_o10"), &format!("sum_order9={o9} sum_order10={o10} zones {}", zones.join(" ")));
            fact(&format!("b9.{}", &tag[..1]), &o9.to_string());
            fact(&format!("b10.{}", &tag[..1]), &o10.to_string());
        }
        Err(e) => put(&format!("{tag}.buddyinfo"), &format!("ERR {e}")),
    }
    match read("/proc/meminfo") {
        Ok(s) => {
            let keys = ["MemFree", "MemAvailable", "AnonHugePages", "ShmemHugePages", "Shmem", "Cached"];
            put(
                &format!("{tag}.meminfo"),
                &kv_lines(&s, &keys).iter().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).collect::<Vec<_>>().join(", "),
            );
            let mb = |k: &str| field(&s, k).and_then(|v| v.split_whitespace().next()?.parse::<u64>().ok()).map_or("?".to_string(), |kb| (kb >> 10).to_string());
            fact(&format!("free.{}", &tag[..1]), &mb("MemFree"));
            fact(&format!("avail.{}", &tag[..1]), &mb("MemAvailable"));
        }
        Err(e) => put(&format!("{tag}.meminfo"), &format!("ERR {e}")),
    }
    let k = "/sys/kernel/mm/transparent_hugepage/khugepaged";
    let parts: Vec<String> = ["pages_to_scan", "scan_sleep_millisecs", "alloc_sleep_millisecs", "max_ptes_none"]
        .iter()
        .map(|f| format!("{f}={}", read1(&format!("{k}/{f}"))))
        .collect();
    put(&format!("{tag}.khugepaged"), &parts.join(" "));
}

/// First touch of 64 MB, one 4 KiB write per page, timed per 2 MB step.
fn touch_steps(huge: bool) -> Result<(f64, f64, f64, i64, String), String> {
    let len = 64 * MB;
    let reg = Region::new(len, Some(huge))?;
    let mut steps: Vec<u64> = Vec::with_capacity(len / (2 * MB));
    let f0 = minflt_thread();
    let t0 = now_ns();
    for c in 0..len / (2 * MB) {
        let a = now_ns();
        let mut off = c * 2 * MB;
        while off < (c + 1) * 2 * MB {
            unsafe { std::ptr::write_volatile(reg.ptr.add(off), 1u8) };
            off += 4096;
        }
        steps.push(now_ns() - a);
    }
    let total = (now_ns() - t0) as f64 / 1e6;
    let faults = minflt_thread() - f0;
    let ahp = smaps_entry(reg.ptr as usize).ok().and_then(|b| field(&b, "AnonHugePages")).unwrap_or_else(|| "?".into());
    steps.sort_unstable();
    let max = *steps.last().unwrap_or(&0) as f64 / 1e6;
    let med = steps.get(steps.len() / 2).copied().unwrap_or(0) as f64 / 1e6;
    Ok((total, max, med, faults, ahp))
}

const MADV_POPULATE_WRITE: i32 = 23;

fn populate() -> Result<(f64, String), String> {
    let len = 64 * MB;
    let reg = Region::new(len, Some(true))?;
    let t0 = now_ns();
    let rc = unsafe { libc::madvise(reg.ptr.cast(), len, MADV_POPULATE_WRITE) };
    let ms = (now_ns() - t0) as f64 / 1e6;
    if rc != 0 {
        return Err(format!("madvise(MADV_POPULATE_WRITE) {}", last_err()));
    }
    let ahp = smaps_entry(reg.ptr as usize).ok().and_then(|b| field(&b, "AnonHugePages")).unwrap_or_else(|| "?".into());
    Ok((ms, ahp))
}

fn touch_block(tag: &str) {
    for huge in [true, false] {
        let name = format!("{tag}.touch64_{}", if huge { "huge" } else { "small" });
        match touch_steps(huge) {
            Ok((total, max, med, faults, ahp)) => {
                put(
                    &name,
                    &format!("total_ms={total:.3} slowest_2MB_ms={max:.3} median_2MB_ms={med:.3} minflt={faults} AnonHugePages={ahp}"),
                );
                let ahp_kb = ahp.split_whitespace().next().unwrap_or("?").to_string();
                if huge {
                    fact(&format!("th.{}", &tag[..1]), &format!("{total:.1}/{max:.2}/{med:.2}/{ahp_kb}"));
                } else {
                    fact(&format!("ts.{}", &tag[..1]), &format!("{total:.1}/{max:.2}"));
                }
            }
            Err(e) => put(&name, &format!("ERR {e}")),
        }
    }
    match populate() {
        Ok((ms, ahp)) => {
            put(&format!("{tag}.populate64_huge"), &format!("ms={ms:.3} AnonHugePages={ahp}"));
            fact(&format!("pop.{}", &tag[..1]), &format!("{ms:.1}/{}", ahp.split_whitespace().next().unwrap_or("?")));
        }
        Err(e) => {
            put(&format!("{tag}.populate64_huge"), &format!("ERR {e}"));
            fact(&format!("pop.{}", &tag[..1]), "ERR");
        }
    }
}

/// START block, first thing in `main`, then the static probes.
pub fn start() {
    let t_entry = now_ns();
    let real = clock(libc::CLOCK_REALTIME);
    let mono = clock(libc::CLOCK_MONOTONIC);
    emit("DIAG begin: diagnostic build v3: START block, static probes, pipeline with phase deltas, END block, benches");
    let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    let start_ticks = read("/proc/self/stat").ok().and_then(|s| {
        let after = &s[s.rfind(')')? + 1..];
        after.split_whitespace().nth(19)?.parse::<u64>().ok()
    });
    let uptime = read("/proc/uptime").ok().and_then(|s| s.split_whitespace().next()?.parse::<f64>().ok());
    let delay = match (start_ticks, uptime, hz > 0) {
        (Some(st), Some(up), true) => format!("{:.0}", (up - st as f64 / hz as f64) * 1e3),
        _ => "?".to_string(),
    };
    put(
        "START.clock",
        &format!(
            "realtime={}.{:09} monotonic={}.{:09} starttime_ticks={} clk_tck={hz} uptime_s={} start_to_main_ms~{delay}",
            real.0,
            real.1,
            mono.0,
            mono.1,
            start_ticks.map_or("ERR".to_string(), |v| v.to_string()),
            uptime.map_or("ERR".to_string(), |v| v.to_string())
        ),
    );
    fact("sdelay_ms", &delay);
    compute_block("START");
    mem_block("START");
    touch_block("START");
    put("START.block_ms", &format!("{:.3}", (now_ns() - t_entry) as f64 / 1e6));
    before();
}

fn end_block() {
    let t0 = now_ns();
    compute_block("END");
    mem_block("END");
    touch_block("END");
    put("END.block_ms", &format!("{:.3}", (now_ns() - t0) as f64 / 1e6));
}

pub fn after(out: &Path, lk: &Lookups, started: Instant) {
    let total_ms = started.elapsed().as_secs_f64() * 1e3;
    let now = snap();
    let base = BASE.lock().ok().and_then(|b| *b);
    let all = base.map_or(Snap::default(), |b| delta(&b, &now));
    put("pipeline", &format!("total_ms={total_ms:.1} {}", long(&all)));
    fact("all", &format!("{}/cf{}/cok{}/tfc{}/pgf{}/as{}", short(&all), all.vm[1], all.vm[2], all.vm[5], all.vm[6], all.vm[7]));
    if let Ok(p) = PHASES.lock() {
        let s: Vec<String> = p.iter().map(|(n, w, c)| format!("{n}:{w:.1}/{c:.0}")).collect();
        fact("phase_wall/cpu_ms", &s.join(" "));
    }
    fact("total_ms", &format!("{total_ms:.1}"));
    let ru = rusage(libc::RUSAGE_SELF);
    fact("minflt_pipe", &ru.ru_minflt.to_string());
    end_block();
    let t0 = now_ns();
    b_seq();
    b_faults();
    b_chase();
    b_lookup();
    b_compute();
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
    for (i, r) in rows.iter().enumerate() {
        put(&format!("row{i:02}"), r);
    }
    write_rows(out, lk, &rows);
    put("rows", &format!("wrote {} marker rows to user_daily/part-diag.parquet", rows.len()));
    put("done_ms", &format!("{:.1}", started.elapsed().as_secs_f64() * 1e3));
}

/// Old priority facts, used after the new ones if rows remain.
const ORDER: &[&str] = &[
    "cpu", "nproc", "aff", "sib", "L2", "L3", "thp", "thpdefrag", "thpshmem", "fs_data", "fs_output", "fs_scratch",
    "data_tmpfs", "mincore", "cpumax", "cpuset", "memmax", "kernel", "total_ms", "dram1ns", "l3ns4", "mlp8", "mlp16",
    "seq8", "fault_huge_ms", "fault_small_ms", "ahp", "cpu1", "cpu8", "throttle", "minflt_pipe", "minflt_all",
    "mempeak_mb", "lk32_8t", "lk128_8t", "bench_ms",
];

fn pack_rows() -> Vec<String> {
    let facts = FACTS.lock().map(|f| f.clone()).unwrap_or_default();
    let get = |k: &str| facts.iter().find(|(f, _)| f == k).map(|(_, v)| v.clone());
    let pair = |k: &str| format!("{k}={}|{}", get(&format!("{k}.S")).unwrap_or("?".into()), get(&format!("{k}.E")).unwrap_or("?".into()));
    let mut items: Vec<String> = Vec::new();
    for k in ["c1", "c8", "c8max", "th", "ts", "pop", "b9", "b10", "free", "avail"] {
        items.push(pair(k));
    }
    if let Ok(p) = PHASE_SNAPS.lock() {
        for (name, d) in p.iter() {
            items.push(format!("{name}={}", short(d)));
        }
    }
    let mut used: Vec<String> = Vec::new();
    for k in ["all", "phase_wall/cpu_ms", "sdelay_ms"] {
        if let Some(v) = get(k) {
            items.push(format!("{k}={v}"));
            used.push(k.to_string());
        }
    }
    for k in ORDER {
        if used.iter().any(|u| u == k) {
            continue;
        }
        if let Some(v) = get(k) {
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
