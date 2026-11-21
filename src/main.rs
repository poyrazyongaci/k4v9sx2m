mod alloc;
mod common;
mod engine;
mod ids;
mod naive;
mod output;
mod parquet;
mod scan;
mod sys;
#[cfg(test)]
mod tests;
mod ts;

use std::path::PathBuf;

#[global_allocator]
static GLOBAL: alloc::Bump = alloc::Bump;

fn main() {
    let started = std::time::Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let env = |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.to_string());
    let (data, out) = if args.len() >= 3 {
        (args[args.len() - 2].clone(), args[args.len() - 1].clone())
    } else {
        (env("DATA_DIR", "/data"), env("OUTPUT_DIR", "/output"))
    };
    let (data, out) = (PathBuf::from(data), PathBuf::from(out));
    let lookups = common::load_lookups(&data);
    // The image has no shell or chmod, and the run is uid 1000: leave the
    // output removable by whoever owns the mounted folder.
    unsafe { libc::umask(0) };
    output::make_dirs(&out);
    let cfg = engine::Config::from_env();
    let naive_only = std::env::var("ENGINE").is_ok_and(|v| v == "naive");
    if naive_only || engine::run(&data, Some(&out), &lookups, &cfg, started).is_none() {
        let tables = naive::run(&data, &lookups);
        output::write_tables(&out, &tables);
    }
    sys::exit_now(0);
}
