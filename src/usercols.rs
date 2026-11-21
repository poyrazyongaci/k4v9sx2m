//! user_daily output of one thread.
//!
//! Output bytes are not free: when the container stops, the dirty pages of
//! /output are written to disk before the run counts as finished (measured:
//! about 2.2 ms per MB on the test box). So rows are routed to three
//! buffers, each a row group shape that standard encodings store tightly:
//!
//! user_id is a dictionary with one RLE run per person in every shape.
//!
//! * quiet rows (no purchase, under 16 sessions and 256 events): the day,
//!   session and event counts are dictionary indices of 4, 4 and 8 bits,
//!   revenue is one run of zeros and top_category one run of nulls;
//! * purchase rows with small counts: the same, plus plain 32-bit revenue
//!   and a 5-bit category index;
//! * everything else (a whale's day): plain 32-bit columns, in the same
//!   file. Only a row group with revenue over 2^31 goes to a second file
//!   whose revenue column is 64-bit.
//!
//! Only PLAIN and RLE_DICTIONARY pages (data page v1) are written.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::common::*;
use crate::ids::{self, FALLBACK};
use crate::output::{USER_SCHEMA, USER_SCHEMA_COMPACT};
use crate::parquet::{as_bytes, bitpack_run, def_levels, plain_bytes, rle_run, Page, PqFile};

const FLUSH_MAX: usize = 1 << 17;
const MAX_SESSIONS: u32 = 16;
const MAX_EVENTS: u32 = 256;

/// The user_id column of one row group. A person's rows are pushed one after
/// another, so the column is a dictionary of the distinct ids in order of
/// arrival and one RLE run per id: no hash map, about 16 bytes per person
/// instead of 13 per row.
#[derive(Default)]
struct IdCol {
    /// Dictionary page body: PLAIN byte arrays.
    dict: Vec<u8>,
    /// Rows per dictionary entry.
    runs: Vec<u32>,
    last: u64,
}

impl IdCol {
    fn with_capacity(rows: usize) -> Self {
        IdCol { dict: Vec::with_capacity(rows * 13), runs: Vec::with_capacity(rows), last: 0 }
    }

    #[inline(always)]
    fn push(&mut self, pkey: u64) {
        if pkey == self.last && !self.runs.is_empty() {
            unsafe { *self.runs.last_mut().unwrap_unchecked() += 1 };
        } else {
            push_id(&mut self.dict, pkey);
            self.runs.push(1);
            self.last = pkey;
        }
    }

    /// The data page body: bit width, then one run per dictionary entry.
    fn encode(&self, out: &mut Vec<u8>) {
        let width = bits(self.runs.len().max(2) - 1);
        out.push(width as u8);
        for (index, &count) in self.runs.iter().enumerate() {
            rle_run(count as usize, index as u32, width, out);
        }
    }

    fn clear(&mut self) {
        self.dict.clear();
        self.runs.clear();
    }
}

#[derive(Default)]
struct Compact {
    ids: IdCol,
    day: Vec<u8>,
    sessions: Vec<u8>,
    events: Vec<u8>,
    revenue: Vec<i32>,
    top: Vec<u8>,
}

#[derive(Default)]
struct Wide {
    ids: IdCol,
    day: Vec<i32>,
    sessions: Vec<i32>,
    events: Vec<i32>,
    revenue: Vec<i64>,
    /// Category id, or 0xFF for null.
    top: Vec<u8>,
}

pub struct UserOut<'a> {
    quiet: Compact,
    buying: Compact,
    wide: Wide,
    dir: Option<PathBuf>,
    thread: usize,
    file: Option<PqFile>,
    wide_file: Option<PqFile>,
    collect: Option<&'a Mutex<Vec<UserRow>>>,
    cats: &'a [Vec<u8>],
    ws_day: i32,
    day_width: u32,
    cat_width: u32,
    day_dict: Vec<i32>,
    count_dict: Vec<i32>,
    cat_dict: Vec<u8>,
    scratch: [Vec<u8>; 6],
    flush_rows: usize,
    pub rows_written: u64,
}

#[inline(always)]
fn push_id(ids: &mut Vec<u8>, pkey: u64) {
    if pkey & FALLBACK == 0 {
        let mut buf = [0u8; 16];
        let len = ids::unpack_packed(pkey, &mut buf);
        ids.reserve(16);
        unsafe {
            let at = ids.as_mut_ptr().add(ids.len());
            (at as *mut u32).write_unaligned((len as u32).to_le());
            std::ptr::copy_nonoverlapping(buf.as_ptr(), at.add(4), 9);
            ids.set_len(ids.len() + 4 + len);
        }
    } else {
        let mut bytes = Vec::new();
        ids::unpack(pkey, &mut bytes);
        plain_bytes(&bytes, ids);
    }
}

fn bits(max: usize) -> u32 {
    (usize::BITS - max.leading_zeros()).max(1)
}

impl<'a> UserOut<'a> {
    pub fn new(
        dir: Option<&Path>,
        thread: usize,
        lk: &'a Lookups,
        ws_day: i64,
        ndays: usize,
        collect: Option<&'a Mutex<Vec<UserRow>>>,
    ) -> Self {
        let mut cat_dict = Vec::new();
        for cat in &lk.cats {
            plain_bytes(cat, &mut cat_dict);
        }
        let flush_rows = std::env::var("ENGINE_FLUSH_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).map_or(1 << 14, |n| n.clamp(1, FLUSH_MAX));
        let reserve = flush_rows + 4096;
        let compact = |with_revenue: bool| Compact {
            ids: IdCol::with_capacity(reserve),
            day: Vec::with_capacity(reserve),
            sessions: Vec::with_capacity(reserve),
            events: Vec::with_capacity(reserve),
            revenue: Vec::with_capacity(if with_revenue { reserve } else { 0 }),
            top: Vec::with_capacity(if with_revenue { reserve } else { 0 }),
        };
        UserOut {
            quiet: compact(false),
            buying: compact(true),
            wide: Wide::default(),
            dir: dir.map(Path::to_path_buf),
            thread,
            file: None,
            wide_file: None,
            collect,
            cats: &lk.cats,
            ws_day: ws_day as i32,
            day_width: bits(ndays.max(2) - 1),
            cat_width: bits(lk.cats.len().max(2) - 1),
            day_dict: (0..ndays as i32).map(|d| ws_day as i32 + d).collect(),
            count_dict: (0..MAX_EVENTS as i32).collect(),
            cat_dict,
            scratch: Default::default(),
            flush_rows,
            rows_written: 0,
        }
    }

    /// One row. `top` is a category id, or 0xFF when the day has no
    /// purchase (and then `revenue` is 0).
    #[inline(always)]
    pub fn push(&mut self, pkey: u64, day_rel: u32, sessions: u32, events: u32, revenue: u64, top: u8) {
        if sessions < MAX_SESSIONS && events < MAX_EVENTS && revenue <= i32::MAX as u64 && day_rel < 256 {
            let buf = if top == 0xFF { &mut self.quiet } else { &mut self.buying };
            buf.ids.push(pkey);
            buf.day.push(day_rel as u8);
            buf.sessions.push(sessions as u8);
            buf.events.push(events as u8);
            if top != 0xFF {
                buf.revenue.push(revenue as i32);
                buf.top.push(top);
            }
        } else {
            let buf = &mut self.wide;
            buf.ids.push(pkey);
            buf.day.push(self.ws_day + day_rel as i32);
            buf.sessions.push(sessions as i32);
            buf.events.push(events as i32);
            buf.revenue.push(revenue as i64);
            buf.top.push(top);
        }
    }

    /// Write out any buffer that has a full row group. True if one was.
    pub fn flush_full(&mut self) -> bool {
        let mut any = false;
        if self.quiet.day.len() >= self.flush_rows {
            self.flush_compact(false);
            any = true;
        }
        if self.buying.day.len() >= self.flush_rows {
            self.flush_compact(true);
            any = true;
        }
        if self.wide.day.len() >= self.flush_rows {
            self.flush_wide();
            any = true;
        }
        any
    }

    pub fn finish(&mut self) {
        if !self.quiet.day.is_empty() {
            self.flush_compact(false);
        }
        if !self.buying.day.is_empty() {
            self.flush_compact(true);
        }
        if !self.wide.day.is_empty() {
            self.flush_wide();
        }
        if let Some(file) = self.file.take() {
            file.finish();
        }
        if let Some(file) = self.wide_file.take() {
            file.finish();
        }
    }

    fn collect_rows(
        &self,
        rows: &mut Vec<UserRow>,
        ids: &IdCol,
        n: usize,
        row: impl Fn(usize) -> (i32, i64, i64, i64, u8),
    ) {
        let mut at = 0;
        let mut i = 0;
        for &count in &ids.runs {
            let len = u32::from_le_bytes(ids.dict[at..at + 4].try_into().unwrap()) as usize;
            for _ in 0..count {
                let (day, sessions, events, revenue, top) = row(i);
                rows.push(UserRow {
                    user_id: ids.dict[at + 4..at + 4 + len].to_vec(),
                    day,
                    sessions,
                    events,
                    revenue,
                    top: (top != 0xFF).then(|| self.cats[top as usize].clone()),
                });
                i += 1;
            }
            at += 4 + len;
        }
        assert_eq!(i, n);
    }

    fn flush_compact(&mut self, buying: bool) {
        let buf = if buying { &self.buying } else { &self.quiet };
        let n = buf.day.len();
        if let Some(collect) = self.collect {
            let mut rows = collect.lock().unwrap();
            self.collect_rows(&mut rows, &buf.ids, n, |i| {
                (
                    self.ws_day + i32::from(buf.day[i]),
                    i64::from(buf.sessions[i]),
                    i64::from(buf.events[i]),
                    if buying { i64::from(buf.revenue[i]) } else { 0 },
                    if buying { buf.top[i] } else { 0xFF },
                )
            });
        } else if let Some(dir) = &self.dir {
            let [day, sessions, events, revenue, top, idx] = &mut self.scratch;
            for page in [&mut *day, &mut *sessions, &mut *events, &mut *revenue, &mut *top, &mut *idx] {
                page.clear();
            }
            buf.ids.encode(idx);
            day.push(self.day_width as u8);
            bitpack_run(buf.day.iter().map(|&v| u32::from(v)), n, self.day_width, day);
            sessions.push(4);
            bitpack_run(buf.sessions.iter().map(|&v| u32::from(v)), n, 4, sessions);
            events.push(8);
            bitpack_run(buf.events.iter().map(|&v| u32::from(v)), n, 8, events);
            let zero = [0i32];
            let (revenue_pages, top_pages) = if buying {
                // Every row has a category: one run of definition level 1.
                top.extend_from_slice(&[0; 4]);
                rle_run(n, 1, 1, top);
                let levels = (top.len() - 4) as u32;
                top[..4].copy_from_slice(&levels.to_le_bytes());
                top.push(self.cat_width as u8);
                bitpack_run(buf.top.iter().map(|&v| u32::from(v)), n, self.cat_width, top);
                (
                    vec![Page::plain(n, vec![as_bytes(&buf.revenue)])],
                    vec![Page::dict_page(self.cats.len(), vec![&self.cat_dict]), Page::dict_indices(n, vec![&top[..]])],
                )
            } else {
                // Revenue is 0 everywhere: a one-entry dictionary and one run.
                revenue.push(1);
                rle_run(n, 0, 1, revenue);
                // top_category is null everywhere: one run of level 0.
                top.extend_from_slice(&[0; 4]);
                rle_run(n, 0, 1, top);
                let levels = (top.len() - 4) as u32;
                top[..4].copy_from_slice(&levels.to_le_bytes());
                (
                    vec![Page::dict_page(1, vec![as_bytes(&zero)]), Page::dict_indices(n, vec![&revenue[..]])],
                    vec![Page::plain(n, vec![&top[..]])],
                )
            };
            let thread = self.thread;
            let file = self
                .file
                .get_or_insert_with(|| PqFile::create(&dir.join(format!("part-{thread}.parquet")), &USER_SCHEMA_COMPACT));
            file.write_row_group(
                n,
                &[
                    vec![Page::dict_page(buf.ids.runs.len(), vec![&buf.ids.dict]), Page::dict_indices(n, vec![&idx[..]])],
                    vec![Page::dict_page(self.day_dict.len(), vec![as_bytes(&self.day_dict)]), Page::dict_indices(n, vec![&day[..]])],
                    vec![
                        Page::dict_page(MAX_SESSIONS as usize, vec![as_bytes(&self.count_dict[..MAX_SESSIONS as usize])]),
                        Page::dict_indices(n, vec![&sessions[..]]),
                    ],
                    vec![Page::dict_page(MAX_EVENTS as usize, vec![as_bytes(&self.count_dict)]), Page::dict_indices(n, vec![&events[..]])],
                    revenue_pages,
                    top_pages,
                ],
            );
        }
        self.rows_written += n as u64;
        let buf = if buying { &mut self.buying } else { &mut self.quiet };
        buf.ids.clear();
        buf.day.clear();
        buf.sessions.clear();
        buf.events.clear();
        buf.revenue.clear();
        buf.top.clear();
    }

    fn flush_wide(&mut self) {
        let buf = &self.wide;
        let n = buf.day.len();
        if let Some(collect) = self.collect {
            let mut rows = collect.lock().unwrap();
            self.collect_rows(&mut rows, &buf.ids, n, |i| {
                (buf.day[i], i64::from(buf.sessions[i]), i64::from(buf.events[i]), buf.revenue[i], buf.top[i])
            });
        } else if let Some(dir) = &self.dir {
            let mut idx = Vec::new();
            buf.ids.encode(&mut idx);
            let mut top = Vec::with_capacity(n / 4 + 64);
            def_levels(buf.top.iter().map(|&t| t != 0xFF), n, &mut top);
            let present = buf.top.iter().filter(|&&t| t != 0xFF).count();
            top.push(self.cat_width as u8);
            bitpack_run(buf.top.iter().filter(|&&t| t != 0xFF).map(|&t| u32::from(t)), present, self.cat_width, &mut top);
            // Same file and schema as the compact row groups when revenue
            // fits 32 bits (it practically always does), so every file of a
            // run has one schema. Only a row group that needs 64 bits goes
            // to the second file.
            let thread = self.thread;
            let narrow: Vec<i32>;
            let (file, revenue) = if buf.revenue.iter().all(|&r| r <= i64::from(i32::MAX)) {
                narrow = buf.revenue.iter().map(|&r| r as i32).collect();
                let file = self.file.get_or_insert_with(|| {
                    PqFile::create(&dir.join(format!("part-{thread}.parquet")), &USER_SCHEMA_COMPACT)
                });
                (file, as_bytes(&narrow))
            } else {
                let file = self.wide_file.get_or_insert_with(|| {
                    PqFile::create(&dir.join(format!("part-{thread}-wide.parquet")), &USER_SCHEMA)
                });
                (file, as_bytes(&buf.revenue))
            };
            file.write_row_group(
                n,
                &[
                    vec![Page::dict_page(buf.ids.runs.len(), vec![&buf.ids.dict]), Page::dict_indices(n, vec![&idx])],
                    vec![Page::plain(n, vec![as_bytes(&buf.day)])],
                    vec![Page::plain(n, vec![as_bytes(&buf.sessions)])],
                    vec![Page::plain(n, vec![as_bytes(&buf.events)])],
                    vec![Page::plain(n, vec![revenue])],
                    vec![Page::dict_page(self.cats.len(), vec![&self.cat_dict]), Page::dict_indices(n, vec![&top])],
                ],
            );
        }
        self.rows_written += n as u64;
        let buf = &mut self.wide;
        buf.ids.clear();
        buf.day.clear();
        buf.sessions.clear();
        buf.events.clear();
        buf.revenue.clear();
        buf.top.clear();
    }
}
