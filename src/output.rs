//! The two output tables as Parquet, from plain rows. The fast engine
//! writes user_daily through its own column buffers (engine.rs) and uses
//! this module's schema; campaign_daily and the fallback path go through
//! the row writers here.

use std::path::Path;

use crate::common::*;
use crate::parquet::*;

pub const USER_SCHEMA: [ColSpec; 6] = [
    ColSpec { name: "user_id", phys: Phys::ByteArray, logical: Logical::Utf8, optional: false },
    ColSpec { name: "day", phys: Phys::Int32, logical: Logical::Date, optional: false },
    ColSpec { name: "sessions", phys: Phys::Int32, logical: Logical::None, optional: false },
    ColSpec { name: "events", phys: Phys::Int32, logical: Logical::None, optional: false },
    ColSpec { name: "revenue_cents", phys: Phys::Int64, logical: Logical::None, optional: false },
    ColSpec { name: "top_category", phys: Phys::ByteArray, logical: Logical::Utf8, optional: true },
];

pub const CAMP_SCHEMA: [ColSpec; 7] = [
    ColSpec { name: "campaign_id", phys: Phys::ByteArray, logical: Logical::Utf8, optional: false },
    ColSpec { name: "channel", phys: Phys::ByteArray, logical: Logical::Utf8, optional: false },
    ColSpec { name: "day", phys: Phys::Int32, logical: Logical::Date, optional: false },
    ColSpec { name: "sessions", phys: Phys::Int64, logical: Logical::None, optional: false },
    ColSpec { name: "purchasing_sessions", phys: Phys::Int64, logical: Logical::None, optional: false },
    ColSpec { name: "revenue_cents", phys: Phys::Int64, logical: Logical::None, optional: false },
    ColSpec { name: "assisted_sessions", phys: Phys::Int64, logical: Logical::None, optional: false },
];

/// The compact user_daily schema: 32-bit revenue. Rows that do not fit go
/// to a second file with `USER_SCHEMA`.
pub const USER_SCHEMA_COMPACT: [ColSpec; 6] = [
    USER_SCHEMA[0],
    USER_SCHEMA[1],
    USER_SCHEMA[2],
    USER_SCHEMA[3],
    ColSpec { name: "revenue_cents", phys: Phys::Int32, logical: Logical::None, optional: false },
    USER_SCHEMA[5],
];

/// Wide (64-bit counts) user_daily schema for the fallback path.
const USER_SCHEMA_WIDE: [ColSpec; 6] = [
    USER_SCHEMA[0],
    USER_SCHEMA[1],
    ColSpec { name: "sessions", phys: Phys::Int64, logical: Logical::None, optional: false },
    ColSpec { name: "events", phys: Phys::Int64, logical: Logical::None, optional: false },
    USER_SCHEMA[4],
    USER_SCHEMA[5],
];

pub fn make_dirs(out: &Path) {
    for sub in ["user_daily", "campaign_daily"] {
        if let Err(e) = std::fs::create_dir_all(out.join(sub)) {
            die(&format!("{}: {e}", out.join(sub).display()));
        }
    }
}

pub fn write_user_rows(path: &Path, rows: &[UserRow]) {
    let mut ids = Vec::new();
    let mut tops = Vec::new();
    for row in rows {
        plain_bytes(&row.user_id, &mut ids);
    }
    def_levels(rows.iter().map(|r| r.top.is_some()), rows.len(), &mut tops);
    for row in rows {
        if let Some(top) = &row.top {
            plain_bytes(top, &mut tops);
        }
    }
    let days: Vec<i32> = rows.iter().map(|r| r.day).collect();
    let sessions: Vec<i64> = rows.iter().map(|r| r.sessions).collect();
    let events: Vec<i64> = rows.iter().map(|r| r.events).collect();
    let revenue: Vec<i64> = rows.iter().map(|r| r.revenue).collect();
    let n = rows.len();
    let mut file = PqFile::create(path, &USER_SCHEMA_WIDE);
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

pub fn write_camp_rows(path: &Path, rows: &[CampRow]) {
    let mut ids = Vec::new();
    let mut channels = Vec::new();
    for row in rows {
        plain_bytes(&row.campaign, &mut ids);
        plain_bytes(&row.channel, &mut channels);
    }
    let days: Vec<i32> = rows.iter().map(|r| r.day).collect();
    let sessions: Vec<i64> = rows.iter().map(|r| r.sessions).collect();
    let purchasing: Vec<i64> = rows.iter().map(|r| r.purchasing).collect();
    let revenue: Vec<i64> = rows.iter().map(|r| r.revenue).collect();
    let assisted: Vec<i64> = rows.iter().map(|r| r.assisted).collect();
    let n = rows.len();
    let mut file = PqFile::create(path, &CAMP_SCHEMA);
    file.write_row_group(
        n,
        &[
            vec![Page::plain(n, vec![&ids])],
            vec![Page::plain(n, vec![&channels])],
            vec![Page::plain(n, vec![as_bytes(&days)])],
            vec![Page::plain(n, vec![as_bytes(&sessions)])],
            vec![Page::plain(n, vec![as_bytes(&purchasing)])],
            vec![Page::plain(n, vec![as_bytes(&revenue)])],
            vec![Page::plain(n, vec![as_bytes(&assisted)])],
        ],
    );
    file.finish();
}

pub fn write_tables(out: &Path, tables: &Tables) {
    make_dirs(out);
    write_user_rows(&out.join("user_daily/part-0.parquet"), &tables.user);
    write_camp_rows(&out.join("campaign_daily/part-0.parquet"), &tables.camp);
}
