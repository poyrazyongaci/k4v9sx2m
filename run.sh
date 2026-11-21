#!/bin/sh
# Entry point. Runs with no arguments; exit 0 means "done, check my output".
set -eu
: "${DATA_DIR:=/data}" "${OUTPUT_DIR:=/output}" "${SCRATCH_DIR:=/scratch}" "${NCPU:=8}"
export DATA_DIR OUTPUT_DIR SCRATCH_DIR
export TMPDIR="$SCRATCH_DIR" HOME="$SCRATCH_DIR" ORACLE_WORKERS="$NCPU"
exec python3 /app/pipeline/oracle.py
