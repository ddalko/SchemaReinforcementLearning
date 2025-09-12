#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a Datasette-ready SQLite from Parquet/JSONL/JSON files.

- Decodes binary columns (bytes) to text (UTF-8; gzip auto-detect; latin-1 fallback)
- Normalizes double-escaped JSON strings into valid JSON text
- Converts dict/list values to JSON strings
- Streams Parquet by row-group and JSONL by chunks to handle large files

Usage examples:

  # Example for your files:
  python build_datasette_db.py \
    --sqlite data.db \
    --parquet train_with_tool_ToS.parquet=train_with_tool_ToS \
    --jsonl test/custom_append.jsonl=custom_append \
    --jsonl test/translation_test.jsonl=translation_test \
    --json mix_train.json=mix_train

Requirements:
  pip install pandas pyarrow
"""
import argparse
import gzip
import io
import json
import os
import sqlite3
from typing import Any, Iterable, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq


def is_byteslike(x: Any) -> bool:
    return isinstance(x, (bytes, bytearray, memoryview))


def try_gunzip(data: bytes) -> bytes:
    # Detect gzip by magic number (1f 8b)
    if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
        try:
            return gzip.decompress(data)
        except Exception:
            pass
    return data


def safe_bytes_to_text(x: Any) -> Any:
    """Decode bytes/bytearray/memoryview to str. Preserve other types."""
    if not is_byteslike(x):
        return x
    b = bytes(x)
    b = try_gunzip(b)
    # Try UTF-8 first, fallback to latin-1 with replacement
    try:
        return b.decode("utf-8")
    except Exception:
        try:
            return b.decode("latin-1", errors="replace")
        except Exception:
            # last resort
            return repr(b)


def _loads_json_maybe(s: str) -> Tuple[bool, Optional[Any]]:
    """Try json.loads; return (ok, value)."""
    try:
        return True, json.loads(s)
    except Exception:
        return False, None


def normalize_jsonish_text(s: Any) -> Any:
    """
    Normalize values that are (or contain) JSON into a valid JSON string.
    - If s is dict/list -> json.dumps
    - If s is string with double-escaped JSON -> unescape once/twice
    - Otherwise return s as-is
    """
    # Convert bytes -> text first
    s = safe_bytes_to_text(s)

    # If native JSON-serializable types: dict/list/tuple -> dump to JSON string
    if isinstance(s, (dict, list, tuple)):
        try:
            return json.dumps(s, ensure_ascii=False)
        except Exception:
            return json.dumps(str(s), ensure_ascii=False)

    if not isinstance(s, str):
        return s

    st = s.strip()
    # Already looks like JSON object/array
    if st.startswith("{") or st.startswith("["):
        ok, val = _loads_json_maybe(st)
        if ok and isinstance(val, (dict, list)):
            # Return canonicalized JSON string (compact; pretty-print is UI concern)
            return json.dumps(val, ensure_ascii=False)
        # If it starts like JSON but not valid - leave as-is (could be partial text)
        return s

    # Sometimes entire JSON is stored as a JSON-encoded string:
    #     "{\"a\": 1}"  -> loads -> dict
    # or it may be nested twice
    ok, val = _loads_json_maybe(s)
    if ok:
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        if isinstance(val, str):
            st2 = val.strip()
            if st2.startswith("{") or st2.startswith("["):
                ok2, val2 = _loads_json_maybe(st2)
                if ok2 and isinstance(val2, (dict, list)):
                    return json.dumps(val2, ensure_ascii=False)
    return s


def clean_object_columns(df: pd.DataFrame, hint_json_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    For object-typed columns, decode bytes -> text and normalize JSON-like values.
    If hint_json_cols provided, apply normalization aggressively to those columns.
    """
    hint_json_cols = set(hint_json_cols or [])

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            if col in hint_json_cols:
                df[col] = df[col].map(normalize_jsonish_text)
            else:
                # lightweight path: only touch rows that are likely bytes/dicts/lists or clearly JSON-ish
                def _maybe_fix(v: Any) -> Any:
                    if is_byteslike(v) or isinstance(v, (dict, list, tuple)):
                        return normalize_jsonish_text(v)
                    if isinstance(v, str):
                        sv = v.strip()
                        if sv.startswith("{") or sv.startswith("[") or "\\\"" in sv or "\\n" in sv:
                            return normalize_jsonish_text(v)
                    return v
                df[col] = df[col].map(_maybe_fix)
    return df


def connect_sqlite(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    # performance pragmas
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA temp_store=MEMORY;")
    con.execute("PRAGMA page_size=32768;")
    return con


def write_df(con: sqlite3.Connection, table: str, df: pd.DataFrame, first: bool) -> None:
    df.to_sql(table, con, if_exists="replace" if first else "append", index=False, chunksize=50_000)


def import_parquet(parquet_path: str, con: sqlite3.Connection, table: str, hint_json_cols: Optional[Iterable[str]] = None) -> None:
    pf = pq.ParquetFile(parquet_path)
    first = True
    for rg in range(pf.num_row_groups):
        batch = pf.read_row_group(rg).to_pandas()
        batch = clean_object_columns(batch, hint_json_cols=hint_json_cols)
        write_df(con, table, batch, first=first)
        first = False
        print(f"[Parquet] wrote row-group {rg+1}/{pf.num_row_groups} -> {table}")


def import_jsonl(jsonl_path: str, con: sqlite3.Connection, table: str, hint_json_cols: Optional[Iterable[str]] = None, chunksize: int = 200_000) -> None:
    first = True
    for chunk in pd.read_json(jsonl_path, lines=True, chunksize=chunksize):
        chunk = clean_object_columns(chunk, hint_json_cols=hint_json_cols)
        write_df(con, table, chunk, first=first)
        first = False
        print(f"[JSONL] wrote chunk -> {table}")


def import_json_array(json_path: str, con: sqlite3.Connection, table: str, hint_json_cols: Optional[Iterable[str]] = None) -> None:
    # Assumes the JSON file is a list of objects
    df = pd.read_json(json_path)
    df = clean_object_columns(df, hint_json_cols=hint_json_cols)
    write_df(con, table, df, first=True)
    print(f"[JSON] wrote array -> {table}")


def parse_kv(arg_list: Iterable[str]) -> Iterable[Tuple[str, str]]:
    """
    Parse entries like 'path=table' into (path, table) tuples.
    """
    for item in arg_list:
        if "=" not in item:
            raise ValueError(f"Expected 'path=table' format, got: {item}")
        path, table = item.split("=", 1)
        yield path, table


def main():
    ap = argparse.ArgumentParser(description="Build SQLite for Datasette from Parquet/JSONL/JSON with JSON/binary normalization.")
    ap.add_argument("--sqlite", required=True, help="Output SQLite path, e.g., data.db")

    ap.add_argument("--parquet", action="append", default=[], help="Parquet spec 'path=table' (can repeat)")
    ap.add_argument("--jsonl", action="append", default=[], help="JSONL spec 'path=table' (can repeat)")
    ap.add_argument("--json", action="append", default=[], help="JSON array spec 'path=table' (can repeat)")

    ap.add_argument("--json-cols", action="append", default=[], help="Column names to aggressively normalize as JSON (can repeat)")

    args = ap.parse_args()

    con = connect_sqlite(args.sqlite)
    try:
        # Import Parquet
        for path, table in parse_kv(args.parquet):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            import_parquet(path, con, table, hint_json_cols=args.json_cols)

        # Import JSONL
        for path, table in parse_kv(args.jsonl):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            import_jsonl(path, con, table, hint_json_cols=args.json_cols)

        # Import JSON array
        for path, table in parse_kv(args.json):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            import_json_array(path, con, table, hint_json_cols=args.json_cols)

        print(f"Done. SQLite at: {args.sqlite}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
