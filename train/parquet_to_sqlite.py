# save as: parquet_to_sqlite.py
import sqlite3, json
import pandas as pd
import pyarrow.parquet as pq
from typing import Any

def _to_json_if_nested(x: Any):
    # dict/list/tuple 등은 JSON 문자열로 직렬화
    if isinstance(x, (dict, list, tuple)):
        return json.dumps(x, ensure_ascii=False)
    return x

def parquet_to_sqlite(parquet_path: str, sqlite_path: str, table_name: str, chunksize: int = 200_000):
    pf = pq.ParquetFile(parquet_path)
    con = sqlite3.connect(sqlite_path)
    first = True
    try:
        for rg in range(pf.num_row_groups):
            batch = pf.read_row_group(rg).to_pandas(types_mapper=None)  # Arrow→Pandas
            # object 컬럼들 중 중첩값 존재 시 JSON 직렬화
            for col in batch.columns:
                if batch[col].dtype == "object":
                    # 값 중 dict/list/tuple가 있으면 JSON으로 변환
                    if batch[col].map(lambda v: isinstance(v, (dict, list, tuple))).any():
                        batch[col] = batch[col].map(_to_json_if_nested)

            batch.to_sql(
                table_name, con,
                if_exists='replace' if first else 'append',
                index=False,
                chunksize=50_000  # SQLite insert 배치
            )
            first = False
            print(f"[OK] wrote row-group {rg+1}/{pf.num_row_groups}")
    finally:
        con.close()
    print(f"Done → {sqlite_path} [{table_name}]")

if __name__ == "__main__":
    parquet_to_sqlite(
        parquet_path="/workspace/SchemaReinforcementLearning/train/data/train_with_tool_ToS.parquet",
        sqlite_path="data.db",
        table_name="train_with_tool_ToS",
    )
