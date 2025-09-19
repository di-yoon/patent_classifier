# methods/db_viewer/handler.py
import sqlite3
import pandas as pd
from .schema import TRAIN_DB_PATH, INFER_DB_PATH, PROMPT_DB_PATH

def _get_db_path(table_name: str) -> str:

    if table_name.startswith("train"):
        return TRAIN_DB_PATH
    elif table_name.startswith("inference"):
        return INFER_DB_PATH
    elif table_name.startswith("prompt"):
        return PROMPT_DB_PATH
    else:
        raise ValueError(f"알 수 없는 테이블명: {table_name}")

def load_table(table_name: str, limit: int = 200) -> pd.DataFrame:

    db_path = _get_db_path(table_name)

    # id 기준 정렬
    if table_name in ("inference_results", "prompt_results"):
        order_by = "id DESC"
    else:
        order_by = "timestamp DESC"

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            f"SELECT * FROM {table_name} ORDER BY {order_by} LIMIT {limit}", conn
        )
    except Exception as e:
        df = pd.DataFrame({"error": [str(e)]})
    finally:
        conn.close()
    return df

def delete_runs(table_name: str, run_ids: list[int]):

    if not run_ids:
        return

    db_path = _get_db_path(table_name)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    q_marks = ",".join("?" for _ in run_ids)
    cursor.execute(f"DELETE FROM {table_name} WHERE id IN ({q_marks})", run_ids)

    conn.commit()
    conn.close()
