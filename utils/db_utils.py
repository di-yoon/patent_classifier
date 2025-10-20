# utils/db_utils.py
import sqlite3, json
from methods.db_viewer.schema import TRAIN_DB_PATH, INFER_DB_PATH, PROMPT_DB_PATH

# train
def init_train_db():
    conn = sqlite3.connect(TRAIN_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS train_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        text_col TEXT,
        label_col TEXT,
        labels TEXT,
        num_labels INTEGER,
        bnb_config TEXT,
        lora_config TEXT,
        training_config TEXT,
        accuracy REAL,
        f1 REAL,
        precision REAL,
        recall REAL,
        loss REAL,
        output_dir TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS test_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        출원번호 TEXT,
        text TEXT,
        label TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()


def save_train_results(eval_results, test_df,
                       model_name, text_col, label_col,
                       labels_list, bnb_config_params,
                       lora_config_params, training_config_params,
                       output_dir):
    conn = sqlite3.connect(TRAIN_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO train_metrics (
        accuracy, f1, precision, recall, loss,
        model_name, text_col, label_col, labels,
        num_labels, output_dir, bnb_config, lora_config, training_config
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        eval_results['eval_accuracy'],
        eval_results['eval_f1'],
        eval_results['eval_precision'],
        eval_results['eval_recall'],
        eval_results['eval_loss'],
        model_name,
        text_col,
        label_col,
        ",".join(labels_list),
        len(labels_list),
        output_dir,
        json.dumps(bnb_config_params),
        json.dumps(lora_config_params),
        json.dumps(training_config_params)
    ))
    conn.commit()
    conn.close()


# inference
def init_infer_db():
    conn = sqlite3.connect(INFER_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inference_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        selected_col TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inference_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        출원번호 TEXT,
        예측_라벨 TEXT,
        신뢰도 REAL,
        FOREIGN KEY(run_id) REFERENCES inference_runs(id)
    )
    """)
    conn.commit()
    conn.close()


def save_inference_results(model_name, selected_col, results_df):
    conn = sqlite3.connect(INFER_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO inference_runs (model_name, selected_col) VALUES (?, ?)",
        (model_name, selected_col)
    )
    run_id = cursor.lastrowid
    if not results_df.empty:
        df = results_df.copy()
        df["run_id"] = int(run_id)
        df.to_sql("inference_results", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


# prompt
def init_prompt_db():
    conn = sqlite3.connect(PROMPT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prompt_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        num_categories INTEGER,
        prompt_template TEXT,
        categories TEXT,  
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prompt_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        text TEXT,
        classification TEXT,
        FOREIGN KEY(run_id) REFERENCES prompt_runs(id)
    )
    """)
    conn.commit()
    conn.close()


def save_prompt_results(num_categories, prompt_template, categories, results_df):  # ✅ categories 인자 추가
    conn = sqlite3.connect(PROMPT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO prompt_runs (num_categories, prompt_template, categories) VALUES (?, ?, ?)",
        (num_categories, prompt_template, json.dumps(categories))  # ✅ categories 저장
    )
    run_id = cursor.lastrowid
    if not results_df.empty:
        df = results_df.copy()
        df["run_id"] = run_id
        df.to_sql("prompt_results", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    return run_id
