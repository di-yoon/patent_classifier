# methods/db_viewer/schema.py

TRAIN_DB_PATH = "train_results.db"
INFER_DB_PATH = "inference_results.db"
PROMPT_DB_PATH = "prompt_results.db"

TABLES = {
    "train_metrics": "학습 기록",
    "inference_runs": "추론 실행 기록",
    "inference_results": "추론 결과",
    "prompt_runs": "프롬프트 실행 기록",
    "prompt_results": "프롬프트 결과"
}
