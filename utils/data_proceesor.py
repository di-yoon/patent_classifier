# utils/data_proceesor.py
from datasets import Dataset, DatasetDict
import pandas as pd

class DataProcessor:
    @staticmethod
    def validate_dataframe(df, required_cols=None):
        if df is None or len(df) == 0:
            raise ValueError("데이터가 비어 있습니다.")
        if required_cols:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(missing_cols)
        return True

    @staticmethod
    def prepare_data(executor, df, text_col="전체청구항", label_col="사용자태그"):
        """단일 텍스트 칼럼 + 라벨 칼럼 기반 전처리"""
        df_copy = df.copy()

        if text_col not in df_copy.columns:
            raise ValueError(f"선택한 텍스트 칼럼({text_col})이 데이터에 없습니다.")
        if label_col not in df_copy.columns:
            raise ValueError(f"선택한 라벨 칼럼({label_col})이 데이터에 없습니다.")

        df_copy["text"] = df_copy[text_col].astype(str)

        # 라벨 매핑
        executor.labels_list = sorted(df_copy[label_col].dropna().unique())
        if len(executor.labels_list) < 2:
            raise ValueError("라벨 종류가 2개 미만입니다. 학습 불가합니다.")

        executor.label2id = {l: i for i, l in enumerate(executor.labels_list)}
        executor.id2label = {i: l for l, i in executor.label2id.items()}

        processed_df = pd.DataFrame({
            "text": df_copy[text_col].astype(str),
            "labels": df_copy[label_col],
            "patent_id": df_copy["출원번호"] if "출원번호" in df_copy.columns else range(len(df_copy))
        })
        processed_df["label"] = processed_df["labels"].map(executor.label2id)

        return processed_df

    @staticmethod
    def create_balanced_datasetdict(df, tokenizer, test_size=0.2, random_state=25):
        """라벨별 동일 개수로 train/test 분할"""
        label_counts = df['label'].value_counts()
        min_count = label_counts.min()

        if min_count < 2:
            raise ValueError("라벨 최소 샘플 수가 너무 적어 학습을 진행할 수 없습니다.")

        train_samples_per_label = int(min_count * (1 - test_size))
        test_samples_per_label = min_count - train_samples_per_label

        if train_samples_per_label == 0 or test_samples_per_label == 0:
            raise ValueError("Train/Test 데이터셋을 만들 수 없습니다. 데이터 수를 확인하세요.")

        train_dfs, test_dfs = [], []
        for label in sorted(df['label'].unique()):
            label_data = df[df['label'] == label].sample(n=min_count, random_state=random_state).reset_index(drop=True)
            train_data = label_data.iloc[:train_samples_per_label]
            test_data = label_data.iloc[train_samples_per_label:]
            train_dfs.append(train_data)
            test_dfs.append(test_data)

        train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=random_state)
        test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=random_state)

        # DatasetDict 생성
        train_data = Dataset.from_dict({'text': train_df['text'].tolist(), 'label': train_df['label'].tolist()})
        test_data = Dataset.from_dict({'text': test_df['text'].tolist(), 'label': test_df['label'].tolist()})
        dataset = DatasetDict({'train': train_data, 'test': test_data})

        # 토큰화
        def preprocess_function(examples):
            tokenized = tokenizer(examples['text'], truncation=True, max_length=512)
            tokenized['labels'] = [int(l) for l in examples['label']]
            return tokenized

        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label'])

        return tokenized_dataset, test_df
