import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
from peft import PeftModel
import pickle


class FineTuningInference:
    def __init__(self, model_name="google/gemma-2-2b", hf_token=None):
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.tokenizer = None
        self.model = None
        self.labels_list = None
        self.label2id = None
        self.id2label = None

    # -------------------------------
    # 라벨 매핑 로드
    # -------------------------------
    def _load_label_mappings(self, model_path, manual_labels=None):
        label_file = os.path.join(model_path, 'label_mappings.pkl')
        parent_label_file = os.path.join(os.path.dirname(model_path), 'label_mappings.pkl')

        pick_path = label_file if os.path.exists(label_file) else (
            parent_label_file if os.path.exists(parent_label_file) else None
        )

        if pick_path:
            with open(pick_path, 'rb') as f:
                mappings = pickle.load(f)
                self.labels_list = mappings.get('labels_list')
                self.label2id = mappings.get('label2id')
                self.id2label = mappings.get('id2label')
                return

        if manual_labels:
            self.labels_list = sorted(manual_labels)
        else:
            # fallback 라벨
            self.labels_list = ['CPC_C01B', 'CPC_C01C', 'CPC_C01D', 'CPC_C01F', 'CPC_C01G']

        self.label2id = {l: i for i, l in enumerate(self.labels_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}

    # -------------------------------
    # 모델 + 어댑터 로드
    # -------------------------------
    def load_model(self, model_path, manual_labels=None, quantize_base=False):
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        self._load_label_mappings(model_path, manual_labels)

        # 토크나이저 로드 (model_path → parent → base 순서)
        for path in [model_path, os.path.dirname(model_path), self.model_name]:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    path, use_auth_token=self.hf_token, trust_remote_code=True
                )
                break
            except Exception:
                continue
        if not self.tokenizer:
            raise ValueError("Tokenizer could not be loaded from any path.")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

        # base 모델 로드
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype='float16',
            bnb_4bit_use_double_quant=True
        ) if quantize_base else None

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_token,
            num_labels=len(self.labels_list),
            device_map='auto',
            trust_remote_code=True,
            quantization_config=bnb_config
        )

        # 어댑터 로드 + 병합
        peft_model = PeftModel.from_pretrained(base_model, model_path, device_map='auto')
        self.model = peft_model.merge_and_unload()
        self.model.eval()

    # -------------------------------
    # 추론 실행
    # -------------------------------
    def predict_patents(self, df, model_path=None, selected_cols=None, max_length=512, batch_size=8):
        if model_path and not self.model:
            self.load_model(model_path)

        if not self.model or not self.tokenizer:
            raise ValueError("Model or tokenizer not loaded. Please load adapter first.")

        if selected_cols is None:
            selected_cols = ["대표청구항"]

        # 텍스트 합치기
        if isinstance(selected_cols, list):
            text_series = (
                df[selected_cols[0]].astype(str)
                if len(selected_cols) == 1
                else df[selected_cols].astype(str).agg(" ".join, axis=1)
            )
        else:
            text_series = df[selected_cols].astype(str)

        processed_df = pd.DataFrame({
            "text": text_series,
            "patent_id": df["출원번호"] if "출원번호" in df.columns else list(range(len(df)))
        })

        # Dataset 생성 및 토크나이징
        test_data = Dataset.from_pandas(processed_df)

        if getattr(self.model.config, 'pad_token_id', None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        def preprocess_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_length,
                padding=True
            )

        tokenized_test = test_data.map(preprocess_function, batched=True)
        tokenized_test = tokenized_test.remove_columns(
            [c for c in ['text', 'patent_id'] if c in tokenized_test.column_names]
        )

        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            tokenized_test,
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
        )

        # 추론
        device = next(self.model.parameters()).device
        all_probs = []
        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu())

        probs = torch.cat(all_probs, dim=0).numpy()

        # 결과 정리
        patent_results = []
        for i, row in processed_df.iterrows():
            pred_idx = int(probs[i].argmax())
            pred_label = self.id2label.get(pred_idx, str(pred_idx))
            confidence = float(round(probs[i][pred_idx], 4))
            patent_results.append({
                "출원번호": row["patent_id"],
                "예측_라벨": pred_label,
                "신뢰도": confidence
            })

        return pd.DataFrame(patent_results)
