import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, DataCollatorWithPadding
from peft import PeftModel
import pickle


# .env 파일 로드


class FineTuningInference:
    def __init__(self, model_name="google/gemma-2-2b", hf_token=None):
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.tokenizer = None
        self.model = None
        self.labels_list = None
        self.label2id = None
        self.id2label = None
    def load_model(self, model_path, manual_labels=None, is_merged_model=False):


        if not model_path or not os.path.exists(model_path):
            raise ValueError(model_path)

        # 병합된 모델 경로 확인
        merged_model_path = os.path.join(model_path, "merged_model")
        if not is_merged_model and os.path.exists(merged_model_path):
            model_path = merged_model_path
            is_merged_model = True

        label_file = os.path.join(model_path, 'label_mappings.pkl')

        if os.path.exists(label_file):
            try:
                with open(label_file, 'rb') as f:
                    mappings = pickle.load(f)
                    self.labels_list = mappings['labels_list']
                    self.label2id = mappings['label2id']
                    self.id2label = mappings['id2label']
            except Exception as e:
                raise ValueError(e)
        elif manual_labels:
            self.labels_list = sorted(manual_labels)
            self.label2id = {l: i for i, l in enumerate(self.labels_list)}
            self.id2label = {i: l for l, i in self.label2id.items()}
        else:
            self.labels_list = ['CPC_C01B', 'CPC_C01C', 'CPC_C01D', 'CPC_C01F', 'CPC_C01G']
            self.label2id = {l: i for i, l in enumerate(self.labels_list)}
            self.id2label = {i: l for l, i in self.label2id.items()}

        # 토크나이저 로드
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                token=self.hf_token,
                trust_remote_code=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

        try:
            if is_merged_model:
                # 병합된 모델 직접 로드
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    token=self.hf_token,
                    num_labels=len(self.labels_list),
                    device_map='auto',
                    trust_remote_code=True,
                    torch_dtype="auto"
                )
                print("병합된 모델을 로드했습니다.")
            else:
                # 기존 방식 (베이스 모델 + 어댑터)
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype='float16',
                    bnb_4bit_use_double_quant=True
                )

                # 베이스 모델을 SEQ_CLS로 로드
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    token=self.hf_token,
                    num_labels=len(self.labels_list),
                    device_map='auto',
                    quantization_config=bnb_config,
                    trust_remote_code=True
                )

                # 어댑터 로드 및 병합
                self.model = PeftModel.from_pretrained(
                    base_model,
                    model_path,
                    device_map='auto'
                )

                # 병합
                self.model = self.model.merge_and_unload()
                print("어댑터를 병합하여 로드했습니다.")

            self.model.eval()

        except Exception as e:
            raise ValueError(e)

    def predict_patents(self, df, model_path=None, selected_cols=None, max_length=512):
        try:
            if model_path and not self.model:
                self.load_model(model_path)

            if not self.model or not self.tokenizer:
                raise ValueError("모델이 로드되지 않았습니다.")

            #추론 컬럼 설정
            if selected_cols is None:
                selected_cols = ["대표청구항"]


            if isinstance(selected_cols, list):
                if len(selected_cols) == 1:
                    text_series = df[selected_cols[0]].astype(str)
                else:
                    text_series = df[selected_cols].astype(str).agg(" ".join, axis=1)
            else:
                text_series = df[selected_cols].astype(str)

            processed_df = pd.DataFrame({
                "text": text_series,
                "patent_id": df["출원번호"] if "출원번호" in df.columns else range(len(df))
            })

        except Exception as e:
            import traceback
            print(e)
            traceback.print_exc()
            raise

        try:
            test_data = Dataset.from_pandas(processed_df)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'right'

            if getattr(self.model.config, 'pad_token_id', None) is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            # tokenization
            def preprocess_function(examples):
                tokenized = self.tokenizer(
                    examples['text'],
                    truncation=True,
                    max_length=max_length,
                    padding=True
                )
                return tokenized

            tokenized_test = test_data.map(preprocess_function, batched=True)

            remove_cols = ['text', 'patent_id']
            if 'label' in processed_df.columns:
                remove_cols.append('label')
            tokenized_test = tokenized_test.remove_columns(remove_cols)

            # DataLoader
            from torch.utils.data import DataLoader

            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
            dataloader = DataLoader(tokenized_test, batch_size=2, collate_fn=data_collator)

            # 추론
            self.model.eval()
            all_predictions = []

            with torch.no_grad():
                for batch in dataloader:
                    if next(self.model.parameters()).device.type == 'cuda':
                        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    outputs = self.model(**batch)
                    logits = outputs.logits
                    predictions = torch.softmax(logits, dim=-1)
                    all_predictions.append(predictions.cpu())

            probs = torch.cat(all_predictions, dim=0).numpy()

        except Exception as e:
            import traceback
            print(e)
            traceback.print_exc()
            raise

        try:
            # 결과 처리
            patent_results = []
            for i, row in processed_df.iterrows():
                pred_idx = probs[i].argmax()
                pred_label = self.id2label[pred_idx]
                confidence = round(probs[i][pred_idx], 4)

                patent_results.append({
                    "출원번호": row["patent_id"],
                    "예측_라벨": pred_label,
                    "신뢰도": confidence
                })

            return pd.DataFrame(patent_results)

        except Exception as e:
            import traceback
            print(e)
            traceback.print_exc()
            raise
