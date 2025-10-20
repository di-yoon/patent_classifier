import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F
from trl import SFTTrainer, SFTConfig
import pickle

class FineTuningTrainer:
    def __init__(self, model_name=" ", hf_token=None):
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.labels_list = None
        self.label2id = None
        self.id2label = None
        self.peft_config = None

    # Tokenizer 초기화
    def initialize_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

    # 모델 + LoRA + 양자화 설정
    def setup_model(self, bnb_config_params=None, lora_config_params=None):
        # 기본 BNB 설정
        if bnb_config_params is None:
            bnb_config_params = {
                'load_in_4bit': True,
                'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_compute_dtype': 'float16',
                'bnb_4bit_use_double_quant': True
            }

        bnb_config = BitsAndBytesConfig(**bnb_config_params)

        # 모델 로드
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            token=self.hf_token,
            num_labels=len(self.labels_list),
            device_map='auto',
            quantization_config=bnb_config
        )

        self.model.config.use_cache = False
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # 기본 LoRA 설정
        if lora_config_params is None:
            lora_config_params = {
                'lora_alpha': 128,
                'lora_dropout': 0.1,
                'r': 64,
                'bias': 'none',
                'task_type': 'SEQ_CLS',
                'target_modules': [
                    'k_proj', 'gate_proj', 'v_proj',
                    'up_proj', 'q_proj', 'o_proj', 'down_proj'
                ]
            }

        # LoRA 구성 저장
        self.peft_config = LoraConfig(**lora_config_params)

        # 모델에 LoRA 적용
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.peft_config)

    # 평가 지표 계산
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)

        logits_tensor = torch.tensor(pred.predictions)
        labels_tensor = torch.tensor(pred.label_ids)
        loss = F.cross_entropy(logits_tensor, labels_tensor).item()

        return {
            'eval_accuracy': acc,
            'eval_f1': f1,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_loss': loss
        }

    # 학습 실행
    def train_model(self, tokenized_dataset, output_dir, training_config_params=None):
        tokenized_train = tokenized_dataset['train']
        tokenized_test = tokenized_dataset['test']

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # 기본 Trainer args
        default_training_args = {
            'output_dir': output_dir,
            'learning_rate': 2e-5,
            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 2,
            'gradient_accumulation_steps': 2,
            'optim': 'paged_adamw_32bit',
            'lr_scheduler_type': 'cosine',
            'num_train_epochs': 5,
            'warmup_steps': 50,
            'logging_steps': 10,
            'fp16': True,
            'gradient_checkpointing': True,
            'dataset_text_field': 'text',
            'max_length': 512,
            'label_names': ['labels']
        }

        # UI에서 받은 값이 있으면 덮어쓰기
        if training_config_params:
            default_training_args.update(training_config_params)

        training_arguments = SFTConfig(**default_training_args)

        # Trainer 생성 (LoRA 설정은 self.peft_config 사용)
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            processing_class=self.tokenizer,
            args=training_arguments,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            peft_config=self.peft_config
        )

        # 학습 시작
        self.trainer.train()

        # 라벨 매핑 저장
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'label_mappings.pkl'), 'wb') as f:
            pickle.dump({
                'labels_list': self.labels_list,
                'label2id': self.label2id,
                'id2label': self.id2label
            }, f)

        return self.trainer.evaluate()

    # 모델 저장
    def save_model(self, output_dir, adapter_subdir="adapter"):
        if not self.trainer:
            raise RuntimeError("Trainer가 없습니다. 학습 후에 저장하세요.")

        os.makedirs(output_dir, exist_ok=True)

        # 토크나이저 저장
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

        # 어댑터만 저장
        adapter_dir = os.path.join(output_dir, adapter_subdir)
        os.makedirs(adapter_dir, exist_ok=True)
        self.trainer.model.save_pretrained(adapter_dir)
        print(f" Adapter saved to: {adapter_dir}")
