import os, streamlit as st
from utils.data_proceesor import DataProcessor
from methods.finetuning.trainer import FineTuningTrainer
from utils.db_utils import init_train_db, save_train_results

def show():
    with st.expander("**COLUMNS TO USE FOR TRAIN**", expanded=True):
        df = st.session_state.uploaded_df

        # 텍스트 칼럼 선택
        text_col = st.selectbox(
            " 학습에 사용할 텍스트 칼럼 선택",
            options=df.columns.tolist(),
            index=df.columns.get_loc("대표청구항") if "대표청구항" in df.columns else 0,
            key="train_text_col"
        )

        # 라벨 칼럼 선택
        label_col = st.selectbox(
            " 라벨(정답) 칼럼 선택",
            options=df.columns.tolist(),
            index=df.columns.get_loc("사용자태그") if "사용자태그" in df.columns else 0,
            key="train_label_col"
        )

        # 라벨 분포
        if label_col in df.columns:
            st.subheader("라벨 분포")
            st.bar_chart(df[label_col].value_counts())

        st.success(f" 텍스트 칼럼: {text_col} → 라벨 칼럼: {label_col} 기준으로 학습합니다.")

    # Hyperparameter Settings UI
    with st.expander("**HYPERPARAMETER SETTINGS**", expanded=True):
        # BNB Quantization
        with st.expander(" Quantization (bnb)", expanded=False):
            bnb_c1, bnb_c2, bnb_c3 = st.columns(3)
            with bnb_c1:
                bnb_4bit_quant_type = st.selectbox("quant_type", ["nf4", "fp4"], index=0)
            with bnb_c2:
                bnb_4bit_compute_dtype = st.selectbox("compute_dtype", ["float16", "bfloat16"], index=0)
            with bnb_c3:
                bnb_use_double_quant = st.checkbox("double_quant", value=True)

        # LoRA Config
        with st.expander(" LoRA Config", expanded=False):
            lora_c1, lora_c2, lora_c3 = st.columns(3)
            with lora_c1:
                lora_alpha = st.number_input("alpha", min_value=1, max_value=1024, value=128, step=1)
            with lora_c2:
                lora_dropout = st.number_input("dropout", min_value=0.0, max_value=1.0, value=0.1,
                                               step=0.05, format="%.2f")
            with lora_c3:
                lora_r = st.number_input("rank (r)", min_value=1, max_value=256, value=64, step=1)

            task_type = st.selectbox(
                "task_type",
                ["SEQ_CLS", "CAUSAL_LM", "TOKEN_CLS", "SEQ_2_SEQ_LM"],
                index=0
            )

            target_modules = st.multiselect(
                "target_modules",
                ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj'],
                default=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
            )

        # Trainer Config
        with st.expander("Trainer Config (SFT)", expanded=True):
            trn1, trn2, trn3, trn4 = st.columns(4)
            with trn1:
                num_epochs = st.number_input("epochs", min_value=1, max_value=50, value=5, step=1)
            with trn2:
                learning_rate = st.number_input("lr", min_value=1e-6, max_value=1e-2,
                                                value=2e-5, step=1e-6, format="%.6f")
            with trn3:
                train_batch = st.number_input("train_bs", min_value=1, max_value=64, value=2, step=1)
            with trn4:
                eval_batch = st.number_input("eval_bs", min_value=1, max_value=64, value=2, step=1)

            trn5, trn6, trn7, trn8 = st.columns(4)
            with trn5:
                grad_accum = st.number_input("grad_accum", min_value=1, max_value=128, value=2, step=1)
            with trn6:
                warmup_steps = st.number_input("warmup", min_value=0, max_value=100_000, value=50, step=10)
            with trn7:
                logging_steps = st.number_input("logging", min_value=1, max_value=10_000, value=10, step=1)
            with trn8:
                max_length = st.number_input("max_len", min_value=128, max_value=8192, value=512, step=64)

        # Output Directory
        with st.expander(" Output Directory", expanded=True):
            default_outdir = os.path.join(r"C:\company\wips", "ft_gemma_2")
            output_dir = st.text_input("output_dir", value=default_outdir)

    # Train Button
    if st.button("**T R A I N**", type="primary", width='stretch'):
        trainer = None
        eval_results = {}

        try:
            init_train_db()  # DB 초기화
            model_name = st.session_state.get('ft_model_name', 'google/gemma-2-2b')
            hf_token = st.session_state.get('ft_hf_token') or os.getenv('HF_TOKEN')
            trainer = FineTuningTrainer(model_name, hf_token)
            with st.spinner("INITIALIZING TOKENIZER ..."):
                trainer.initialize_tokenizer()
            # 데이터 전처리
            with st.spinner("PREPROCESSING DATA ..."):
                processed_df = DataProcessor.prepare_data(trainer, df, text_col=text_col, label_col=label_col)
                tokenized_dataset, test_df = DataProcessor.create_balanced_datasetdict(
                    processed_df, trainer.tokenizer, test_size=0.2
                )
            # 모델 설정
            with st.spinner("CONFIGURING MODEL ..."):
                bnb_config_params = {
                    'load_in_4bit': True,
                    'bnb_4bit_quant_type': bnb_4bit_quant_type,
                    'bnb_4bit_compute_dtype': bnb_4bit_compute_dtype,
                    'bnb_4bit_use_double_quant': bnb_use_double_quant
                }
                lora_config_params = {
                    'lora_alpha': lora_alpha,
                    'lora_dropout': lora_dropout,
                    'r': lora_r,
                    'bias': 'none',
                    'task_type': task_type,
                    'target_modules': target_modules
                }
                trainer.setup_model(bnb_config_params, lora_config_params)

            # 학습 실행
            with st.spinner("TRAINING MODEL ..."):
                os.makedirs(output_dir, exist_ok=True)
                training_config_params = {
                    'num_train_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'per_device_train_batch_size': train_batch,
                    'per_device_eval_batch_size': eval_batch,
                    'gradient_accumulation_steps': grad_accum,
                    'warmup_steps': warmup_steps,
                    'logging_steps': logging_steps,
                    'max_length': max_length
                }
                eval_results = trainer.train_model(
                    tokenized_dataset,
                    output_dir,
                    training_config_params
                )
                trainer.save_model(output_dir)

                # 테스트 데이터 저장
                try:
                    test_save_path = os.path.join(output_dir, "test_data.csv")
                    test_df.to_csv(test_save_path, index=False, encoding="utf-8-sig")
                    st.info(f"Test dataset saved to {test_save_path}")
                except Exception as e:
                    st.warning(f"Could not save test dataset: {e}")

            # DB 저장
            if trainer:
                save_train_results(
                    eval_results, test_df,
                    model_name=model_name,
                    text_col=text_col,
                    label_col=label_col,
                    labels_list=trainer.labels_list,
                    bnb_config_params=bnb_config_params,
                    lora_config_params=lora_config_params,
                    training_config_params=training_config_params,
                    output_dir=output_dir
                )
                st.success("학습 결과가 SQLite DB에 저장되었습니다.")

            st.toast("TRAIN IS COMPLETE ")

            # 결과 메트릭 표시
            st.subheader("TRAIN RESULT")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("**ACCURACY**", f"{eval_results.get('eval_accuracy', 0.0):.4f}")
            with col2:
                st.metric("**F1 SCORE**", f"{eval_results.get('eval_f1', 0.0):.4f}")
            with col3:
                st.metric("**PRECISION**", f"{eval_results.get('eval_precision', 0.0):.4f}")
            with col4:
                st.metric("**RECALL**", f"{eval_results.get('eval_recall', 0.0):.4f}")

            if trainer:
                st.session_state.model_info = {
                    "model_path": output_dir,
                    "labels_list": trainer.labels_list,
                    "label2id": trainer.label2id,
                    "id2label": trainer.id2label
                }

        except Exception as e:
            st.error(f"학습 중단: {e}")
            st.code(str(e))
            if trainer is None:
                st.warning("Trainer 객체가 생성되지 못했습니다.")