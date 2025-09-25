# methods/db_viewer/ui.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from . import handler, schema

def format_kst(ts_str: str) -> str:
    #UTC -> KST
    try:
        ts_obj = datetime.fromisoformat(ts_str)
        ts_kst = ts_obj + timedelta(hours=9)
        return ts_kst.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts_str

def show():
    st.header("DB Viewer")

    tab1, tab2, tab3 = st.tabs(["학습 기록", "추론 기록", "프롬프트 기록"])

    # train
    with tab1:
        metrics_df = handler.load_table("train_metrics", limit=500)
        if metrics_df.empty:
            st.info("아직 저장된 학습 결과가 없습니다.")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                display_df = metrics_df[["id", "timestamp", "model_name", "accuracy"]].copy()
                display_df.rename(columns={
                    "id": "Run ID", "timestamp": "Time",
                    "model_name": "Model", "accuracy": "Acc"
                }, inplace=True)
                st.dataframe(display_df, width='stretch', height=400, hide_index=True)

            with col2:
                st.markdown("**상세보기 & 삭제**")
                run_id_list = metrics_df["id"].tolist()
                selected_id = st.selectbox("상세보기 Run ID", run_id_list, key="train_detail_id")
                delete_ids = st.multiselect("삭제할 Run ID", run_id_list, key="train_delete_ids")
                if st.button("선택한 Run 삭제", key="delete_train"):
                    handler.delete_runs("train_metrics", delete_ids)
                    st.success(f"삭제 완료: {delete_ids}")
                    st.rerun()

            if selected_id:
                row = metrics_df.loc[metrics_df["id"] == selected_id].iloc[0]
                st.markdown("---")
                st.markdown(f"### Run {row['id']} | {format_kst(row['timestamp'])} | {row['model_name']}")

                c1, c2, c3, c4, c5 = st.columns(5)
                with c1: st.metric("Accuracy", f"{row.get('accuracy', 0.0):.4f}")
                with c2: st.metric("F1 Score", f"{row.get('f1', 0.0):.4f}")
                with c3: st.metric("Precision", f"{row.get('precision', 0.0):.4f}")
                with c4: st.metric("Recall", f"{row.get('recall', 0.0):.4f}")
                with c5: st.metric("Loss", f"{row.get('loss', 0.0):.4f}")

                st.subheader("Information")
                info_rows = [
                    {"Key": "Text Column", "Value": row["text_col"]},
                    {"Key": "Label Column", "Value": row["label_col"]},
                    {"Key": "Labels", "Value": row["labels"]},
                    {"Key": "Output Dir", "Value": row["output_dir"]},
                    {"Key": "Timestamp", "Value": format_kst(row["timestamp"])},
                ]
                st.dataframe(pd.DataFrame(info_rows), width='stretch', hide_index=True)

                # 파라미터 표시
                import json
                st.subheader("Hyperparameter Settings")

                def show_params_as_table(title, cfg_json):
                    try:
                        cfg = json.loads(cfg_json) if isinstance(cfg_json, str) else cfg_json
                        if cfg:
                            rows = [{"Parameter": k, "Value": v} for k, v in cfg.items()]
                            st.markdown(f"**{title}**")
                            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
                        else:
                            st.write(f"{title}: (no data)")
                    except Exception as e:
                        st.warning(f"{title} 로드 실패: {e}")

                show_params_as_table("BNB Config", row.get("bnb_config", "{}"))
                show_params_as_table("LoRA Config", row.get("lora_config", "{}"))
                show_params_as_table("Trainer Config (SFT)", row.get("training_config", "{}"))
    # inference
    with tab2:
        runs_df = handler.load_table("inference_runs", limit=50)
        results_df = handler.load_table("inference_results", limit=200)

        if not runs_df.empty:
            col1, col2 = st.columns([3, 1])
            with col1:
                display_df = runs_df[["id", "timestamp", "model_name", "selected_col"]].copy()
                display_df.rename(columns={
                    "id": "Run ID", "timestamp": "Time",
                    "model_name": "Model", "selected_col": "Column"
                }, inplace=True)
                st.dataframe(display_df, width='stretch', height=250, hide_index=True)

            with col2:
                st.markdown("**상세보기 & 삭제**")
                run_id_list = runs_df["id"].tolist()
                selected_id = st.selectbox("상세보기 Run ID", run_id_list, key="infer_detail_id")
                delete_ids = st.multiselect("삭제할 Run ID", run_id_list, key="infer_delete_ids")
                if st.button("선택한 Run 삭제", key="delete_inference"):
                    handler.delete_runs("inference_runs", delete_ids)
                    handler.delete_runs("inference_results", delete_ids)
                    st.success(f"삭제 완료: {delete_ids}")
                    st.rerun()

            if selected_id:
                row = runs_df.loc[runs_df["id"] == selected_id].iloc[0]
                st.markdown("---")
                st.markdown(f"### Run {row['id']} | {format_kst(row['timestamp'])} | {row['model_name']}")

                st.subheader("추론 결과")
                if "run_id" in results_df.columns:
                    results_df["run_id"] = pd.to_numeric(results_df["run_id"], errors="coerce").astype("Int64")
                    run_results = results_df[results_df["run_id"] == int(selected_id)]
                else:
                    run_results = pd.DataFrame()

                if run_results.empty:
                    st.warning("해당 Run ID에 대한 저장된 결과가 없습니다.")
                else:
                    cols = [c for c in ["출원번호", "예측_라벨", "신뢰도"] if c in run_results.columns]

                    # 🔍 필터링 UI 추가
                    with st.expander("🔍 결과 필터링", expanded=True):
                        unique_labels = run_results["예측_라벨"].unique().tolist() if "예측_라벨" in run_results.columns else []
                        selected_labels = st.multiselect("라벨 선택", unique_labels, default=unique_labels)

                        conf_min, conf_max = 0.0, 1.0
                        if "신뢰도" in run_results.columns:
                            conf_range = st.slider("신뢰도 범위", 0.0, 1.0, (0.0, 1.0), step=0.05)
                            conf_min, conf_max = conf_range

                        search_id = st.text_input("출원번호 검색", "")

                    # === 필터 적용 ===
                    filtered = run_results.copy()
                    if selected_labels:
                        filtered = filtered[filtered["예측_라벨"].isin(selected_labels)]
                    if "신뢰도" in filtered.columns:
                        filtered = filtered[(filtered["신뢰도"] >= conf_min) & (filtered["신뢰도"] <= conf_max)]
                    if search_id:
                        filtered = filtered[filtered["출원번호"].astype(str).str.contains(search_id)]

                    # 결과 테이블 표시
                    st.dataframe(filtered[cols] if cols else filtered,
                                 width='stretch', height=400, hide_index=True)

                    # 분포 차트도 필터 반영
                    if "예측_라벨" in filtered.columns:
                        st.subheader("PREDICTION DISTRIBUTION (Filtered)")
                        st.bar_chart(filtered["예측_라벨"].value_counts())

        else:
            st.info("아직 저장된 추론 실행 기록이 없습니다.")

    # prompt
    with tab3:
        runs_df = handler.load_table("prompt_runs", limit=50)
        results_df = handler.load_table("prompt_results", limit=200)

        if not runs_df.empty:
            col1, col2 = st.columns([3, 1])
            with col1:
                display_df = runs_df[["id", "timestamp", "num_categories"]].copy()
                display_df.rename(columns={
                    "id": "Run ID", "timestamp": "Time",
                    "num_categories": "Num Categories"
                }, inplace=True)
                st.dataframe(display_df, width='stretch', height=250, hide_index=True)

            with col2:
                st.markdown("**상세보기 & 삭제**")
                run_id_list = runs_df["id"].tolist()
                selected_id = st.selectbox("상세보기 Run ID", run_id_list, key="prompt_detail_id")
                delete_ids = st.multiselect("삭제할 Run ID", run_id_list, key="prompt_delete_ids")
                if st.button("선택한 Run 삭제", key="delete_prompt"):
                    handler.delete_runs("prompt_runs", delete_ids)
                    handler.delete_runs("prompt_results", delete_ids)
                    st.success(f"삭제 완료: {delete_ids}")
                    st.rerun()

            if selected_id:
                row = runs_df.loc[runs_df["id"] == selected_id].iloc[0]
                st.markdown("---")
                st.markdown(f"### Run {row['id']} | {format_kst(row['timestamp'])}")

                # 프롬프트 템플릿
                st.subheader("사용된 프롬프트 템플릿")
                st.code(row["prompt_template"], language="markdown")

                # 카테고리
                st.subheader("카테고리")
                import json
                try:
                    categories = json.loads(row.get("categories", "{}"))
                    if isinstance(categories, dict) and categories:
                        cat_rows = [{"Code": k, "Description": v} for k, v in categories.items()]
                        st.dataframe(pd.DataFrame(cat_rows), width='stretch', hide_index=True)
                    else:
                        st.write("카테고리 정보가 없습니다.")
                except Exception:
                    st.write(row.get("categories", "카테고리 정보 없음"))

                # 결과
                st.subheader("프롬프트 결과")
                if "run_id" in results_df.columns:
                    results_df["run_id"] = pd.to_numeric(results_df["run_id"], errors="coerce").astype("Int64")
                    run_results = results_df[results_df["run_id"] == int(selected_id)]
                else:
                    run_results = pd.DataFrame()

                if run_results.empty:
                    st.warning(" 해당 Run ID에 대한 저장된 결과가 없습니다.")
                else:
                    cols = [c for c in ["text", "classification"] if c in run_results.columns]
                    st.dataframe(run_results[cols] if cols else run_results,
                                 width='stretch', height=400)

                    if "classification" in run_results.columns:
                        st.subheader("분류 결과 분포")
                        st.bar_chart(run_results["classification"].value_counts())
        else:
            st.info("아직 저장된 프롬프트 실행 기록이 없습니다.")
