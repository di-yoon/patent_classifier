# methods/prompt_engineering/ui.py
import streamlit as st, pandas as pd, time
from . import prompts, results
from .client import LMStudioClient
from utils.db_utils import save_prompt_results
from .optimizer import optimize_prompt

class PromptUI:
    def __init__(self):
        st.session_state.setdefault("main_prompt", prompts.DEFAULT_PROMPT)
        st.session_state.setdefault("categories", {"": ""})
        st.session_state.setdefault("num_categories", 1)

    def show(self, df):
        st.header("PROMPT ENGINEERING")

        # === Data prep ===
        with st.expander("**DATA PREPARATION**", expanded=True):
            st.dataframe(df.head(), width='stretch')
            st.metric("ROWS", len(df))
            sel = st.multiselect("COLUMNS TO INCLUDE", df.columns.tolist())
            sep = ""
            if len(sel)>1:
                m = st.selectbox("MERGE METHOD", ["SPACE","LINE BREAKS","CUSTOM"])
                sep = " " if m=="SPACE" else "\n" if m=="LINE BREAKS" else st.text_input("DELIMITER"," | ")

        # === API ===
        client = LMStudioClient()
        with st.expander("**LM STUDIO SETTING**"):
            st.text_input("BASE URL", value=client.api_url, key="api_url")
            st.text_input("MODEL", value=client.api_model, key="api_model")
            if st.button("API CONNECTION"): client.connect()

        # === Category ===
        # 세션 초기화
        if "categories" not in st.session_state:
            st.session_state.categories = {"": ""}
            st.session_state.num_categories = 1

        with st.expander("**CATEGORY TO CLASSIFY**", expanded=True):
            updated = {}
            for i in range(st.session_state.num_categories):
                col1, col2, col3 = st.columns([1, 3, 0.15])
                code = col1.text_input(f"CATEGORY {i + 1}", key=f"code_{i}")
                desc = col2.text_area(f"DESCRIPTION {i + 1}", key=f"desc_{i}", height=30)
                if col3.button("✖️", key=f"rm{i}") and st.session_state.num_categories > 1:
                    st.session_state.num_categories -= 1
                    st.rerun()
                updated[code.strip()] = desc.strip()

            st.session_state.categories = updated or {"": ""}

            col1, col2 = st.columns([1, 1])
            if col1.button("➕ ADD CATEGORY"):
                st.session_state.num_categories += 1
                st.rerun()
            if col2.button("®️ RESET"):
                st.session_state.categories = {"": ""}
                st.session_state.num_categories = 1
                st.rerun()

        # === Prompt ===
        with st.expander("**PROMPT TEMPLATE**"):
            st.session_state.main_prompt=st.text_area("CLASSIFICATION PROMPT",st.session_state.main_prompt,height=400)
            col1,col2,col3=st.columns(3)
            if col1.button("RESET EMPTY"): st.session_state.main_prompt=""; st.rerun()
            if col2.button("LOAD SAMPLE"): st.session_state.main_prompt=prompts.SAMPLE_PROMPT; st.rerun()
            if col3.button("OPTIMIZE PROMPT"):
                optimized = optimize_prompt(st.session_state.main_prompt)
                st.session_state.main_prompt = optimized
                st.success("프롬프트가 최적화되었습니다 ")
                st.rerun()

        # === Run ===
        if st.button("**C L A S S I F Y**",width='stretch'):
            if not st.session_state.get("api_connection_success"):
                st.warning("Connect API first"); return
            texts=df[sel].astype(str).agg(sep.join,axis=1) if len(sel)>1 else df[sel[0]].astype(str)
            results_data=[]
            prog=st.progress(0)
            for i,t in enumerate(texts.dropna()):
                cls = client.classify(t, st.session_state.categories, st.session_state.main_prompt)
                results_data.append({
                    "index":i,"text":t,"classification":cls,
                    "preview":t[:100]+"..." if len(t)>100 else t
                })
                prog.progress((i+1)/len(texts)); time.sleep(0.2)
            st.session_state.classification_results=results_data
            st.toast("DONE")

        # === Results ===
        if st.session_state.classification_results:
            dfres = pd.DataFrame(st.session_state.classification_results)
            st.dataframe(dfres, width='stretch')
            st.bar_chart(dfres["classification"].value_counts())
            results.export_excel(dfres, dfres.groupby("classification"))

            run_id = save_prompt_results(
                st.session_state.num_categories,
                st.session_state.main_prompt,
                st.session_state.categories,
                dfres[["text", "classification"]]
            )
            st.success(f"프롬프트 결과가 SQLite DB에 저장되었습니다. Run ID: {run_id}")
