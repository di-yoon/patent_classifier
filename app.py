from components import sidebar
from methods.finetuning import ui as finetuning
from methods.prompt_engineering.ui import PromptUI
import streamlit as st
from methods.db_viewer import ui as db_viewer
from utils.db_utils import init_train_db, init_infer_db, init_prompt_db
# DB 초기화
init_train_db()
init_infer_db()
init_prompt_db()

# Streamlit 페이지 설정
st.set_page_config(page_title="특허 문서 분류 자동화 플랫폼", layout="wide")
st.title("특허 문서 분류 자동화 플랫폼")
st.markdown("---")

# 공통 세션 초기값
_defaults = {
    "api_url": "http://localhost:1234/v1/chat/completions",
    "api_model": None,
    "classification_results": None,
    "model_loaded": False,
    "tokenizer": None,
    "model": None,
    "training_data": None,
    "uploaded_df": None,
}
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)

classification_method = sidebar.show()

if classification_method == "PROMPT ENGINEERING":
    if st.session_state.uploaded_df is not None:
        ui = PromptUI()
        ui.show(st.session_state.uploaded_df)
    else:
        st.info("⬅️ 특허 문서를 업로드해 주세요.")

elif classification_method == "FINE TUNING":
    finetuning.show()

elif classification_method == "DB VIEWER":
    db_viewer.show()


else:
    st.info("⬅️ 왼쪽에서 작업 모드를 선택하세요.")

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:gray;'>
        특허 문서 분류 자동화 플랫폼 © IPickYou
    </div>
    """,
    unsafe_allow_html=True,
)