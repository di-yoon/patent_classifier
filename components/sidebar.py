import pandas as pd
import streamlit as st

def show():
    st.sidebar.title("[ CLASSIFICATION METHOD ]")
    classification_method = st.sidebar.selectbox(
        "분류 방법 선택",
        ["-- SELECT --", "PROMPT ENGINEERING", "FINE TUNING", "DB VIEWER"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    st.sidebar.title("[ UPLOAD DATA ]")
    uploaded_file = st.sidebar.file_uploader(
        "데이터 업로드",
        type=['csv', 'xlsx', 'xls'],
        help="파일 형식 : CSV, EXCEL",
        key="data_upload",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                selected_sheet = st.sidebar.selectbox("CHOOSE SHEET", sheet_names)
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

            # ✅ 업로드 완료 메시지
            st.sidebar.success("데이터 업로드 완료")

            # ✅ 자동 컬럼 매핑 추천
            default_cols = [col for col in ["발명의 명칭", "요약", "전체청구항"] if col in df.columns]
            st.session_state.default_cols = default_cols

            # 원본 데이터 저장
            st.session_state.uploaded_df = df

        except Exception as e:
            st.sidebar.error(e)

    st.sidebar.markdown("---")

    if classification_method == "PROMPT ENGINEERING":
        st.sidebar.title("[ HOW TO USE ]")
        st.sidebar.subheader("PROMPT ENGINEERING")
        st.sidebar.markdown("""
        1. CLASSIFICATION METHOD
        2. UPLOAD DATA
        3. DATA PREPARATION _ COLUMNS TO INCLUDE IN PROMPT
        4. CATEGORY TO CLASSIFY
        5. PROMPT TEMPLATE
        6. LM STUDIO SETTING
        7. CLASSIFY
        """)
    elif classification_method == "FINE TUNING":
        st.sidebar.title("[ HOW TO USE ]")
        st.sidebar.subheader("FINE TUNING")
        st.sidebar.markdown("""
        1. 데이터 파일 업로드
        2. 학습 탭: 모델 학습 실행
        3. 추론 탭: 학습된 모델로 추론
        4. 결과 다운로드
        """)
    else:
        pass
        
    return classification_method