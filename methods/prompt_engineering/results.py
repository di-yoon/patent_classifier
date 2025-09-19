# methods/prompt_engineering/results.py
import pandas as pd, time
from io import BytesIO
import streamlit as st

def export_excel(results_df, groups):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        results_df[['text','classification']].rename(
            columns={'text':'TEXT','classification':'CLASSIFICATION'}
        ).to_excel(writer, sheet_name="전체", index=False)
        for cat,g in groups:
            safe = str(cat).replace('/','_').replace(':','_')[:31]
            g[['text','classification']].rename(
                columns={'text':'TEXT','classification':'CLASSIFICATION'}
            ).to_excel(writer, sheet_name=safe, index=False)
        stats = results_df['classification'].value_counts().reset_index()
        stats.columns = ['CLASSIFICATION','COUNT']
        stats.to_excel(writer,sheet_name="통계",index=False)
    buf.seek(0)
    st.download_button(
        " DOWNLOAD PROMPT ENGINEERING RESULT (EXCEL)",
        buf.getvalue(),
        f"patent_classification_prompt_{int(time.time())}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
