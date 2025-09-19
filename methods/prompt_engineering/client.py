# methods/prompt_engineering/client.py
import requests, streamlit as st

class LMStudioClient:
    def __init__(self, api_url=None, api_model=None):
        self.api_url = api_url or st.session_state.get("api_url", "http://localhost:1234/v1/chat/completions")
        self.api_model = api_model or st.session_state.get("api_model", "qwen/qwen3-14b")

    def connect(self):
        try:
            r = requests.post(self.api_url, json={
                "model": self.api_model,
                "messages":[{"role":"user","content":"Hello"}],
                "max_tokens":10
            }, timeout=30)
            if r.status_code == 200:
                st.success("API CONNECTION SUCCESSFUL")
                st.session_state.api_connection_success = True
                return True
        except Exception as e:
            st.error(f"API CONNECTION FAILED: {e}")
        st.session_state.api_connection_success = False
        return False

    def classify(self, text, candidates, prompt_template):
        candidate_text = "\n".join([f"{c}: {d}" for c,d in candidates.items()])
        prompt = prompt_template.format(text=text, candidate_text=candidate_text)
        try:
            r = requests.post(self.api_url, json={
                "model": self.api_model,
                "messages":[
                    {"role":"system","content":"당신은 특허 문서를 분류코드 체계로 분류하는 전문가입니다."},
                    {"role":"user","content":prompt}
                ],
                "temperature":0, "max_tokens":15
            }, timeout=60)
            if r.status_code==200:
                return r.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            st.error(e)
        return "ERROR"
