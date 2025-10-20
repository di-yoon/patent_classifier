# methods/prompt_engineering/client.py
import requests, streamlit as st

class LMStudioClient:
    def __init__(self, api_url=None, api_model=None):
        # API URL과 사용할 모델명
        self.api_url = api_url or st.session_state.get("api_url", "http://localhost:1234/v1/chat/completions")
        self.api_model = api_model or st.session_state.get("api_model")

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

    def classify(self, text: str, optimized_prompt: str) -> str:

        full_prompt = f"""{optimized_prompt}

    [분류할 텍스트]
    {text}
    """

        payload = {
            "model": self.api_model,
            "messages": [
                {"role": "system", "content": "당신은 텍스트를 분류하는 전문가입니다."},
                {"role": "user", "content": full_prompt}
            ],
            "temperature": 0,
            "max_tokens": 200
        }

        try:
            r = requests.post(self.api_url, json=payload, timeout=60)
            if r.status_code == 200:
                result = r.json()["choices"][0]["message"]["content"].strip()
                return result
            else:
                return f"Error: {r.status_code} {r.text}"
        except Exception as e:
            return f"Exception: {str(e)}"
