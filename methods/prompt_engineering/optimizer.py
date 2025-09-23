from .client import LMStudioClient
import requests, streamlit as st

def optimize_prompt(raw_prompt: str) -> str:
    """
    사용자가 입력한 프롬프트를 분류 작업에 최적화된 형태로 변환
    """
    client = LMStudioClient()
    meta_prompt = f"""
    당신은 Prompt Engineering 전문가입니다.
    아래 사용자가 작성한 프롬프트를 '분류 작업(classification task)'에 최적화해 주세요.

    [개선 지침]
    불필요하거나 모호한 표현은 제거한다.
    모델의 역할(Role)을 명확히 정의한다. 
    작업(Task)을 구체적으로 설명한다.
    출력 규칙(Output Rules)을 엄격히 제한한다. 
     - 최종 출력에는 카테고리 이름 하나만 포함하세요.
     - 추가 설명이나 부가 텍스트는 포함하지 마세요.
    분류 기준(Classification Criteria)을 명시한다.
    원본 프롬프트의 언어(한국어/영어 등)를 그대로 유지한다.
     - 반드시 최적화된 프롬프트 한 개만 출력하세요.
     - 다른 설명, 분석, 비판, 예시는 절대 포함하지 마세요.
     - 출력은 프롬프트 본문만 남기세요.
    사용자의 짧은 입력 내용을 최대한 보존하면서, 누락된 요소를 보완해라.  
    모델의 역할(Role)을 명확히 정의한다.  
     - 예: "당신은 텍스트를 분류하는 전문가입니다."  
    작업(Task)을 구체적으로 확장한다.  
     - 분류할 대상이 무엇인지, 어떤 기준으로 나눠야 하는지를 설명한다.  
    출력 규칙(Output Rules)을 반드시 포함한다.  
     - 최종 출력은 반드시 단 하나의 카테고리명만 포함해야 한다.  
     - 다른 설명, 분석, 비판, 예시는 절대 포함하지 않는다.  
     - 출력 예시: `CATEGORY_X`  
    분류 기준(Classification Criteria)을 합리적으로 추론해 제시한다.  
    짧은 입력에 명확히 정의되지 않은 경우, 일반적인 상식이나 기본 분류 원칙을 기반으로 작성한다.  
    원본 입력의 언어(한국어/영어 등)를 그대로 유지한다.  
    최종 출력에는 오직 최적화된 프롬프트 한 개만 작성한다.
    [입력 프롬프트]
    {raw_prompt}

    [출력]
    - 위 지침을 반영하여 최적화된 프롬프트 한 개만 작성한다.
    """

    try:
        r = requests.post(client.api_url, json={
            "model": client.api_model,
            "messages": [
                {"role": "system", "content": "You are a prompt optimization assistant."},
                {"role": "user", "content": meta_prompt}
            ],
            "temperature": 0,
            "max_tokens": 500
        }, timeout=60)

        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content'].strip()
        else:
            st.error(f"Optimizer API error: {r.status_code}")
            return raw_prompt
    except Exception as e:
        st.error(f"Optimizer request failed: {e}")
        return raw_prompt
