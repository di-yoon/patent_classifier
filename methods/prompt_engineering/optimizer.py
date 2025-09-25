# methods/prompt_engineering/optimizer.py
import requests, streamlit as st
from .client import LMStudioClient

def _call_llm(messages, max_tokens=700):
    """LMStudio API 호출 공통 함수"""
    client = LMStudioClient()
    try:
        r = requests.post(client.api_url, json={
            "model": client.api_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens
        }, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        else:
            st.error(f"Optimizer API error: {r.status_code}")
            return None
    except Exception as e:
        st.error(f"Optimizer request failed: {e}")
        return None


def optimize_prompt(raw_prompt: str, categories: dict) -> str:
    """
    사용자 프롬프트(raw_prompt) + 카테고리(categories)를 기반으로
    분류 작업에 최적화된 프롬프트 생성.
    - 카테고리 키(c)는 숫자/문자 구분 없이 그대로 결과에 사용됨
    - 설명(d)은 분류 기준 제시에 활용
    """

    # 카테고리 후보 문자열 생성 (사용자 입력 그대로 반영)
    candidate_text = "\n".join([
        f"{c}: {d}" for c, d in categories.items() if c.strip()
    ])

    meta_prompt = f"""
    당신은 Prompt Engineering 전문가입니다.  
아래에는 사용자가 정의한 [카테고리 코드와 설명]이 제공됩니다.  
또한 사용자가 작성한 [추가 요청 문구]도 함께 주어집니다.  

여기서 반드시 기억해야 할 점은 다음과 같습니다:  
1. [카테고리 코드와 설명]이 분류 작업의 절대적인 기준입니다.  
   - 각 카테고리 코드(숫자든 문자든)는 사용자가 직접 정의한 것이며,  
     최종 출력은 반드시 이 코드들 중 하나여야 합니다.  
   - 각 설명은 해당 코드가 의미하는 바와 분류 기준을 명확히 정의합니다.  
2. [추가 요청 문구]는 단순 참고용입니다.  
   - 사용자의 의도나 배경을 이해하는 데 도움을 줄 수 있지만,  
     최종 프롬프트를 구성할 때는 오직 [카테고리 코드와 설명]만을 근거로 해야 합니다.  
   - 즉, [추가 요청 문구]의 표현이 모호하거나 불완전하더라도  
     분류 지침은 언제나 [카테고리 코드와 설명]을 중심으로 구성해야 합니다.  
3. 최종 프롬프트는 다음을 반드시 포함해야 합니다:  
   - 모델의 역할(Role)  
   - 수행할 작업(Task)  
   - 카테고리 정의 및 분류 기준  
   - 출력 규칙(Output Rules)  

⚠️ 절대적으로 중요한 규칙:  
- 최종 출력은 반드시 사용자가 정의한 카테고리 코드 중 하나만 포함해야 합니다.  
- 카테고리 코드는 사용자가 입력한 그대로 사용해야 하며, 다른 변형은 허용되지 않습니다.  
- 추가 설명, 잡담, 불필요한 텍스트는 포함하지 말아야 합니다.  

따라서 최적화된 프롬프트를 작성할 때는 [카테고리 코드와 설명]을 중심으로,  
사용자가 정의한 분류 체계에 충실한 형태로 지침을 만들어야 합니다.  

    [사용자 요청]
    {raw_prompt}

    [카테고리 후보]
    {candidate_text}

    [지침]
    1. 각 카테고리 코드와 설명을 분류 기준으로 포함한다.
    2. 프롬프트에는 모델의 역할(Role), 작업(Task), 출력 규칙(Output Rules)을 포함한다.
    3. 출력 규칙:
       - 최종 출력은 반드시 위 카테고리 코드 중 하나만 출력하게 유도한다.
       - 카테고리 코드는 사용자가 입력한 그대로 사용해야 한다. (숫자, 문자 모두 허용)
       - 다른 설명, 문장, 부가 텍스트는 절대 포함하지 않는다.
    4. 최종 출력에는 최적화된 분류 프롬프트 한 개만 작성한다.

    [출력]
    - 최적화된 분류 프롬프트 한 개
    """

    stage_output = _call_llm([
        {"role": "system", "content": "You are a prompt optimizer. Return only the optimized classification prompt."},
        {"role": "user", "content": meta_prompt}
    ])

    if not stage_output:
        return raw_prompt

    # 후처리: '당신은' 또는 'You are' 로 시작하는 줄부터 반환
    lines = stage_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("당신은") or line.strip().lower().startswith("you are"):
            return "\n".join(lines[i:]).strip()

    return stage_output.strip()
