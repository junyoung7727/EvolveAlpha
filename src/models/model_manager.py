import os
import openai
from openai import OpenAI
from huggingface_hub import InferenceClient
from anthropic import Anthropic
from typing import Optional

class ModelManager:
    def __init__(self, model_type: str = "openai"):
        self.model_type = model_type
        self.client = None
        self.model_name = None
        self._setup_client()

    def _setup_client(self):
        """선택된 모델 타입에 따라 클라이언트 설정"""
        if self.model_type == "huggingface":
            self.client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
            self.model_name = "HuggingFaceH4/zephyr-7b-beta"
        elif self.model_type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = "gpt-4o-mini"  # 또는 "gpt-4" 또는 "gpt-3.5-turbo"
        elif self.model_type == "anthropic":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model_name = "claude-3-sonnet-20240229"
        else:
            raise ValueError(f"지원하지 않는 모델 타입입니다: {self.model_type}")

    def generate_response(self, role: str, speciality: str, role_prompt: str) -> str:
        """모델을 사용하여 응답 생성"""
        system_prompt = f"""당신은 {role}입니다. 전문 분야는 {speciality}입니다.
        전문가로서 객관적이고 논리적인 분석을 제공해주세요."""
        
        try:
            if self.model_type == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": role_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                opinion = response.choices[0].message.content.strip()
                print(f"{role}의 의견: {opinion}")
                return opinion
                
            elif self.model_type == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": f"{system_prompt}\n\n{role_prompt}"
                        }
                    ]
                )
                opinion = response.content[0].text
                print(f"{role}의 의견: {opinion}")
                return opinion
                
            elif self.model_type == "huggingface":
                full_prompt = f"{system_prompt}\n\n{role_prompt}"
                response = self.client.text_generation(
                    full_prompt,
                    max_new_tokens=1000,
                    temperature=0.7
                )
                opinion = response
                print(f"{role}의 의견: {opinion}")
                return opinion
                
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {str(e)}")
            return f"오류 발생: {str(e)}"

# 사용 예시
if __name__ == "__main__":
    model_manager = ModelManager(model_type="openai")
    response = model_manager.generate_response(
        role="Quantitative Analyst",
        speciality="Financial Modeling",
        role_prompt="Analyze the impact of interest rate changes on stock prices."
    )
    print(response) 