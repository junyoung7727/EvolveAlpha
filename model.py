from huggingface_hub import InferenceClient
from anthropic import Anthropic
import openai
from dotenv import load_dotenv
import os
import time
from typing import List, Dict

load_dotenv()
print(os.getenv("HF_TOKEN"))
class LLMHandler:
    def __init__(self, model_type: str = "huggingface"):
        self.model_type = model_type.lower()
        self._setup_client()

    def _setup_client(self):
        """선택된 모델 타입에 따라 클라이언트 설정"""
        if self.model_type == "huggingface":
            self.client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
            self.model_name = "HuggingFaceH4/zephyr-7b-beta"
        elif self.model_type == "openai":
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = "gpt-4o-mini"
        elif self.model_type == "anthropic":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model_name = "claude-3-sonnet-20240229"
        else:
            raise ValueError(f"지원하지 않는 모델 타입입니다: {self.model_type}")

    def generate_completion(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        """메시지를 받아 LLM 응답을 생성"""
        for attempt in range(max_retries):
            try:
                if self.model_type == "huggingface":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7
                    )
                    return response.choices[0].message.content

                elif self.model_type == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7
                    )
                    return response.choices[0].message.content

                elif self.model_type == "anthropic":
                    response = self.client.messages.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=1000
                    )
                    return response.content[0].text

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"최종 오류 발생: {str(e)}")
                print(f"시도 {attempt + 1}/{max_retries} 실패: {str(e)}")
                time.sleep(5)

def generate_alpha_strategy(model_type: str = "huggingface"):
    """알파 전략 생성 함수"""
    try:
        llm = LLMHandler(model_type)
        
        messages = [
            {
                "role": "user",
                "content": """
Create three distinct alpha strategies in the specified format:

Create three distinct alpha strategies with the following specifications:

1. **FORMULA STRUCTURE:**
   - You have full creative freedom to design the alpha formula.
   - The formula should capture meaningful patterns or inefficiencies in the market using available variables and indicators.
   - Use variables or combinations of variables innovatively, focusing on both economic intuition and practical execution.

2. **AVAILABLE VARIABLES:**
   - **Price Variables:**
     - CLOSE, OPEN, HIGH, LOW, VWAP
   - **Volume Variables:**
     - VOLUME, TURNOVER
   - **Technical Indicators:**
     - MA(price, N): Moving average
     - EMA(price, N): Exponential moving average
     - RSI(N): Relative strength index
     - BOLL_UP, BOLL_DOWN: Bollinger Bands
     - ATR(N): Average true range
     - STD(price, N): Standard deviation
   - **Fundamental Ratios:**
     - EPS, BOOK_VALUE, PE, PB
   - **Time Series Functions:**
     - DELAY(X, N): Value of X, N periods ago
     - MEAN(X, N): Mean of X over N periods
     - RANK(X): Cross-sectional rank of X
     - CORR(X, Y, N): Correlation between X and Y over N periods

3. **REQUIREMENTS:**
   - Formulas should reflect a specific market inefficiency or behavior (e.g., momentum, mean-reversion, liquidity anomalies).
   - Incorporate appropriate statistical methods (e.g., Z-score normalization or ranking) for signal standardization and comparability.
   - Use robust lookback periods (e.g., 5–60 days) based on the strategy's objectives.
   - Avoid unrealistic assumptions (e.g., zero division errors or overfitting).

4. **OUTPUT FORMAT:**
   For each alpha strategy, output:
   - The complete formula, written in a clear and logical manner.
   - The economic intuition behind the strategy (e.g., why it should work in theory).
   - Expected behavior in different market conditions (e.g., trending, volatile, or range-bound markets).
   - Typical lookback period or parameter settings used.
   - Potential risks or limitations to the strategy.

5. **STRATEGY TYPES:**
   - Generate three different alpha strategies:
     1. A **momentum-based alpha** (e.g., trend-following, breakout detection).
     2. A **mean-reversion-based alpha** (e.g., oscillation, oversold/overbought conditions).
     3. A **volume or liquidity-based alpha** (e.g., volume spikes, turnover anomalies).

6. **INNOVATIVE APPROACH:**
   - Feel free to combine variables in unique ways or introduce new perspectives on market behavior.
   - For example:
     - Combine price and volume metrics to identify hidden patterns.
     - Use cross-sectional rankings or correlations across multiple assets.
     - Focus on advanced statistics like decay, z-score, or weighted averages for dynamic signals.


Write alpha expression formula below in json format.
"""
            }
        ]
        
        response = llm.generate_completion(messages)
        print(response)
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    # 원하는 모델 선택하여 실행
    # "huggingface", "openai", "anthropic" 중 선택
    generate_alpha_strategy("openai")
