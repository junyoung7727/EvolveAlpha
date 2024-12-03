from typing import Dict, Any, List
from ..models.model_manager import ModelManager
from .role_definitions import ROLE_DESCRIPTIONS

class BaseAgent:
    def __init__(self, role_name: str, description: str = None):
        self.role_name = role_name
        self.description = description or ROLE_DESCRIPTIONS.get(role_name, "전문가")
        self.model_manager = ModelManager(model_type="openai")
        self.speciality = self._get_speciality()
        self.role_prompt = self._get_role_prompt()
        
    def _get_speciality(self) -> str:
        """역할별 전문 분야 정의"""
        specialities = {
            "Quantitative Analyst": "Financial Modeling and Statistical Analysis",
            "Behavioral Economist": "Human Behavior in Economic Decision Making",
            "Risk Manager": "Risk Assessment and Mitigation Strategies",
            "Market Historian": "Historical Market Trends and Patterns",
            "Data Engineer": "Data Processing and Pipeline Architecture",
            "Technical Analyst": "Technical Indicators and Chart Analysis",
            "Macro Economist": "Global Economic Trends and Policies",
            "Machine Learning Expert": "AI/ML Applications in Finance",
            "통계학자": "Statistical Methods and Probability Theory",
            "금융공학자": "Financial Instrument Design and Pricing",
            "생명공학자": "Biological Systems and Their Applications",
            "시계열 데이터 전문가": "Time Series Analysis and Forecasting",
            "수학자": "Mathematical Modeling and Abstract Theory",
            "심리학자": "Human Psychology and Behavior Analysis"
        }
        return specialities.get(self.role_name, "General Analysis")

    def _get_role_prompt(self) -> str:
        """역할별 기본 프롬프트 정의"""
        return f"""당신은 {self.role_name}입니다. {self.description}
        
        다음 관점에서 분석해주세요:
        1. 당신의 전문 분야인 {self.speciality}의 관점
        2. 현재 논의되는 주제와의 연관성
        3. 다른 전문가들의 의견을 고려한 통합적 시각
        
        명확하고 논리적인 분석을 제공해주세요."""

    def generate_response(self, context: str, previous_responses: List[str]) -> str:
        """토론에서 응답 생성"""
        full_context = f"""
        주제: {context}
        
        이전 응답들:
        {'\n'.join(previous_responses)}
        
        당신의 분석을 제공해주세요.
        """
        
        response = self.model_manager.generate_response(
            role=self.role_name,
            speciality=self.speciality,
            role_prompt=full_context
        )
        return response
