from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime
from src.agents.base_agent import BaseAgent

@dataclass
class RoundResult:
    round_number: int
    topic: str
    responses: List[Dict[str, str]]
    summary: str
    timestamp: datetime

class DiscussionRound:
    def __init__(self, round_number: int, agents: List[BaseAgent]):
        self.round_number = round_number
        self.agents = agents
        self.model_manager = agents[0].model_manager  # 첫 번째 에이전트의 모델 매니저 사용
        
    def conduct_round(self, topic: str) -> RoundResult:
        """한 라운드의 토론 진행"""
        responses = []
        previous_responses = []
        
        for agent in self.agents:
            response = agent.generate_response(topic, previous_responses)
            response_dict = {
                "role": agent.role_name,
                "response": response
            }
            responses.append(response_dict)
            previous_responses.append(f"{agent.role_name}: {response}")
            
        summary = self._generate_round_summary(topic, responses)
        
        return RoundResult(
            round_number=self.round_number,
            topic=topic,
            responses=responses,
            summary=summary,
            timestamp=datetime.now()
        )
    
    def _generate_round_summary(self, topic: str, responses: List[Dict[str, str]]) -> str:
        """라운드 요약 생성"""
        summary_prompt = f"""
        주제: {topic}
        
        전문가들의 응답:
        {'\n'.join([f"{r['role']}: {r['response']}" for r in responses])}
        
        위 토론 내용을 다음 형식으로 요약해주세요:
        1. 주요 논점
        2. 합의된 사항
        3. 대립된 의견
        4. 다음 라운드에서 더 논의가 필요한 부분
        """
        
        summary = self.model_manager.generate_response(
            role="Discussion Facilitator",
            speciality="Discussion Summary",
            role_prompt=summary_prompt
        )
        return summary
