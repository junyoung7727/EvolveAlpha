from dataclasses import dataclass
from datetime import datetime
import json
from typing import List
from src.agents.base_agent import BaseAgent
from src.discussion.discussion_round import DiscussionRound

@dataclass
class DiscussionResult:
    roles: List[str]
    rounds: List[dict]
    final_conclusion: str
    timestamp: datetime

class DiscussionManager:
    def __init__(self, num_rounds: int = 5):
        self.num_rounds = num_rounds
        
    def run_discussion(self, roles: List[str]) -> DiscussionResult:
        """전체 토론 진행"""
        agents = [BaseAgent(role, f"{role}의 전문가") for role in roles]
        rounds = []
        
        topics = self._generate_topics()
        
        for round_num in range(self.num_rounds):
            discussion_round = DiscussionRound(round_num, agents)
            round_result = discussion_round.conduct_round(topics[round_num])
            print(f"라운드 {round_num + 1} 결과: {round_result.summary}")
            rounds.append(round_result.__dict__)
            
        final_conclusion = self._generate_conclusion(rounds)
        
        return DiscussionResult(
            roles=roles,
            rounds=rounds,
            final_conclusion=final_conclusion,
            timestamp=datetime.now()
        )
    
    def _generate_topics(self) -> List[str]:
        """각 라운드별 주제 생성"""
        return [
            "현재 시장 상황 분석",
            "주요 리스크 요인 식별",
            "대응 전략 수립",
            "장기적 영향 평가",
            "최종 권고사항 도출"
        ]
    
    def _generate_conclusion(self, rounds: List[dict]) -> str:
        """최종 결론 도출"""
        conclusion_prompt = f"""
        전체 토론 내용:
        {json.dumps(rounds, ensure_ascii=False, indent=2)}
        
        위 토론 내용을 종합하여 다음 사항을 포함한 최종 결론을 도출해주세요:
        1. 핵심 발견사항
        2. 전문가들의 주요 합의점
        3. 실행 가능한 권고사항
        4. 향후 모니터링이 필요한 부분
        """
        
        model_manager = BaseAgent("Conclusion Generator", "").model_manager
        conclusion = model_manager.generate_response(
            role="Discussion Facilitator",
            speciality="Final Conclusion",
            role_prompt=conclusion_prompt
        )
        return conclusion
    
    def save_result(self, result: DiscussionResult, filename: str):
        """토론 결과 저장"""
        with open(f"results/discussions/{filename}.json", "w", encoding="utf-8") as f:
            json.dump(result.__dict__, f, ensure_ascii=False, default=str, indent=2)