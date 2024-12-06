from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json
from src.agents.base_agent import BaseAgent
from src.models.model_manager import ModelManager
from src.utils.logger_config import setup_logger

@dataclass
class RoundResult:
    round_number: int
    topic: str
    responses: List[Dict[str, Any]]
    evidence_collection: List[Dict[str, Any]]
    proposed_alphas: List[Dict[str, Any]]
    summary: str
    timestamp: datetime
    context_maintained: Dict[str, Any]

class DiscussionRound:
    def __init__(self, agents: List[BaseAgent], round_number: int, model_manager: ModelManager, previous_context: Dict[str, Any] = None):
        self.agents = agents
        self.round_number = round_number
        self.model_manager = model_manager
        self.previous_context = previous_context or {}
        self.logger = setup_logger("Round_" + str(round_number))
        
    def log_round_progress(self, stage: str, content: Dict[str, Any]):
        """라운드 진행 상황 로깅"""
        self.logger.info(f"\n{'#'*70}")
        self.logger.info(f"라운드 {self.round_number} | 단계: {stage}")
        self.logger.info(f"{'#'*70}")
        
        for key, value in content.items():
            self.logger.info(f"\n{key}:")
            if isinstance(value, (dict, list)):
                self.logger.info(f"{json.dumps(value, ensure_ascii=False, indent=2)}")
            else:
                self.logger.info(f"{value}")
                
        self.logger.info(f"\n{'#'*70}\n")

    def _prepare_context_for_agent(self, agent: BaseAgent) -> Dict[str, Any]:
        """각 에이전트를 위한 컨텍스트 준비"""
        if not self.previous_context:
            return {}
            
        relevant_context = {
            "previous_experiments": [],
            "previous_findings": [],
            "proposed_alphas": [],
            "ongoing_research": {}
        }
        
        # 이전 라운드들의 정보 수집
        for round_data in self.previous_context.get("rounds", []):
            # 해당 에이전트의 이전 실험 결과 수집
            for response in round_data.responses:
                if response["role"] == agent.role_name:
                    if "evidence" in response:
                        relevant_context["previous_experiments"].extend(
                            response["evidence"].get("methods_used", [])
                        )
                    relevant_context["previous_findings"].append({
                        "round": round_data.round_number,
                        "findings": response.get("response", "")
                    })
            
            # 제안된 알파 전략 수집
            for alpha in round_data.proposed_alphas:
                if alpha["role"] == agent.role_name:
                    relevant_context["proposed_alphas"].append({
                        "round": round_data.round_number,
                        "alpha": alpha["proposal"]
                    })
        
        # 진행 중인 연구 주제 식별
        if relevant_context["previous_findings"]:
            ongoing_research_prompt = f"""
            이전 라운드들의 연구 내용을 분석하여 현재 진행 중인 연구 주제들을 식별해주세요.
            
            이전 발견사항들:
            {json.dumps(relevant_context["previous_findings"], ensure_ascii=False, indent=2)}
            
            JSON 형식으로 응답해주세요:
            {{
                "ongoing_topics": ["진행 중인 연구 주제들"],
                "unresolved_questions": ["미해결된 질문들"],
                "promising_directions": ["유망한 연구 방향들"]
            }}
            """
            
            ongoing_research = self.model_manager.generate_response(
                role=agent.role_name,
                speciality="Research Continuity Analysis",
                role_prompt=ongoing_research_prompt
            )
            relevant_context["ongoing_research"] = ongoing_research
            
        return relevant_context

    def conduct_round(self, topic: str) -> RoundResult:
        """라운드 진행"""
        self.logger.info(f"\n=== 라운드 {self.round_number} 시작: {topic} ===")
        
        context = {
            "topic": topic,
            "round_number": self.round_number,
            "previous_context": self.previous_context
        }
        
        responses = []
        evidence_collection = []
        proposed_alphas = []
        previous_opinions = []
        
        # 첫 번째 에이전트의 특별 처리
        first_agent = self.agents[0]
        self.logger.info(f"\n{first_agent.role_name}의 차례...")
        self.logger.info(f"\n{first_agent.role_name}에게 논의의 기반 설정 요청")
        
        foundation_prompt = f"""
        당신은 {first_agent.role_name} 역할을 맡고 있는 전문가입니다.
        현재 주제는 '{topic}'입니다.
        당신은 앞으로 다른 Ai Agent들과의 논의를 통하여 주식시장에서 초과수익을 달성할 수있는 알파식을 생성해야합니다.
        현재 논의의 가장 첫번째 시작으로 제공되는 컨텍스트가 존재하지 않으니 당신은 주어진 역할과
        전문분야를 살려 다른 전문가들이 앞으로 논의를 진행할때 도움이 될 수 있는 논의의 기반을 설정해주세요.
        """
        
        foundation_response = first_agent.model_manager.generate_response(
            role=first_agent.role_name,
            speciality=f"{first_agent.speciality} - Discussion Foundation",
            role_prompt=foundation_prompt
        )
        
        context.update(foundation_response)
        first_agent_response = first_agent.analyze_and_respond(context)
        
        if "error" not in first_agent_response:
            responses.append(first_agent_response)
            if "analysis" in first_agent_response:
                previous_opinions.append(first_agent_response["analysis"])
            
            if "technical_implementation" in first_agent_response:
                evidence_collection.append({
                    "role": first_agent.role_name,
                    "evidence": first_agent_response["technical_implementation"]
                })
            
            self.log_agent_response(first_agent, first_agent_response)
        
        # 나머지 에이전트들의 처리
        for agent in self.agents[1:]:
            self.logger.info(f"\n{agent.role_name}의 차례...")
            
            agent_response = agent.analyze_and_respond(context)
            
            if "error" not in agent_response:
                responses.append(agent_response)
                if "analysis" in agent_response:
                    previous_opinions.append(agent_response["analysis"])
                
                if "technical_implementation" in agent_response:
                    evidence_collection.append({
                        "role": agent.role_name,
                        "evidence": agent_response["technical_implementation"]
                    })
                
                self.log_agent_response(agent, agent_response)
        
        # 라운드 요약 생성 (모든 필요한 인자 전달)
        round_summary = self._generate_round_summary(
            responses=responses,
            topic=topic,
            evidence_collection=evidence_collection,
            proposed_alphas=proposed_alphas,
            maintained_context=True  # 컨텍스트 유지 여부
        )
        
        return RoundResult(
            round_number=self.round_number,
            topic=topic,
            responses=responses,
            evidence_collection=evidence_collection,
            proposed_alphas=proposed_alphas,
            summary=round_summary,
            timestamp=datetime.now(),
            context_maintained=True
        )

    def _generate_agent_response(self, 
                               agent: BaseAgent, 
                               topic: str, 
                               previous_responses: List[str],
                               strategy_plan: Dict[str, Any],
                               evidence: Dict[str, Any],
                               evaluation: Dict[str, Any],
                               agent_context: Dict[str, Any]) -> str:
        """에이전트 응답 생성"""
        response_prompt = f"""
        당신은 {agent.role_name} 역할을 맡고 있습니다.
        전문분야: {agent.speciality}
        
        주제: {topic}
        
        이전 응답들:
        {json.dumps(previous_responses, ensure_ascii=False, indent=2)}
        
        당신의 전략 계획:
        {json.dumps(strategy_plan, ensure_ascii=False, indent=2)}
        
        수집된 근거:
        {json.dumps(evidence, ensure_ascii=False, indent=2)}
        
        평가 결과:
        {json.dumps(evaluation, ensure_ascii=False, indent=2)}
        
        이전 컨텍스트:
        {json.dumps(agent_context, ensure_ascii=False, indent=2)}
        
        위 정보를 바탕으로 다음을 포함하는 전문가 의견을 제시해주세요:
        1. 주요 발견사항
        2. 다른 전문가들의 의견에 대한 평가
        3. 새로운 통찰이나 제안
        4. 추가 연구가 필요한 부분
        """
        
        return self.model_manager.generate_response(
            role=agent.role_name,
            speciality=agent.speciality,
            role_prompt=response_prompt
        )
    def _generate_round_summary(
        self,
        responses: List[Dict[str, Any]],
        topic: str,
        evidence_collection: List[Dict[str, Any]],
        proposed_alphas: List[Dict[str, Any]],
        maintained_context: bool
    ) -> str:
        """라운드 요약 생성"""
        try:
            summary_prompt = f"""
            주제: {topic}
            라운드: {self.round_number}
            
            전문가들의 의견:
            {json.dumps(responses, ensure_ascii=False, indent=2)}
            
            수집된 증거:
            {json.dumps(evidence_collection, ensure_ascii=False, indent=2)}
            
            제안된 알파 전략:
            {json.dumps(proposed_alphas, ensure_ascii=False, indent=2)}
            
            위 내용을 바탕으로 이번 라운드의 주요 논점과 결론을 간단히 요약해주세요.
            """
            
            summary_response = self.model_manager.generate_response(
                role="Summarizer",
                speciality="Round Summary",
                role_prompt=summary_prompt
            )
            
            return summary_response.get("summary", "요약 생성 실패")
            
        except Exception as e:
            self.logger.error(f"라운드 요약 생성 중 오류 발생: {str(e)}")
            return f"라운드 요약 생성 실패: {str(e)}"

    def log_agent_response(self, agent: BaseAgent, response: Dict[str, Any]):
        """에이전트 응답 로깅"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"{agent.role_name}의 의견 요약:")
        
        try:
            # 응답 구조 검증
            if not isinstance(response, dict):
                self.logger.warning(f"예상치 못한 응답 형식: {type(response)}")
                return
                
            # analysis 필드 확인
            if "analysis" in response:
                analysis = response["analysis"]
                
                # 주요 포인트 로깅
                if "main_points" in analysis:
                    self.logger.info("\n주요 포인트:")
                    for point in analysis["main_points"]:
                        self.logger.info(f"- {point}")
                
                # 추천사항 로깅
                if "recommendations" in analysis:
                    self.logger.info("\n추천사항:")
                    for rec in analysis["recommendations"]:
                        self.logger.info(f"- {rec}")
                
                # 통찰 로깅
                if "insights" in analysis:
                    self.logger.info("\n통찰:")
                    for insight in analysis["insights"]:
                        self.logger.info(f"- {insight}")
            
            # technical_details 필드 확인
            if "technical_details" in response:
                tech_details = response["technical_details"]
                self.logger.info("\n기술적 세부사항:")
                self.logger.info(f"방법론: {tech_details.get('methodology', 'N/A')}")
                
                if "data_requirements" in tech_details:
                    self.logger.info("\n필요한 데이터:")
                    for req in tech_details["data_requirements"]:
                        self.logger.info(f"- {req}")
                
                if "implementation_steps" in tech_details:
                    self.logger.info("\n구현 단계:")
                    for step in tech_details["implementation_steps"]:
                        self.logger.info(f"- {step}")
            
            # 신뢰도 로깅
            if "confidence_level" in response:
                self.logger.info(f"\n신뢰도: {response['confidence_level']}")
                
            # 전체 응답 디버깅을 위해 로깅
            self.logger.debug(f"\n전체 응답:\n{json.dumps(response, ensure_ascii=False, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"응답 로깅 중 오류 발생: {str(e)}")
            self.logger.debug(f"문제가 된 응답: {response}")
        
        self.logger.info(f"\n{'='*50}\n")

