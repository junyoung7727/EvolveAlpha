from typing import List, Dict, Any, NamedTuple
import logging
from datetime import datetime
import json
import os
from src.agents.base_agent import BaseAgent
from src.agents.role_definitions import ROLE_DESCRIPTIONS

class DiscussionResult(NamedTuple):
    """토론 결과를 담는 데이터 클래스"""
    status: str
    topic: str
    history: List[Dict[str, Any]]
    timestamp: str
    error: str = None

class DiscussionManager:
    def __init__(self, agents: List[str], num_rounds: int = 3):
        """
        토론 관리자 초기화
        
        Args:
            agents (List[str]): 참여할 에이전트 역할 목록
            num_rounds (int, optional): 토론 라운드 수. Defaults to 3.
        """
        self.agent_roles = agents
        self.num_rounds = num_rounds
        self.logger = logging.getLogger("Discussion_Manager")
        self.discussion_history = []
        self.agents = []
        self._initialize_agents()

    def _initialize_agents(self):
        """에이전트 초기화"""
        try:
            self.agents = []
            for role in self.agent_roles:
                if role not in ROLE_DESCRIPTIONS:
                    self.logger.warning(f"역할 '{role}'에 대한 설명이 없습니다. 기본 설명을 사용합니다.")
                
                agent = BaseAgent(
                    role=role,
                    description=ROLE_DESCRIPTIONS[role]
                )
                self.agents.append(agent)
            self.logger.info(f"에이전트 초기화 완료: {len(self.agents)}명")
        except Exception as e:
            self.logger.error(f"에이전트 초기화 중 오류: {str(e)}")
            raise

    def run_discussion(self, topic: str) -> Dict[str, Any]:
        """
        토론 실행
        
        Args:
            topic (str): 토론 주제

        Returns:
            Dict[str, Any]: 토론 결과
        """
        try:
            self.logger.info(f"\n=== 토론 시작: {topic} ===")
            
            for round_num in range(self.num_rounds):
                round_result = self._run_round(round_num + 1, topic)
                if round_result.get("status") == "error":
                    return round_result

            return self._compile_final_result(topic)

        except Exception as e:
            error_msg = f"토론 진행 중 오류: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }

    def _run_round(self, round_num: int, topic: str) -> Dict[str, Any]:
        """
        단일 라운드 실행
        
        Args:
            round_num (int): 라운드 번호
            topic (str): 토론 주제

        Returns:
            Dict[str, Any]: 라운드 결과
        """
        try:
            self.logger.info(f"\n=== 라운드 {round_num} 시작 ===")
            round_responses = []

            for agent in self.agents:
                self.logger.info(f"\n{agent.role}의 차례...")
                
                try:
                    # 컨텍스트 구성
                    context = {
                        "round": round_num,
                        "previous_responses": round_responses,
                        "total_rounds": self.num_rounds,
                        "discussion_history": self.discussion_history
                    }

                    # 분석 수행
                    response = agent.analyze(topic, context)
                    
                    if response["status"] == "success":
                        round_responses.append({
                            "role": agent.role,
                            "content": response["content"]
                        })
                        self.logger.info(f"{agent.role}의 분석 완료")
                    else:
                        self.logger.error(f"{agent.role}의 분석 실패: {response.get('error')}")

                except Exception as e:
                    self.logger.error(f"{agent.role} 처리 중 오류: {str(e)}")
                    continue

            # 라운드 결과 저장
            round_result = {
                "round": round_num,
                "responses": round_responses,
                "timestamp": datetime.now().isoformat()
            }

            # 라운드 요약 생성 및 저장
            summary = self._create_round_summary(round_num, round_responses)
            if summary:
                round_result["summary"] = summary

            self.discussion_history.append(round_result)
            
            return {"status": "success"}

        except Exception as e:
            error_msg = f"라운드 {round_num} 실행 중 오류: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg
            }

    def _create_round_summary(self, round_num: int, responses: List[Dict[str, Any]]) -> str:
        """
        라운드 요약 생성
        
        Args:
            round_num (int): 라운드 번호
            responses (List[Dict[str, Any]]): 라운드의 응답들

        Returns:
            str: 생성된 요약 또는 None
        """
        try:
            if not responses:
                return None
                
            summarizer = BaseAgent("Moderator", "Discussion Summarizer")
            summary_context = {
                "round": round_num,
                "responses": responses
            }
            
            summary = summarizer.analyze("라운드 요약 생성", summary_context)
            
            if summary["status"] == "success":
                self.logger.info(f"\n=== 라운드 {round_num} 요약 ===\n{summary['content']}")
                return summary["content"]
                
            return None
                
        except Exception as e:
            self.logger.error(f"라운드 요약 생성 중 오류: {str(e)}")
            return None

    def _compile_final_result(self, topic: str) -> Dict[str, Any]:
        """
        최종 토론 결과 컴파일
        
        Args:
            topic (str): 토론 주제

        Returns:
            Dict[str, Any]: 최종 결과
        """
        if not self.discussion_history:
            return {
                "status": "error",
                "error": "토론 결과가 없습니다.",
                "timestamp": datetime.now().isoformat()
            }

        return {
            "status": "success",
            "topic": topic,
            "history": self.discussion_history,
            "timestamp": datetime.now().isoformat()
        }

    def save_result(self, filename: str):
        """
        토론 결과 저장
        
        Args:
            filename (str): 저장할 파일 이름
        """
        try:
            filepath = f"results/discussions/{filename}.json"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.discussion_history, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"토론 결과가 저장되었습니다: {filepath}")
            
        except Exception as e:
            self.logger.error(f"결과 저장 중 오류: {str(e)}")

    def get_discussion_history(self) -> List[Dict[str, Any]]:
        """
        토론 이력 반환
        
        Returns:
            List[Dict[str, Any]]: 전체 토론 이력
        """
        return self.discussion_history

    def _validate_round_responses(self, responses: List[Dict[str, Any]]) -> bool:
        """
        라운드 응답 유효 검사
        
        Args:
            responses (List[Dict[str, Any]]): 검사할 응답들

        Returns:
            bool: 유효성 검사 통과 여부
        """
        try:
            if not responses:
                self.logger.warning("응답이 없습니다.")
                return False

            required_fields = ["role", "content"]
            
            for response in responses:
                missing_fields = [field for field in required_fields if field not in response]
                if missing_fields:
                    self.logger.error(f"응답에 필수 필드가 누락됨: {missing_fields}")
                    return False
                
                if not response["content"]:
                    self.logger.error(f"{response['role']}의 응답 내용이 비어있습니다.")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"응답 유효성 검사 중 오류: {str(e)}")
            return False

    def _analyze_discussion_progress(self) -> Dict[str, Any]:
        """
        토론 진행 상황 분석
        
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            if not self.discussion_history:
                return {
                    "status": "error",
                    "error": "분석할 토론 이력이 없습니다."
                }

            analysis = {
                "total_rounds": len(self.discussion_history),
                "participation_rate": {},
                "response_lengths": {},
                "key_topics": set(),
                "completion_status": "진행 중"
            }

            # 참여율 및 응답 길이 분석
            for round_data in self.discussion_history:
                for response in round_data["responses"]:
                    role = response["role"]
                    analysis["participation_rate"][role] = analysis["participation_rate"].get(role, 0) + 1
                    
                    content_length = len(str(response["content"]))
                    if role not in analysis["response_lengths"]:
                        analysis["response_lengths"][role] = []
                    analysis["response_lengths"][role].append(content_length)

            # 평균 응답 길이 계산
            for role in analysis["response_lengths"]:
                lengths = analysis["response_lengths"][role]
                analysis["response_lengths"][role] = sum(lengths) / len(lengths)

            # 완료 상태 확인
            if len(self.discussion_history) >= self.num_rounds:
                analysis["completion_status"] = "완료"

            return {
                "status": "success",
                "analysis": analysis
            }

        except Exception as e:
            self.logger.error(f"토론 진행 상황 분석 중 오류: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _generate_next_round_topic(self, current_round: int) -> str:
        """
        다음 라운드 토론 주제 생성
        
        Args:
            current_round (int): 현재 라운드 번호

        Returns:
            str: 생성된 다음 라운드 주제
        """
        try:
            if not self.discussion_history:
                return "알파 전략 개발을 위한 기초 분석"

            # 이전 라운드의 요약 및 주요 포인트 수집
            previous_round = self.discussion_history[-1]
            summary = previous_round.get("summary", "")
            
            # 주제 생성을 위한 에이전트 생성
            topic_generator = BaseAgent("Topic Generator", "Discussion Topic Expert")
            
            context = {
                "current_round": current_round,
                "total_rounds": self.num_rounds,
                "previous_summary": summary,
                "discussion_history": self.discussion_history
            }
            
            response = topic_generator.analyze("다음 라운드 주제 생성", context)
            
            if response["status"] == "success":
                return response["content"]
            else:
                return f"알파 전략 개발 - 라운드 {current_round + 1}"

        except Exception as e:
            self.logger.error(f"다음 라운드 주제 생성 중 오류: {str(e)}")
            return f"알파 전략 개발 - 라운드 {current_round + 1}"

    def _create_final_summary(self) -> Dict[str, Any]:
        """
        전체 토론 최종 요약 생성
        
        Returns:
            Dict[str, Any]: 최종 요약 결과
        """
        try:
            if not self.discussion_history:
                return {
                    "status": "error",
                    "error": "요약할 토론 이력이 없습니다."
                }

            summarizer = BaseAgent("Final Summarizer", "Discussion Summary Expert")
            
            context = {
                "total_rounds": len(self.discussion_history),
                "discussion_history": self.discussion_history,
                "participant_roles": [agent.role for agent in self.agents]
            }
            
            summary = summarizer.analyze("최종 토론 요약 생성", context)
            
            if summary["status"] == "success":
                return {
                    "status": "success",
                    "summary": summary["content"]
                }
            else:
                return {
                    "status": "error",
                    "error": "최종 요약 생성 실패"
                }

        except Exception as e:
            self.logger.error(f"최종 요약 생성 중 오류: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def export_discussion_metrics(self, filename: str):
        """
        토론 지표 내보내기
        
        Args:
            filename (str): 저장할 파일 이름
        """
        try:
            analysis = self._analyze_discussion_progress()
            if analysis["status"] != "success":
                raise ValueError(analysis["error"])

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "total_rounds_completed": len(self.discussion_history),
                "total_participants": len(self.agents),
                "participation_metrics": analysis["analysis"]["participation_rate"],
                "average_response_lengths": analysis["analysis"]["response_lengths"],
                "completion_status": analysis["analysis"]["completion_status"]
            }

            filepath = f"results/metrics/{filename}_metrics.json"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"토론 지표가 저장되었습니다: {filepath}")
            
        except Exception as e:
            self.logger.error(f"지표 내보내기 중 오류: {str(e)}")

    def visualize_discussion_flow(self, save_path: str = None):
        """
        토론 흐름 시각화
        
        Args:
            save_path (str, optional): 저장할 경로. Defaults to None.
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            from matplotlib.font_manager import FontProperties
            
            # 한글 폰트 설정
            font_path = "path/to/your/korean/font.ttf"  # 한글 폰트 경로 설정 필요
            font_prop = FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()

            # 그래프 생성
            G = nx.DiGraph()
            
            # 노드와 엣지 추가
            for round_data in self.discussion_history:
                round_num = round_data["round"]
                
                # 라운드 노드 추가
                round_node = f"Round {round_num}"
                G.add_node(round_node, node_type="round")
                
                # 응답 노드 추가 및 연결
                for response in round_data["responses"]:
                    response_node = f"{response['role']} (R{round_num})"
                    G.add_node(response_node, node_type="response")
                    G.add_edge(round_node, response_node)
                    
                    # 이전 라운드의 응답과 연결
                    if round_num > 1:
                        for prev_response in self.discussion_history[round_num-2]["responses"]:
                            if prev_response["role"] == response["role"]:
                                prev_node = f"{prev_response['role']} (R{round_num-1})"
                                G.add_edge(prev_node, response_node)

            # 그래프 레이아웃 설정
            pos = nx.spring_layout(G)
            
            # 그래프 그리기
            plt.figure(figsize=(15, 10))
            
            # 노드 그리기
            nx.draw_networkx_nodes(G, pos,
                                 node_color=['lightblue' if d['node_type']=='round' else 'lightgreen' 
                                           for (n,d) in G.nodes(data=True)],
                                 node_size=2000)
            
            # 엣지 그리기
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
            
            # 레이블 그리기
            nx.draw_networkx_labels(G, pos)
            
            plt.title("토론 흐름도", fontproperties=font_prop, pad=20)
            
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"토론 흐름도가 저장되었습니다: {save_path}")
            else:
                plt.show()
                
            plt.close()

        except Exception as e:
            self.logger.error(f"토론 흐름 시각화 중 오류: {str(e)}")

    def analyze_sentiment(self) -> Dict[str, Any]:
        """
        토론 내용의 감성 분석
        
        Returns:
            Dict[str, Any]: 감성 분석 결과
        """
        try:
            sentiment_analyzer = BaseAgent("Sentiment Analyzer", "Sentiment Analysis Expert")
            
            sentiments = []
            for round_data in self.discussion_history:
                round_sentiments = []
                
                for response in round_data["responses"]:
                    analysis = sentiment_analyzer.analyze(
                        "감성 분석",
                        {
                            "content": response["content"],
                            "role": response["role"]
                        }
                    )
                    
                    if analysis["status"] == "success":
                        round_sentiments.append({
                            "role": response["role"],
                            "sentiment": analysis["content"]
                        })
                
                sentiments.append({
                    "round": round_data["round"],
                    "sentiments": round_sentiments
                })
            
            return {
                "status": "success",
                "sentiments": sentiments
            }
            
        except Exception as e:
            self.logger.error(f"감성 분석 중 오류: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def extract_key_insights(self) -> Dict[str, Any]:
        """
        주요 인사이트 추출
        
        Returns:
            Dict[str, Any]: 추출된 인사이트
        """
        try:
            insight_extractor = BaseAgent("Insight Extractor", "Key Insight Expert")
            
            insights = []
            for round_data in self.discussion_history:
                context = {
                    "round": round_data["round"],
                    "responses": round_data["responses"],
                    "summary": round_data.get("summary", "")
                }
                
                analysis = insight_extractor.analyze("주요 인사이트 추출", context)
                
                if analysis["status"] == "success":
                    insights.append({
                        "round": round_data["round"],
                        "insights": analysis["content"]
                    })
            
            return {
                "status": "success",
                "insights": insights
            }
            
        except Exception as e:
            self.logger.error(f"인사이트 추출 중 오류: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
