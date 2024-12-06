from typing import Dict, Any, List
import logging
import json
from datetime import datetime
from openai import OpenAI
import re
from src.models.model_manager import ModelManager
from src.utils.python_executor import PythonExecutor
from src.agents.role_definitions import get_role_info
from src.utils.logger_config import setup_logger
import os
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

class BaseAgent:
    def __init__(self, role: str, description: str):
        self.role = role
        self.description = description
        self.logger = logging.getLogger(f"Agent_{self.role}")
        self.llm = ModelManager()
        self.python_executor = PythonExecutor()
        self.web_search_agent = WebSearchAgent()
        self.finance_coder = FinanceCoder()
        self.analysis_history = []
        self.role_info = get_role_info(self.role)
        self.role_description = self.role_info['description']

    def analyze(self, topic: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """주제 분석 수행"""
        try:
            # 1. 분석 목적 이해
            purpose_response = self._understand_purpose(topic, context)
            if purpose_response["status"] != "success":
                return purpose_response
            purpose = purpose_response["content"]

            # 2. 실험 계획 수립
            plan_response = self._create_experiment_plan(purpose)
            if plan_response["status"] != "success":
                return plan_response
            plan = plan_response["plan"]  # 계획된 실험들
            instruct = plan_response["instruct"]  # 실험 실행에 필요한 추가 정보

            # 3. 실험 실행
            results_response = self._execute_experiments(plan)
            if results_response["status"] != "success":
                return results_response
            results = results_response["content"]

            # 4. 결과 분석
            analysis_response = self._analyze_results(results)
            if analysis_response["status"] != "success":
                return analysis_response
            analysis = analysis_response["content"]

            # 5. 결과 검증
            validation_response = self._validate_results(analysis)
            if validation_response["status"] != "success":
                return validation_response
            validated_analysis = validation_response["content"]["validated_analysis"]

            # 6. 최종 보고서 생성
            final_report_response = self._generate_final_report(validated_analysis)
            return final_report_response

        except Exception as e:
            self.logger.error(f"분석 중 오류 발생: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _understand_purpose(self, topic: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """분석 목적 이해"""
        system_prompt = f"""당신은 {self.role} 역할을 맡은 전문가입니다.
전문 분야: {self.description}"""

        analysis_prompt = f"""
주제: {topic}
{f'컨텍스트: {json.dumps(context, ensure_ascii=False)}' if context else ''}

당신의 전문성을 바탕으로 이 주제에 대한 분석 목적을 설정하고,
구체적인 연구 방향을 제시해주세요.
"""
        response = self.llm.generate_response(analysis_prompt, system_prompt)
        return response

    def _create_experiment_plan(self, purpose: str) -> Dict[str, Any]:
        """실험 계획 수립"""
        # 데이터 구조 정의
        data_json_structure = {
            "experiments": [
                {
                    "id": "EXP001",
            "title": "실험 제목",
            "hypothesis": "검증하고자 하는 구체적인 가설",
            "method": {
                "description": "실험 방법 상세 설명",
                "python_tasks": [
                    {
                        "task_description": "파이썬으로 수행할 작업 설명",
                        "code_instructions": "구체적인 코드 설계 지시 사항",
                        "expected_outputs": "기대하는 출력 결과 및 확인해야 하는 값들",
                        "validation_criteria": "결과 검증을 위한 기준",
                        "required_data": {
                            "api_sources": ["필요한 API 목록"],
                            "data_fields": ["필요한 데이터 필드들"]
                        }
                    }
                    # 추가 파이썬 작업들...
                ],
                "search_tasks": [
                    {
                        "task_description": "검색으로 수행할 작업 설명",
                        "search_objective": "검색의 구체적인 목적 또는 탐색적 검색 여부",
                        "search_type": "exploratory(탐색적) 또는 targeted(목표 지향적)",
                        "search_keywords": ["검색에 사용할 키워드들"],
                        "expected_findings": "기대하는 검색 결과 또는 확인해야 하는 정보"
                    }
                    # 추가 검색 작업들...
                ]
            },
            "expected_results": "예상되는 실험 결과",
            "success_criteria": ["실험 성공을 판단하는 기준들"],
            "estimated_duration": "예상 소요 시간",
            "dependencies": ["다른 실험 ID들 (있는 경우)"]
        }
        # 추가 실험들...
    ],
    "execution_plan": {
        "sequence": ["실험 실행 순서 (실험 ID들의 리스트)"],
        "parallel_possible": ["병렬로 실행 가능한 실험 ID들"],
        "priority": "실험들의 우선순위 설명"
    },
    "resource_requirements": {
        "apis": ["필요한 모든 API 목록"],
        "computational": "필요한 컴퓨팅 리소스 설명",
        "collaboration": ["필요한 협력 분야들"]
    }
}


        # 프롬프트 작성
        system_prompt = f"""당신은 {self.role} 역할을 맡고 있는 Ph.D 수준의 전문가입니다.
당신은 {self.role_description}
당신은 주식시장의 알파를 발견하기 위한 연구 프로젝트에 참여하고 있습니다.
당신의 전문성과 개성을 바탕으로 창의적이고 독창적인 아이디어 또는 금융시장의 움직임을 설명하기 위한 가설을 제시하고 그 가설을 실험하기 위한 실험을 제시해주세요.
"""

        plan_prompt = f"""
주어진 목적(주제): {purpose}

탐구 지침:
1. 당신만의 독특한 관점과 지식을 최대한 활용하세요.
2. 깊이 있는 사고를 통해 혁신적인 아이디어를 도출하세요.
3. 다양한 학문 분야와 아이디어를 융합하여 새로운 접근법을 제시하세요.
4. 다른 전문가들과 협력하여 시너지를 창출하세요.
5. 실패를 두려워하지 말고 과감하게 도전하세요.

제약사항:
- 모든 데이터는 외부 API를 통해 획득 (yfinance, Alpha Vantage, FRED 등)
- 룩어헤드 바이어스 방지
- 거래비용과 시장 충격 고려

추가 지침:
- JSON에서는 파이썬 코드를 직접 작성하지 마세요.
- 대신, 어떻게 구체적으로 코드를 설계하고 실행하여 어떤 결과를 확인해야 하는지에 대한 지시를 상세하고 길게 작성하세요.
- 코드 작성 지침은 'code_instructions' 필드에 포함하세요.
- 기대하는 출력 결과나 확인해야 하는 값은 'expected_outputs' 필드에 상세히 기술하세요.

먼저 아래의 구분자를 지켜 당신의 연구와 아이디어를 자유롭게 서술해주세요:

<RESPONSE>
# 연구 제목
[당신이 탐구하고자 하는 주제를 창의적으로 제시하세요]

# 연구 배경
[당신의 성격과 특색을 반영하여 이 주제를 선택한 이유와 배경을 자세히 설명하세요]

# 핵심 아이디어
[당신만의 독특한 관점에서 발견하고자 하는 알파의 원천과 그 메커니즘을 상세히 설명하세요]

# 주요 가설
[깊이 있는 사고를 통해 도출한 주요 가설을 제시하세요]

# 세부 가설
1. [세부 가설 1]
2. [세부 가설 2]
# 추가 세부 가설들...

# 검증 방법
[주요 가설과 세부 가설을 검증하기 위한 창의적인 방법론을 설명하세요.
실험 설계, 필요한 데이터, 분석 방법 등을 포함하세요]

# 예상되는 도전과제
[이 접근법에서 예상되는 어려움과 극복 방안을 논하세요]

# 협력 포인트
[다른 전문가들과 협력하여 시너지를 낼 수 있는 부분을 제시하세요]

# 추가 고려사항
[위 항목에 포함되지 않은 추가적인 생각이나 아이디어를 자유롭게 작성하세요]
</RESPONSE>

그리고 구체적인 실험 계획을 JSON 형식으로 제시해주세요:

<JSON>
{json.dumps(data_json_structure, ensure_ascii=False, indent=2)}
</JSON>

당신의 개성과 전문성을 최대한 발휘하여 깊이 있는 탐구를 진행해주세요.
자유로운 응답에서는 당신의 생각을 제한 없이 펼쳐주세요.
JSON 형식의 실험 계획에서는 가능한 한 구체적으로 작성해주세요.
필요한 만큼 여러 개의 실험과 작업을 계획할 수 있습니다.
새로운 시도를 주저하지 말고 독창적인 가설을 제시해주세요.
"""

        response = self.llm.generate_response(plan_prompt, system_prompt)
        if response["status"] == "success":
            # 응답에서 실험 계획 추출
            plan = self._extract_plan_from_response(response["content"])
            return {
                "status": "success",
                "plan": plan,
                "instruct": response["content"]
            }
        else:
            return response

    def _extract_plan_from_response(self, response_text: str) -> Dict[str, Any]:
        """응답에서 JSON 형식의 실험 계획 추출"""
        try:
            # <JSON>와 </JSON> 사이의 내용을 추출
            
            json_pattern = re.compile(r"<JSON>(.*?)</JSON>", re.DOTALL)
            json_match = json_pattern.search(response_text)
            if json_match:
                json_text = json_match.group(1).strip()
                plan = json.loads(json_text)
                return plan
            else:
                self.logger.error("실험 계획을 추출하지 못했습니다.")
                return {}
        except Exception as e:
            self.logger.error(f"실험 계획 추출 중 오류 발생: {str(e)}")
            return {}

    def _execute_experiments(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """실험 실행"""
        results = {
            "experiments": [],
            "timestamp": datetime.now().isoformat()
        }

        experiments = plan.get("experiments", [])
        execution_sequence = plan.get("execution_plan", {}).get("sequence", [])

        # 실험 ID를 키로 하는 딕셔너리 생성
        experiment_dict = {exp["id"]: exp for exp in experiments}

        # 실행 순서에 따라 실험 실행
        for exp_id in execution_sequence:
            experiment = experiment_dict.get(exp_id)
            if not experiment:
                self.logger.error(f"실험 ID {exp_id}를 찾을 수 없습니다.")
                continue

            self.logger.info(f"실험 {exp_id} 실행 중...")
            exp_result = self._execute_single_experiment(experiment)
            results["experiments"].append({
                "experiment_id": exp_id,
                "title": experiment.get("title", ""),
                "result": exp_result
            })

        return {
            "status": "success",
            "content": results
        }

    def _execute_single_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """단일 실험 실행"""
        exp_results = {
            "python_results": [],
            "search_results": []
        }

        method = experiment.get("method", {})
        # 파이썬 작업 실행
        python_tasks = method.get("python_tasks", [])
        for task in python_tasks:
            task_description = task.get("task_description", "")
            code_instructions = task.get("code_instructions", "")
            if code_instructions:
                self.logger.info(f"파이썬 작업 실행: {task_description}")
                # LLM을 사용하여 코드 생성
                experiment_output = self.finance_coder.execute_code_generation(code_instructions)
                if experiment_output["status"] == "success":
                    generated_code = experiment_output["code"]
                    experiment_result = experiment_output["result"]

                    return f"""실험 결과
<excuted code>
{generated_code}
</excuted code>

<output>
{experiment_result}
</output>
                    """

                else:
                    self.logger.error(f"코드 생성 실패: {task_description}")
                    exp_results["python_results"].append({
                        "task_description": task_description,
                        "code_instructions": code_instructions,
                        "error": code_generation_response.get("error", "코드 생성 중 알 수 없는 오류 발생")
                    })

        # 검색 작업 실행
        search_tasks = method.get("search_tasks", [])

        if search_tasks:
            try: 
                search_type = search_tasks.get("search_type", "")
                search_keywords = search_tasks.get("search_keywords", [])
                search_objective = search_tasks.get("search_objective", "")
                search_result = self.web_search_agent.search(search_objective, search_type)
                self.logger.info(f"검색 작업 실행: {task_description}")
            
                search_result = self.web_search_agent.execute_search_task(task)
                exp_results["search_results"].append({
                    "task_description": task_description,
                    "result": search_result
                })
            except Exception as e:
                self.logger.error(f"검색 작업 실행 오류: {str(e)}")

        return exp_results


    def _analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """실험 결과 분석"""
        system_prompt = f"""당신은 {self.role} 역할을 맡은 전문가입니다.
실험 결과를 분석해주세요."""

        analysis_prompt = f"""
실험 결과:
{json.dumps(results, ensure_ascii=False, indent=2)}

다음 사항들을 중점적으로 분석해주세요:
1. 주요 발견사항
2. 통계적 유의성
3. 실무적 시사점
4. 한계점
5. 추가 연구 방향
"""
        response = self.llm.generate_response(analysis_prompt, system_prompt)
        return response

    def _validate_results(self, analysis_response: Dict[str, Any]) -> Dict[str, Any]:
        """결과 검증"""
        try:
            analysis_content = analysis_response
            # 필수 항목 체크
            required_fields = ["주요 발견사항", "실무적 시사점", "한계점"]
            missing_fields = [field for field in required_fields if field not in analysis_content]

            if missing_fields:
                return {
                    "status": "error",
                    "error": f"필수 항목 누락: {', '.join(missing_fields)}",
                    "timestamp": datetime.now().isoformat()
                }

            # 검증 성공
            return {
                "status": "success",
                "content": {
                    "is_valid": True,
                    "validated_analysis": analysis_content,
                    "confidence_level": "high"
                }
            }

        except Exception as e:
            self.logger.error(f"결과 검증 중 오류 발생: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _generate_final_report(self, analysis: str) -> Dict[str, Any]:
        """최종 보고서 생성"""
        system_prompt = f"""당신은 {self.role} 역할을 맡은 전문가입니다.
분석 결과를 종합하여 최종 보고서를 작성해주세요."""

        report_prompt = f"""
분석 결과:
{analysis}

다음 형식으로 최종 보고서를 작성해주세요:

# 연구 제목
# 연구 배경
# 연구 방법
# 주요 발견사항
# 시사점
# 한계점 및 향후 연구방향
"""
        response = self.llm.generate_response(report_prompt, system_prompt)

        if response["status"] == "success":
            self.analysis_history.append({
                "type": "final_report",
                "content": response["content"],
                "timestamp": datetime.now().isoformat()
            })

        return response

    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """분석 이력 반환"""
        return self.analysis_history

class WebSearchAgent:
    def __init__(self):
        self.logger = logging.getLogger("Web_Search")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.serper_api_key = os.getenv("SERPER_API_KEY")

    def search(self, query: str, search_type: str = "tavily") -> Dict[str, Any]:
        """웹 검색 수행"""
        try:
            if search_type == "objective":
                return self._tavily_search(query)
            elif search_type == "exploratory":
                return self._serper_search(query)
            else:
                raise ValueError(f"지원하지 않는 검색 타입: {search_type}")
        except Exception as e:
            self.logger.error(f"검색 중 오류 발생: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _tavily_search(self, query: str) -> Dict[str, Any]:
        """Tavily API를 사용한 검색"""
        try:
            url = "https://api.tavily.com/search"
            headers = {"api-key": self.tavily_api_key}
            params = {
                "query": query,
                "include_images": False,
                "include_answer": True
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            return {
                "status": "success",
                "results": response.json()
            }
            
        except Exception as e:
            self.logger.error(f"Tavily 검색 오류: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _serper_search(self, query: str) -> Dict[str, Any]:
        """Serper API를 사용한 검색"""
        try:
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": self.serper_api_key,
                "Content-Type": "application/json"
            }
            data = {"q": query}
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            return {
                "status": "success",
                "results": response.json()
            }
            
        except Exception as e:
            self.logger.error(f"Serper 검색 오류: {str(e)}")
            raise

    def extract_hypotheses(text: str) -> dict:
        """주요 가설과 세부 가설을 추출하여 딕셔너리로 반환"""
        try:
            # 정규식 패턴
            main_pattern = r'#\s*주요\s*가설\s*\n"([^"]+)"'
            sub_pattern = r'#\s*세부\s*가설\s*\n((?:\d+\.\s*[^\n]+\n?)+)'
            
            # 주요 가설 추출
            main_match = re.search(main_pattern, text)
            main_hypothesis = main_match.group(1).strip() if main_match else ""
            
            # 세부 가설 추출
            sub_match = re.search(sub_pattern, text)
            if sub_match:
                # 각 줄을 분리하고 번호를 제거하여 리스트로 만듦
                sub_hypotheses = [
                    re.sub(r'^\d+\.\s*', '', line.strip())
                    for line in sub_match.group(1).strip().split('\n')
                    if line.strip()
                ]
            else:
                sub_hypotheses = []
                
            return {
                "main_hypothesis": main_hypothesis,
                "sub_hypotheses": sub_hypotheses
            }
            
        except Exception as e:
            logging.error(f"가설 추출 중 오류 발생: {str(e)}")
            return {
                "main_hypothesis": "",
                "sub_hypotheses": []
            }
class FinanceCoder:
    """
    FinanceCoder 클래스는 사용자의 지시에 따라 고품질의 데이터 분석 코드를 작성하는 역할을 합니다.
    Chain of Thought 사고 프롬프트 구조를 사용하여 GPT 모델의 응답 품질을 향상시킵니다.
    """
    def __init__(self):
        self.logger = setup_logger("Finance_Coder")
        self.model_manager = ModelManager()
        self.client = OpenAI()
        self.python_executor = PythonExecutor()

    def execute_code_generation(self, query: str) -> Dict[str, Any]:
        """사용자의 요청에 따라 코드 생성 및 실행"""
        try:
            # 1. 코드 생성 목적 이해
            purpose = self._understand_code_purpose(query)
            if "error" in purpose:
                raise ValueError(f"목적 이해 실패: {purpose['error']}")

            # 2. 코드 생성 계획 수립
            code_plan = self._create_code_plan(purpose)
            if "error" in code_plan:
                raise ValueError(f"코드 계획 수립 실패: {code_plan['error']}")

            # 3. 코드 생성 및 실행 (최대 5회 시도)
            code = ""
            result = {}
            for attempt in range(5):
                self.logger.info(f"\n=== 코드 생성 시도 {attempt + 1} ===")

                # 코드 생성
                code_response = self._generate_code(code_plan)
                if "error" in code_response:
                    self.logger.error(f"코드 생성 오류: {code_response['error']}")
                    continue

                code = code_response.get("code", "")
                if not code:
                    self.logger.error("생성된 코드가 비어있습니다.")
                    continue

                # 코드 실행
                result_response = self._run_code(code)
                if "error" not in result_response:
                    result = result_response.get("results", {})
                    break
                else:
                    self.logger.error(f"코드 실행 오류: {result_response['error']}")

            if not result:
                raise ValueError("코드 실행에 실패했습니다.")
            
            # 6. 최종 결과 반환
            return {
                "status": "success",
                "code": code,
                "analysis": result_response,
            }

        except Exception as e:
            self.logger.error(f"코드 실행 중 오류 발생: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _understand_code_purpose(self, query: str) -> Dict[str, Any]:
        """코드 생성 목적 이해"""
        system_prompt = "당신은 금융 데이터 분석에 특화된 고품질의 코드를 작성하는 전문가입니다."
        user_prompt = f"""
사용자의 요청: {query}

이 요청에 따라 코드가 달성해야 할 구체적인 목표와 요구사항을 분석해주세요.

- 목표:
- 요구사항:
- 필요한 데이터:
- 사용해야 할 라이브러리:
"""

        # Chain of Thought를 유도하기 위해 프롬프트에 생각 과정 포함
        response = self.model_manager.generate_response(
            user_prompt,
            system_prompt
        )
        return response

    def _create_code_plan(self, purpose: Dict[str, Any]) -> Dict[str, Any]:
        """코드 생성 계획 수립"""
        system_prompt = "당신은 금융 데이터 분석에 특화된 고품질의 코드를 작성하는 전문가입니다."
        user_prompt = f"""
목적 분석 결과: {json.dumps(purpose, ensure_ascii=False, indent=2)}

이 목적을 달성하기 위한 구체적인 코드 구현 계획을 수립해주세요.

- 전체적인 접근 방식:
- 주요 단계:
- 필요한 함수와 클래스:
- 데이터 흐름:
- 예외 처리 및 오류 검증 방안:
"""

        response = self.model_manager.generate_response(
            user_prompt,
            system_prompt
        )
        return response

    def _generate_code(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """코드 생성"""
        system_prompt = "당신은 금융 데이터 분석에 특화된 고품질의 파이썬 코드를 작성하는 전문가입니다."
        user_prompt = f"""
코드 구현 계획: {json.dumps(plan, ensure_ascii=False, indent=2)}

이 계획에 따라 실행 가능한 파이썬 코드를 생성해주세요.

- 코드는 명확하고 효율적이어야 합니다.
- 적절한 주석을 포함해야 합니다.
- 함수와 클래스의 이름은 직관적으로 작성하세요.
- 코드의 가독성을 높이기 위해 일관된 코딩 스타일을 유지하세요.
"""

        response = self.model_manager.generate_response(
            user_prompt,
            system_prompt
        )
        return response

    def _run_code(self, code: str) -> Dict[str, Any]:
        """코드 실행"""
        try:
            # PythonExecutor를 사용하여 코드 실행
            execution_result = self.python_executor.execute_code(code)

            if execution_result["status"] == "error":
                return {"error": execution_result.get("error", {}).get("message", "Unknown error")}

            # 실행 결과 분석 및 정리
            analysis = self._analyze_execution_result(execution_result)

            # GPT에 전달할 프롬프트 생성
            prompt = self._create_analysis_prompt(analysis)

            # GPT를 통한 결과 해석
            interpretation = self._get_gpt_interpretation(prompt)

            return {
                "results": analysis,
                "interpretation": interpretation
            }

        except Exception as e:
            return {"error": str(e)}

    def _get_gpt_interpretation(self, prompt: str) -> Dict[str, Any]:
        """GPT를 사용한 결과 해석"""
        system_prompt = "당신은 금융 데이터 분석 전문가입니다."
        user_prompt = prompt

        response = self.model_manager.generate_response(
            user_prompt,
            system_prompt
        )
        return response

    def _analyze_execution_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """실행 결과 분석 및 정리"""
        analysis = {
            "execution_status": result["status"],
            "timestamp": result["timestamp"],
            "data_summary": [],
            "plots": [],
            "statistics": {},
            "outputs": {}
        }

        # DataFrame 분석
        for df_info in result.get("dataframes", []):
            df_summary = {
                "name": df_info["name"],
                "shape": df_info["shape"],
                "description": df_info["description"]
            }
            analysis["data_summary"].append(df_summary)

        # 플롯 처리
        for plot in result.get("plots", []):
            plot_info = {
                "figure_number": plot["figure_number"],
                "image_data": plot["data"]  # base64 인코딩된 이미지
            }
            analysis["plots"].append(plot_info)

        # 기타 출력 처리
        for var_name, output in result.get("outputs", {}).items():
            analysis["outputs"][var_name] = {
                "type": output["type"],
                "shape": output.get("shape"),
                "summary": self._summarize_output(output["value"])
            }

        # 표준 출력/에러 포함
        if result.get("stdout"):
            analysis["stdout"] = result["stdout"]
        if result.get("stderr"):
            analysis["stderr"] = result["stderr"]

        return analysis

    def _summarize_output(self, value: Any) -> Dict[str, Any]:
        """출력값 요약"""
        if isinstance(value, (list, tuple)):
            return {
                "length": len(value),
                "sample": value[:5] if len(value) > 5 else value,
                "type": "sequence"
            }
        elif isinstance(value, dict):
            return {
                "keys": list(value.keys()),
                "length": len(value),
                "type": "dictionary"
            }
        else:
            return {
                "value": str(value),
                "type": str(type(value))
            }

    def _create_analysis_prompt(self, analysis: Dict[str, Any]) -> str:
        """GPT에 전달할 분석 프롬프트 생성"""
        prompt = f"""
금융 데이터 분석 결과를 검토하고 해석해주세요.

실행 결과 요약:
- 실행 상태: {analysis["execution_status"]}
- 실행 시간: {analysis["timestamp"]}

데이터 분석 요약:
{json.dumps(analysis["data_summary"], ensure_ascii=False, indent=2)}

생성된 그래프 개수: {len(analysis["plots"])}

주요 통계:
{json.dumps(analysis["statistics"], ensure_ascii=False, indent=2)}

추가 출력:
{json.dumps(analysis["outputs"], ensure_ascii=False, indent=2)}

다음 사항들을 중점적으로 분석해주세요:
1. 데이터의 주요 특징과 패턴
2. 그래프가 보여주는 중요한 인사이트
3. 통계적으로 유의미한 결과
4. 투자 전략에 활용할 수 있는 시사점
5. 추가 분석이 필요한 부분

응답은 마크다운 형식으로 작성해주세요.
"""

        return prompt

