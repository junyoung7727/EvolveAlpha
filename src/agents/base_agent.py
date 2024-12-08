from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime
import re
import os
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from src.models.model_manager import ModelManager
from src.utils.python_executor import PythonExecutor
from src.agents.role_definitions import get_role_info
from src.utils.logger_config import setup_logger
from openai import OpenAI
from tavily import TavilyClient


class CodePromptTemplate:
    """
    FinanceCoder가 코드를 생성할 때 사용할 긴 컨텍스트 템플릿.
    - 여기에 terminal 실행, DB 저장, 웹 검색 결과 활용 등을 장려하는 지시사항 추가
    - JSON 형식, <PYTHON></PYTHON> 태그 사용 강조
    """
    def __init__(self):
        self.base_system_prompt = """당신은 금융 데이터 분석에 특화된 고품질 파이썬 코드 작성 전문가입니다.
목표:
- 사용자 요청에 따라 외부 API(yfinance, FRED 등)를 활용해 풍부한 결과 생성
- S&P500 지수, 비트코인 시계열 데이터, 코스피 지수 중 하나를 선택하여 가격 데이터 활용 시 명시적으로 특정 대상 언급
- 필요 시 터미널 명령 (예: pip install), 웹 검색 결과 활용
- 결과 출력 시 최대한 많은 print, 다양한 DataFrame, 그래프, 통계치 생성
- LLM 응답은 JSON 형식 또는 <PYTHON></PYTHON> 블록 사용을 권장. 파싱 오류 없게 주의.
- env 파일의 API_KEY들(HF_TOKEN, PINECONE_API_KEY 등) 적극 활용 가능
Chain-of-Thought는 내부적으로만 하고, 최종 코드만 제공."""

        self.user_instructions_header = "아래는 사용자 code_instructions, 목적/계획 정보입니다.\n"
        self.requirements = """요구사항:
- code는 독립 실행 가능
- 풍부한 결과물(DataFrame, 그래프, 통계치, 텍스트 설명)
- yfinance로 특정 자산 데이터 명시적 획득(S&P 500 상위 기업 중 하나, 혹은 비트코인, 코스피 지수 등)
- DB(예: Neo4j, Pinecone) 연동 로직(정보 저장/조회), terminal 명령 실행 코드(예: subprocess로 pip install) 예시 포함 가능
- 웹 검색 결과를 받아 전처리 후 DB 저장 가능

-현재 env파일에 존재하는 API_KEY목록
-env 파일을 사용할 때는 python-dotenv와 load_dotenv 모듈 사용
api_list = [
    "HF_TOKEN",  # Hugging Face
    "PINECONE_API_KEY",  # Pinecone
    "TAVILY_API_KEY",  # Tavily
    "IEX_CLOUD_API_KEY",  # IEX Cloud
    "SEEKING_ALPHA_API_KEY",  # Seeking Alpha
    "NEWS_API_KEY",  # NewsAPI
    "OPENAI_API_KEY",  # OpenAI
    "SERPER_API_KEY",  # Serper
    "ALPHA_VANTAGE_API_KEY",  # Alpha Vantage
    "X_API_KEY",  # Custom API Key
    "X_API_KEY_SECRET",  # Custom API Key Secret
    "X_API_BEARER_TOKEN",  # Custom API Bearer Token
    "X_API_ACCESS_TOKEN",  # Custom API Access Token
    "X_API_ACCESS_TOKEN_SECRET",  # Custom API Access Token Secret
    "X_API_CLIENT_ID",  # Custom API Client ID
    "X_API_CLIENT_SECRET",  # Custom API Client Secret
    "REDDIT_CLIENT_ID",  # Reddit
    "REDDIT_CLIENT_SECRET",  # Reddit
    "QUANDL_API_KEY",  # Quandl
    "DART_API_KEY",  # DART
    "BANK_OF_KOREA_API_KEY",  # Bank of Korea
    "FRED_API_KEY"  # FRED
]

****출력결과에 파이썬 코드가 포함된다면 반드시 <PYTHON> </PYTHON> 구분자를 사용해 출력할 내용과 코드 구분하세요.
마지막으로 <PYTHON> </PYTHON>구분자 안에있는 코드는 문자열 그대로 컴파일러에게 전달되므로 코드를 제외한 다른 문자를 넣는다면 감점됩니다."""

    def build_prompt(self, purpose: Dict[str, Any], code_plan: Dict[str, Any], user_query: str) -> str:
        purpose_str = json.dumps(purpose, ensure_ascii=False, indent=2)
        code_plan_str = json.dumps(code_plan, ensure_ascii=False, indent=2)
        prompt = f"""
{self.user_instructions_header}
사용자 요청 (code_instructions): {user_query}

목적 분석 결과:
{purpose_str}

코드 구현 계획:
{code_plan_str}

{self.requirements}

데이터 수집 시 명확히 특정 자산(SP500 대기업 주가 or 비트코인 or 코스피) 선택해서 사용.
DB에 정보 저장/조회, terminal 명령어 실행, 웹 검색 결과 활용, 가능한 많은 출력 및 해석적 텍스트 제공.
****출력결과에 파이썬 코드가 포함된다면 반드시 <PYTHON> </PYTHON> 구분자를 사용해 출력할 내용과 코드 구분하세요
"""
        return prompt

class BaseAgent:
    """
    기존 로직 유지.
    """
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
        self.system_prompt = f"""당신은 {self.role} 역할을 맡고 있는 Ph.D 수준의 전문가입니다.
당신은 {self.role_description}
당신은 주식시장의 모멘텀 알파를 발견하기 위한 연구 프로젝트에 참여하고 있습니다.
당신의 전문성과 개성을 바탕으로 창의적이고 독창적인 아이디어, 특정 자산 선택(S&P500 지수 혹은 비트코인, 코스피) 및 외부 API 활용, DB 저장 등을 포함한 실험 계획을 제시해주세요.
"""

    def analyze(self, topic: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            # 1. 분석 목적 이해
            purpose_response = self._understand_purpose(topic, context)
            if purpose_response["status"] != "success":
                return purpose_response
            purpose = purpose_response["content"]

            pre_search_response = self.pre_search(topic)

            experiment_plan_response = self._create_plan_prompt(purpose,pre_search_response)
            # 2. 실험 계획 수립
            plan_response = self._create_experiment_plan(purpose,experiment_plan_response)
            if plan_response["status"] != "success":
                return plan_response
            plan = plan_response["plan"]
            instruct = plan_response["instruct"]

            # 3. 실험 실행
            results_response = self._execute_experiments(plan,instruct)
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

            # 여기서 DB에서 추가 정보 읽기(가상), 혹은 이전에 저장한 웹 검색 결과를 불러와 종합도 가능
            # 실제 구현 시 DB 연동 코드 필요
            # validated_analysis에 DB 정보 결합 (개념적)
            # validated_analysis["db_info"] = self._fetch_from_knowledge_base() # 가상의 메서드

            # 6. 최종 보고서 생성
            final_report_response = self._generate_final_report(validated_analysis)

            # 마지막 최종 반환 전 DB 저장/조회(개념), 종합 의견 출력
            return final_report_response

        except Exception as e:
            self.logger.error(f"분석 중 오류 발생: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _understand_purpose(self, topic: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        system_prompt = f"""당신은 {self.role} 역할을 맡은 전문가입니다.
전문 분야: {self.description}"""

        context_str = f'컨텍스트: {json.dumps(context, ensure_ascii=False)}' if context else ''
        analysis_prompt = f"""
주제: {topic}
{context_str}

당신의 전문성을 바탕으로 이 주제에 대한 분석 목적을 설정하고,
구체적인 연구 방향을 제시해주세요.
"""
        response = self.llm.generate_response(analysis_prompt, system_prompt)
        return response
    
    def pre_search(self, topic: str) -> Dict[str, Any]:
        search_prompt = f"""
주제: {topic}, 목적:당신은 {self.role} 역할을 맡은 전문가입니다. 전문성을 바탕으로 주제에 대한 인터넷 사전 조사를 진행해주세요.
"""
        response = self.web_search_agent.search(search_prompt)
        return response


    def _create_experiment_plan(self, purpose: str, pre_plan: str) -> Dict[str, Any]:
        data_json_structure = {
            "experiments": [
                {
                    "id": "EXP001",
                    "title": "특정 자산 기반 시장 분석 실험",
                    "hypothesis": "S&P500 대기업 주가(또는 비트코인/코스피)와 거시경제 지표 간 상관관계로 알파를 발견할 수 있다",
                    "method": {
                        "description": "외부 API(yfinance 등)로 특정 자산 가격(예:S&P500 상위 기업 주가)과 FRED 지표 수집, 회귀분석, 그래프생성",
                        "python_tasks": [
                            {
                                "task_description": "외부 API 활용 데이터를 받아 다양한 통계 및 시각화 결과를 생성. DB에 지식 저장. 웹 검색 결과 활용.",
                                "code_instructions": (
                                    "JSON 혹은 <PYTHON></PYTHON> 구분자를 사용. "
                                    "yfinance로 S&P500 주요 기업(예: AAPL) 주가 데이터 수집, FRED API로 금리 등 거시지표 수집. "
                                    "터미널 명령어로 필요 라이브러리 설치(subprocess 사용), "
                                    "웹 검색 결과(WebSearchAgent로 미리 검색했다고 가정) 전처리 후 DB(Neo4j or Pinecone)에 저장. "
                                    "다양한 그래프(시계열, 상관 히트맵) 생성 후 base64 인코딩, "
                                    "통계치 계산 및 print로 상세 출력. "
                                    "최종적으로 풍부한 해석적 텍스트 출력."
                                ),
                                "expected_outputs": "DataFrame 다수, 상관계수, 회귀분석 결과, base64 그래프, DB 저장/조회 결과, 텍스트 해석",
                                "validation_criteria": "요구한 모든 결과를 정확히 출력",
                                "required_data": {
                                    "api_sources": ["yfinance", "FRED API"],
                                    "data_fields": ["종가 시계열 데이터", "거래량 데이터", "금리, GDP 등"]
                                }
                            }
                        ],
                        "search_tasks": []
                    },
                    "expected_results": "기대하는 결과",
                    "success_criteria": ["풍부한 결과물", "알파 시사점"],
                    "dependencies": []
                }
            ],
            "execution_plan": {
                "sequence": ["EXP001"],
                "parallel_possible": [],
                "priority": "EXP001 먼저 실행"
            },
            "resource_requirements": {
                "apis": ["yfinance", "FRED", "Neo4j", "Pinecone", "web"],
                "computational": "적당한 메모리/CPU",
                "collaboration": ["데이터 엔지니어", "경제학자"]
            }
        }

        user_prompt = f"""
분석 목적: 
{purpose}
분석 계획서: 
{pre_plan}

위의 분석 목적과 분석 계획서를 참고하여 아래의 json 형식에 맞게 실험 계획을 작성해주세요. 응답 구분자 내에는 json 형식이외의 다른 문자열이 들어가면 감점됩니다.

응답할 json 형식:
<JSON>
{json.dumps(data_json_structure, ensure_ascii=False, indent=2)}
</JSON>
        """
        response = self.llm.generate_response(user_prompt, self.system_prompt)
        if response["status"] == "success":
            plan = self._extract_plan_from_response(response["content"])
            return {
                "status": "success",
                "plan": plan,
                "instruct": response["content"]
            }
        else:
            return response

    def _create_plan_prompt(self, purpose: str, pre_search: str) -> str:


        plan_prompt = f"""
주어진 목적(주제): {purpose}

인터넷 사전 조사 자료:
{pre_search}

탐구 지침:
- 특정 자산 명시적으로 선택할 것 (예: S&P500 대기업 AAPL 주가)
- 외부 API(yfinance, FRED) 및 env 파일 내 API 키 활용 가능
- 데이터 분석, 그래프 생성, DB 저장/조회, 웹 검색 활용

<RESPONSE>
# 연구 제목
[특정 자산 기반 알파 발견 연구]

# 연구 배경
[...]

# 핵심 아이디어
[...]

# 주요 가설
[...]

# 세부 가설
1. [...]
2. [...]

# 검증 방법
[...]

# 예상되는 도전과제
[...]

# 협력 포인트
[...]

# 추가 고려사항
[...]
</RESPONSE>
"""
        response = self.llm.generate_response(plan_prompt, self.system_prompt)
        return response

    def _extract_plan_from_response(self, response_text: str) -> Dict[str, Any]:
        try:
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
            self.logger.error(f"실험 계획 추출 중 오류 발생: {str(e)}", exc_info=True)
            return {}


    def _execute_experiments(self, plan: Dict[str, Any],instruct:str) -> Dict[str, Any]:
        results = {
            "experiments": [],
            "timestamp": datetime.now().isoformat()
        }

        experiments = plan.get("experiments", [])
        execution_sequence = plan.get("execution_plan", {}).get("sequence", [])
        experiment_dict = {exp["id"]: exp for exp in experiments}

        for exp_id in execution_sequence:
            experiment = experiment_dict.get(exp_id)
            if not experiment:
                self.logger.error(f"실험 ID {exp_id}를 찾을 수 없습니다.")
                continue

            self.logger.info(f"실험 {exp_id} 실행 중...")
            exp_result = self._execute_single_experiment(experiment,instruct)
            results["experiments"].append({
                "experiment_id": exp_id,
                "title": experiment.get("title", ""),
                "result": exp_result
            })

        return {
            "status": "success",
            "content": results
        }

    def _execute_single_experiment(self, experiment: Dict[str, Any],instruct:str) -> Dict[str, Any]:
        exp_results = {
            "python_results": [],
            "search_results": []
        }

        method = experiment.get("method", {})
        python_tasks = method.get("python_tasks", [])
        for task in python_tasks:
            task_description = task.get("task_description", "")
            code_instructions = task.get("code_instructions", "")
            if code_instructions:
                self.logger.info(f"파이썬 작업 실행: {task_description}")
                experiment_output = self.finance_coder.execute_code_generation(instruct)
                if experiment_output["status"] == "success":
                    code = experiment_output["code"]
                    analysis = experiment_output["analysis"]
                    exp_results["python_results"].append({
                        "task_description": task_description,
                        "executed_code": code,
                        "analysis": analysis
                    })
                else:
                    self.logger.error(f"코드 생성 실패: {task_description}")
                    exp_results["python_results"].append({
                        "task_description": task_description,
                        "code_instructions": code_instructions,
                        "error": experiment_output.get("error", "코드 생성 중 알 수 없는 오류 발생")
                    })

        return exp_results

    def _analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = f"""당신은 {self.role} 역할을 맡은 전문가입니다.
실험 실행 결과로, 다양한 결과물(데이터프레임, 그래프, 통계치, DB에 저장한 정보, 웹 검색 정보)가 있습니다.
이들을 종합적으로 해석하고, 알파 발견 가능성을 평가해주세요."""

        analysis_prompt = f"""
실험 결과:
{json.dumps(results, ensure_ascii=False, indent=2)}

분석 지침:
1. DataFrame 패턴 분석
2. 그래프 해석(시계열, 히트맵)
3. 통계치(상관계수, 회귀 계수) 유의성 평가
4. 거시지표와 자산 가격 관계 해석
5. DB 저장 정보(가정) 재활용
6. 알파 가능성 제안
7. 한계점, 추가 연구 제안

결과는 JSON 형식으로 "주요 발견사항", "실무적 시사점", "한계점" 등을 포함하여 반환.
"""
        response = self.llm.generate_response(analysis_prompt, system_prompt)
        return response

    def _validate_results(self, analysis_response: Dict[str, Any]) -> Dict[str, Any]:
        try:
            analysis_content = analysis_response
            required_fields = ["주요 발견사항", "실무적 시사점", "한계점"]
            missing_fields = [field for field in required_fields if field not in analysis_content]

            if missing_fields:
                return {
                    "status": "error",
                    "error": f"필수 항목 누락: {', '.join(missing_fields)}",
                    "timestamp": datetime.now().isoformat()
                }

            return {
                "status": "success",
                "content": {
                    "is_valid": True,
                    "validated_analysis": analysis_content,
                    "confidence_level": "high"
                }
            }

        except Exception as e:
            self.logger.error(f"결과 검증 중 오류 발생: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    def _generate_final_report(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = f"""당신은 {self.role} 역할을 맡은 전문가입니다.
분석 결과를 종합하여 최종 의견을 작성해주세요. 당신의 의견은 다른 전문가들에게 제시되어 더 발전된 알파전략에 대한 논의가 이어질 것입니다.
(DataFrame, 그래프, 통계분석, DB 정보를 활용하여 알파 발굴 가능성에 대한 결론 제시.)
"""

        report_prompt = f"""
분석 결과:
{json.dumps(analysis, ensure_ascii=False, indent=2)}
"""
        
        response = self.llm.generate_response(report_prompt, system_prompt)
        if response["status"] == "success":
            final_report = response["content"]
            return {
                "status": "success",
                "content": final_report
            }
        else:
            return response
class WebSearchAgent:
    def __init__(self):
        self.logger = logging.getLogger("Web_Search")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.serper_api_key = os.getenv("SERPER_API_KEY")

    def search(self, query: str, search_type: str = "tavily") -> Dict[str, Any]:
        """웹 검색 수행"""
        try:
            # if search_type == "objective":
            #     return self._tavily_search(query)
            # elif search_type == "exploratory":
            #     return self._serper_search(query)
            return {"tavily":self._tavily_search(query),
                   "serper":self._serper_search(query),
                   }
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
            client = TavilyClient(api_key=self.tavily_api_key)
            response = client.search(
                query=query,
                include_images=False,
                include_answer=True
            )
            
            return {
                "status": "success",
                "results": response
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
    def __init__(self):
        self.logger = setup_logger("Finance_Coder")
        self.model_manager = ModelManager()
        self.python_executor = PythonExecutor()
        self.code_template = CodePromptTemplate()

    def execute_code_generation(self, query: str) -> Dict[str, Any]:
        """
        코드 생성 및 실행을 시도하고, 오류 발생 시 최대 5번까지 수정하여 재시도합니다.
        """
        try:
            purpose = self._understand_code_purpose(query)
            if "error" in purpose:
                raise ValueError(f"목적 이해 실패: {purpose['error']}")
            code_plan = self._create_code_plan(purpose)
            if "error" in code_plan:
                raise ValueError(f"코드 계획 수립 실패: {code_plan['error']}")

            code = ""
            result = {}
            max_attempts = 5

            for attempt in range(1, max_attempts + 1):
                self.logger.info(f"\n=== 코드 생성 시도 {attempt} ===")
                system_prompt = self.code_template.base_system_prompt
                user_prompt = self.code_template.build_prompt(purpose, code_plan, query)

                code_response = self.model_manager.generate_response(user_prompt, system_prompt)
                if "error" in code_response or code_response["status"] != "success":
                    self.logger.error(f"코드 생성 오류: {code_response.get('stderr', 'Unknown error')}")
                    continue

                code = self._extract_code_from_response(code_response.get("content", ""))
                if not code:
                    self.logger.error("생성된 코드가 비어있거나 오류를 포함하고 있음")
                    continue

                result_response = self.python_executor.execute_code(code)
                if "error" not in result_response:
                    result = result_response
                    self.logger.info("코드 실행 성공")
                    break
                else:
                    self.logger.error(f"코드 실행 오류: {result_response['error']}")
                    # 오류 메시지를 바탕으로 코드 수정 시도
                    error_message = result_response.get("stderr", "Unknown error")
                    code = self._modify_code_based_on_error(code, error_message)
                    print("--------------------------------")
                    print(code)
                    print("--------------------------------")
                    if not code:
                        self.logger.error("오류 기반 코드 수정 실패")
                        continue
                    if code:
                        code_response = self.python_executor.execute_code(code)
                        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                        print(code_response)
                        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                        if "error" not in code_response:
                            result = code_response
                            self.logger.info("코드 실행 성공")
                            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                            print(result)
                            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                            break

                        

            if not result:
                raise ValueError("코드 실행에 반복해서 실패하였습니다.")

            return {
                "status": "success",
                "code": code,
                "analysis": result
            }

        except Exception as e:
            self.logger.error(f"코드 실행 중 오류 발생: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _understand_code_purpose(self, query: str) -> Dict[str, Any]:
        system_prompt = "당신은 금융 데이터 분석을 위한 고품질 코드를 작성하는 전문가입니다."
        user_prompt = f"""
사용자 요청: {query}

요청 분석:
- 어떤 외부 API를 활용할 수 있는가?
- 어떤 데이터와 결과를 출력해야 하는가?
- 어떤 그래프나 통계치가 요구되는가?
- 명확한 데이터 대상(SP500 대기업, 비트코인, 코스피 등) 선정 필요
"""
        response = self.model_manager.generate_response(user_prompt, system_prompt)
        return response

    def _create_code_plan(self, purpose: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = "당신은 금융 데이터 분석을 위한 고품질 코드를 작성하는 전문가입니다."
        user_prompt = f"""
목적 분석 결과: {json.dumps(purpose, ensure_ascii=False, indent=2)}

이 목적을 달성하기 위한 코드 구현 계획을 세우세요:
- 외부 API(yfinance, FRED 등) 데이터 수집
- 특정 자산(S&P500 대기업 주가 or 비트코인 or 코스피) 명확히 선택
- 데이터 전처리 및 특징 추출
- 다양한 통계량 산출
- 다양한 그래프 생성
- DB 저장/조회, terminal 명령 실행, 웹 검색 활용 가능성 제시
- print 통한 풍부한 결과 출력
계획을 상세히 기술
"""
        response = self.model_manager.generate_response(user_prompt, system_prompt)
        return response


    def _modify_code_based_on_error(self, code: str, error_message: str) -> Optional[str]:
        """
        오류 메시지를 기반으로 코드를 수정하도록 모델에 요청합니다.
        """
        try:
            self.logger.info("오류 기반 코드 수정을 시도합니다.")
            system_prompt = "당신은 발생한 오류를 바탕으로 코드를 수정하는 전문가입니다. 오류 메시지를 참고하여 코드를 수정하세요. 또한 원래코드의 기능을 잃지 않으면서도 항상 고품질의 코드를 쓸수록 점수를 높게 받습니다."
            user_prompt = f"""
다음은 실행 중 발생한 오류 메시지입니다:

{error_message}

아래의 파이썬 코드에서 오류를 수정하고, 오류가 발생하지 않도록 개선된 코드를 제공하세요.

<PYTHON>
{code}
</PYTHON>

응답은 아래의 형식에 맞춰 작성해주세요.

<PYTHON>
...
</PYTHON>
"""

            response = self.model_manager.generate_response(user_prompt, system_prompt)
            if "error" in response or response["status"] != "success":
                self.logger.error(f"코드 수정 요청 중 오류 발생: {response.get('error', 'Unknown error')}")
                return None

            modified_code = self._extract_code_from_response(response.get("content", ""))
            if not modified_code:
                self.logger.error("수정된 코드를 추출하지 못했습니다.")
                return None

            self.logger.info("수정된 코드를 성공적으로 추출했습니다.")
            return modified_code

        except Exception as e:
            self.logger.error(f"코드 수정 중 오류 발생: {str(e)}", exc_info=True)
            return None

    def _extract_code_from_response(self, response_text: str) -> Optional[str]:
        """
        응답 텍스트에서 <PYTHON></PYTHON> 태그 사이의 코드를 추출합니다.
        """
        try:
            code_pattern = re.compile(r"<PYTHON>(.*?)</PYTHON>", re.DOTALL)
            code_match = code_pattern.search(response_text)
            if code_match:
                code_text = code_match.group(1).strip()
                return code_text
            else:
                self.logger.error("파이썬 코드를 추출하지 못했습니다.")
                return None
        except Exception as e:
            self.logger.error(f"파이썬 코드 추출 중 오류 발생: {str(e)}", exc_info=True)
            return None
