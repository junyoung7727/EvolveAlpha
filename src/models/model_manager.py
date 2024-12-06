from typing import Dict, Any
import logging
import re
from openai import OpenAI
import json
from datetime import datetime

class ModelManager:
    """LLM과의 통신만을 담당하는 클래스"""
    
    def __init__(self):
        self.client = OpenAI()
        self.model_name = "gpt-4o-mini"
        self.logger = logging.getLogger("Model_Manager")
        self.response_history = []

    def generate_plan(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """LLM으로부터 응답 생성"""
        try:
            # 메시지 구성
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # API 호출
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7
            )

            print("-----------------------------------------------------------")
            print(response.choices[0].message.content)
            print("-----------------------------------------------------------")


            result = self.parse_response(str(response.choices[0].message.content))

            # 응답 처리
            result = {
                "status": "success",
                "plan": result["plan"],
                "instruct": result["instruct"],
                "timestamp": datetime.now().isoformat()
            }

            # 응답 이력 저장
            self.response_history.append(result)

            return result

        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

            self.response_history.append(error_result)
            self.logger.error(f"응답 생성 중 오류: {str(e)}")
            return error_result
        
    def generate_response(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """LLM으로부터 응답 생성"""
        try:
            # 메시지 구성
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # API 호출
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7
            )
            print("-----------------------------------------------------------")
            print(response.choices[0].message.content)
            print("-----------------------------------------------------------")
            # 응답 처리
            result = {
                "status": "success",
                "content": str(response.choices[0].message.content),
                "timestamp": datetime.now().isoformat()
            }

            # 응답 이력 저장
            self.response_history.append(result)

            return result

        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.response_history.append(error_result)
            self.logger.error(f"응답 생성 중 오류: {str(e)}")
            return error_result
        
    def parse_response(self, content: str) -> Dict[str, Any]:
        """응답 파싱"""
        try:
            # RESPONSE와 JSON 부분 추출
            response_match = re.search(r'<RESPONSE>(.*?)</RESPONSE>', content, re.DOTALL)
            json_match = re.search(r'<JSON>(.*?)</JSON>', content, re.DOTALL)
            
            result = {
                "status": "success",
                "instruct": "",
                "plan": None,
                "timestamp": datetime.now().isoformat()
            }
            
            # RESPONSE 처리
            if response_match:
                response_content = response_match.group(1).strip()
                result["instruct"] = response_content
            
            # JSON 처리
            if json_match:
                try:
                    json_content = json_match.group(1).strip()
                    result["plan"] = json.loads(json_content)
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON 파싱 오류: {str(e)}")
                    # JSON 파싱 실패 시에도 원본 텍스트는 유지
                    result["plan"] = {"raw": json_content}
            
            return result
            
        except Exception as e:
            self.logger.error(f"응답 파싱 오류: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "content": content,
                "timestamp": datetime.now().isoformat()
            }


    def get_response_history(self) -> list:
        """응답 이력 반환"""
        return self.response_history
