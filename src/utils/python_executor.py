import sys
import json
import traceback
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import redirect_stdout, redirect_stderr
import yfinance as yf
from datetime import datetime, timedelta
import requests
import base64
import subprocess
import logging

class PythonExecutor:
    def __init__(self):
        self.execution_count = 0
        self.max_executions = 5
        self.results_history = []
        
    def execute_code(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Python 코드 실행 및 결과 반환"""
        try:
            if self.execution_count >= self.max_executions:
                return {
                    "error": "최대 실행 횟수 초과",
                    "status": "error",
                    "timestamp": datetime.now().isoformat()
                }
            
            self.execution_count += 1
            
            # 실행 환경 설정
            execution_globals = {
                '__builtins__': __builtins__,
                'pd': pd,
                'np': np,
                'plt': plt,
                'sns': sns,
                'yf': yf,
                'datetime': datetime,
                'timedelta': timedelta
            }
            
            if context:
                execution_globals.update(context)
            
            # 출력 캡처를 위한 버퍼
            stdout_buffer = StringIO()
            stderr_buffer = StringIO()
            
            result = {
                "execution_id": self.execution_count,
                "timestamp": datetime.now().isoformat(),
                "code": code,
                "outputs": {},
                "plots": [],
                "dataframes": [],
                "stdout": "",
                "stderr": "",
                "status": "success"
            }
            
            # 코드 실행 및 결과 캡처
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                try:
                    exec(code, execution_globals)
                    
                    # 반환된 변수들 처리
                    for var_name, value in execution_globals.items():
                        if var_name.startswith('__'):
                            continue  # 내장 변수 무시
                        if isinstance(value, pd.DataFrame):
                            result["dataframes"].append({
                                "name": var_name,
                                "data": value.to_dict(),
                                "shape": value.shape,
                                "description": value.describe().to_dict()
                            })
                        elif isinstance(value, (np.ndarray, list, dict)):
                            result["outputs"][var_name] = {
                                "type": str(type(value)),
                                "value": value if isinstance(value, (list, dict)) else value.tolist(),
                                "shape": value.shape if isinstance(value, np.ndarray) else len(value)
                            }
                    
                    # 플롯 캡처
                    if plt.get_fignums():
                        for fig_num in plt.get_fignums():
                            fig = plt.figure(fig_num)
                            plot_buffer = BytesIO()
                            fig.savefig(plot_buffer, format='png')
                            plot_buffer.seek(0)
                            plot_data = base64.b64encode(plot_buffer.read()).decode()
                            result["plots"].append({
                                "figure_number": fig_num,
                                "data": plot_data
                            })
                            plt.close(fig)
                    
                except Exception as e:
                    result["status"] = "error"
                    result["error"] = {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc()
                    }
            
            # 표준 출력/에러 캡처
            result["stdout"] = stdout_buffer.getvalue()
            result["stderr"] = stderr_buffer.getvalue()
            
            # 실행 이력 저장
            self.results_history.append(result)
            
            return result
            
        except Exception as e:
            error_result = {
                "execution_id": self.execution_count,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            }
            self.results_history.append(error_result)
            return error_result

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """실행 기록 반환"""
        return self.results_history

    def reset_execution_count(self):
        """실행 횟수 초기화"""
        self.execution_count = 0
        self.results_history = []

    def execute(self, script: str) -> Dict[str, Any]:
        """파이썬 스크립트 실행"""
        try:
            # 스크립트를 파일로 저장
            with open("temp_script.py", "w", encoding='utf-8') as file:
                file.write(script)
            
            # 스크립트 실행
            result = subprocess.run(
                ["python", "temp_script.py"],
                capture_output=True,
                text=True
            )
            
            # 결과 반환
            execution_result = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "status": "success" if result.returncode == 0 else "error",
                "timestamp": datetime.now().isoformat()
            }
            
            if result.returncode != 0:
                execution_result["error"] = {
                    "message": "스크립트 실행 중 오류 발생",
                    "traceback": result.stderr
                }
            
            # 실행 이력 저장
            self.results_history.append(execution_result)
            
            return execution_result
        
        except Exception as e:
            error_result = {
                "status": "error",
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc()
                },
                "timestamp": datetime.now().isoformat()
            }
            self.results_history.append(error_result)
            return error_result
