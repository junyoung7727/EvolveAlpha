import os
from pathlib import Path
from dotenv import load_dotenv
import re
import logging
from src.utils.role_selector import select_random_roles
from src.discussion.discussion_manager import DiscussionManager
from src.agents.role_definitions import ROLE_DESCRIPTIONS
from datetime import datetime
import re

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def setup_environment():
    """환경 설정"""
    load_dotenv()
    
    required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY", "SERPER_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        raise ValueError(f"다음 API 키가 설정되지 않았습니다: {', '.join(missing_keys)}")

def setup_directories():
    """필요한 디렉토리 생성"""
    directories = [
        "results/discussions",
        "results/analysis",
        "results/plots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    try:
        # 기본 설정
        setup_logging()
        setup_environment()
        setup_directories()
        
        # 토론 주제 설정
        topic = """
        주식시장의 모멘텀 알파를 발견하기 위한 연구 프로젝트입니다.
        각자의 전문성을 바탕으로 독창적인 투자 전략과 아이디어를 제시해주세요.
        주제가 진부하다면 감점입니다. 복잡하고 단계적 사고과정을 거쳐 퀀트가 흥미를 가질 수 있을 만한 주제를 창의적으로 제시하세요.
        """
        
        # 6명의 랜덤 역할 선택
        selected_roles = select_random_roles(6)
        print(f"\n선택된 역할들: {', '.join(selected_roles)}")
        
        # 토론 매니저 생성 및 실행
        manager = DiscussionManager(agents=selected_roles, num_rounds=3)
        result = manager.run_discussion(topic)
        
        if result["status"] == "success":
            # 결과 저장
            filename = f"discussion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            manager.save_result(filename)
            print(f"\n토론이 성공적으로 완료되었습니다.")
        else:
            print(f"\n토론 중 오류가 발생했습니다: {result.get('error')}")
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()