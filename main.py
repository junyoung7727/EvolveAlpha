import os
from pathlib import Path
from src.utils.role_selector import select_random_roles
from src.discussion.discussion_manager import DiscussionManager
from dotenv import load_dotenv

def setup_directories():
    """필요한 디렉토리 생성"""
    Path("results/discussions").mkdir(parents=True, exist_ok=True)

def main():
    # 환경 변수 로드
    load_dotenv()
    
    # 디렉토리 설정
    setup_directories()
    
    # 6개의 랜덤 역할 선택
    selected_roles = select_random_roles(6)
    print(f"선택된 역할들: {', '.join(selected_roles)}")
    
    # 토론 매니저 생성 및 실행
    manager = DiscussionManager(num_rounds=5)
    result = manager.run_discussion(selected_roles)
    
    # 결과 저장
    filename = f"discussion_{result.timestamp.strftime('%Y%m%d_%H%M%S')}"
    manager.save_result(result, filename)
    print(f"\n토론 결과가 저장되었습니다: {filename}.json")

if __name__ == "__main__":
    main()