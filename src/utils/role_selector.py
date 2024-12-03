import random
from typing import List
from src.agents.role_definitions import AVAILABLE_ROLES, ROLE_DESCRIPTIONS

def select_random_roles(num_roles: int = 6) -> List[str]:
    """무작위로 역할 선택"""
    if num_roles > len(AVAILABLE_ROLES):
        raise ValueError("요청된 역할 수가 가능한 역할의 총 개수를 초과합니다.")
    return random.sample(AVAILABLE_ROLES, num_roles)