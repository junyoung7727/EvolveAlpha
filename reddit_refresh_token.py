import requests
import base64
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Reddit 애플리케이션 정보
client_id = os.getenv("REDDIT_CLIENT_ID")  # 클라이언트 ID
client_secret = os.getenv("REDDIT_CLIENT_SECRET")  # 클라이언트 시크릿
redirect_uri = "http://localhost:8080/callback"  # 등록된 리다이렉트 URI
authorization_code = "qRudl5irehC02s8S8DqUiLqPpAemzw"  # 새로운 Authorization Code

# Basic 인증 헤더 생성
auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

# 액세스 및 리프레시 토큰 요청
url = "https://www.reddit.com/api/v1/access_token"
headers = {
    "Authorization": f"Basic {auth}",
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": "junyoung727/0.1 by junyoung727"
}
data = {
    "grant_type": "authorization_code",
    "code": authorization_code,
    "redirect_uri": redirect_uri
}

response = requests.post(url, headers=headers, data=data)

if response.status_code == 200:
    tokens = response.json()
    print("Access Token:", tokens["access_token"])
    print("Refresh Token:", tokens.get("refresh_token"))
else:
    print("Error:", response.status_code, response.text)
