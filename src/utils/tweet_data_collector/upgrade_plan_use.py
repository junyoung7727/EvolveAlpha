import tweepy
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 트위터 API 인증 정보 설정
api_key = os.getenv('X_API_KEY')
api_secret_key = os.getenv('X_API_KEY_SECRET')
access_token = os.getenv('X_API_ACCESS_TOKEN')
access_token_secret = os.getenv('X_API_ACCESS_TOKEN_SECRET')
bearer_token = os.getenv('X_API_BEARER_TOKEN')

# Tweepy 클라이언트 초기화 (Bearer Token 사용)
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# 검색할 키워드
query = "Bitcoin -is:retweet"

# 최근 트윗 검색 (Free 권한은 search_recent_tweets만 가능)
try:
    response = client.search_recent_tweets(
        query=query,
        tweet_fields=['id', 'text', 'created_at', 'author_id', 'lang', 'public_metrics'],
        max_results=10  # Free 권한은 한 번에 최대 10개 트윗 반환
    )
    if response.data:
        for tweet in response.data:
            print(f"ID: {tweet.id}, Text: {tweet.text}, Created At: {tweet.created_at}")
    else:
        print("No tweets found.")
except Exception as e:
    print(f"An error occurred: {e}")
