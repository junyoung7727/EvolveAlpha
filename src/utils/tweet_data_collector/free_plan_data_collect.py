import tweepy
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
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
client_id = os.getenv('X_API_CLIENT_ID')
client_secret = os.getenv('X_API_CLIENT_SECRET')

# 인증 정보 확인
if not all([api_key, api_secret_key, access_token, access_token_secret, bearer_token, client_id, client_secret]):
    raise ValueError("환경 변수에 API 자격 증명이 누락되었습니다. 확인해주세요.")

# OAuth 2.0 Client Credentials Flow로 토큰 가져오기
token_url = "https://api.twitter.com/oauth2/token"
auth = (client_id, client_secret)
data = {"grant_type": "client_credentials"}

response = requests.post(token_url, auth=auth, data=data)
if response.status_code == 200:
    access_token = response.json().get("access_token")
    print(f"OAuth 2.0 Access Token: {access_token}")
else:
    print(f"Error fetching OAuth 2.0 token: {response.status_code}")
    print(response.json())
    exit()

# Tweepy Client 초기화
client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=api_key,
    consumer_secret=api_secret_key,
    access_token=access_token,
    access_token_secret=access_token_secret,
    wait_on_rate_limit=True
)

# 자신의 사용자 정보를 요청하여 인증 확인
try:
    user = client.get_user(username='JohnDoe123')  # 자신의 트위터 사용자 이름으로 변경
    print(f"인증 성공: {user.data.username}")
except Exception as e:
    print(f"인증 실패: {e}")

# 검색할 키워드
keywords = ['Bitcoin', 'Ethereum', 'Kospi']

# CSV 파일로 저장할 컬럼 정의
columns = [
    'id', 'text', 'created_at', 'author_id', 'lang',
    'reply_count', 'like_count', 'retweet_count', 'quote_count', 'keyword'
]

# 데이터 저장을 위한 리스트
tweets_data = []

# 5년 전 날짜부터 현재까지 설정
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(days=5 * 365)

# 월 단위로 날짜 범위 생성
def generate_date_ranges(start, end):
    current = start
    while current < end:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        yield current, min(next_month, end)
        current = next_month

# 트윗 검색 함수
def search_tweets(query, start_time, end_time, max_results=100):
    tweets = []
    next_token = None
    while True:
        try:
            response = client.search_all_tweets(
                query=query,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                max_results=max_results,
                tweet_fields=['id', 'text', 'created_at', 'author_id', 'lang', 'public_metrics'],
                next_token=next_token
            )
            if response.data:
                for tweet in response.data:
                    tweets.append([
                        tweet.id,
                        tweet.text.replace('\n', ' ').replace('\r', ' '),
                        tweet.created_at,
                        tweet.author_id,
                        tweet.lang,
                        tweet.public_metrics.get('reply_count', 0),
                        tweet.public_metrics.get('like_count', 0),
                        tweet.public_metrics.get('retweet_count', 0),
                        tweet.public_metrics.get('quote_count', 0),
                        query.replace('"', '')
                    ])
                print(f"Collected {len(response.data)} tweets for '{query}' from {start_time} to {end_time}")
                if 'next_token' in response.meta:
                    next_token = response.meta['next_token']
                else:
                    break
            else:
                break
        except tweepy.TooManyRequests:
            print("Rate limit reached. Sleeping for 15 minutes.")
            time.sleep(15 * 60)
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    return tweets

# 메인 데이터 수집 루프
for keyword in keywords:
    print(f"Collecting tweets for keyword: {keyword}")
    query = f'"{keyword}" -is:retweet (lang:ko OR lang:en)'

    for start, end in generate_date_ranges(start_time, end_time):
        tweets = search_tweets(query, start, end)
        if tweets:
            temp_df = pd.DataFrame(tweets, columns=columns)
            output_dir = 'tweets_data'
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f'{output_dir}/tweets_data_{keyword}_{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}.csv'
            temp_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"Data saved to {output_filename}")
        else:
            print(f"No tweets found for '{keyword}' from {start} to {end}")

# (옵션) 전체 데이터를 하나의 CSV로 병합
merged_filename = 'tweets_data_5years.csv'
csv_files = [os.path.join('tweets_data', f) for f in os.listdir('tweets_data') if f.endswith('.csv')]

df_list = []
for file in csv_files:
    df = pd.read_csv(file)
    df_list.append(df)

if df_list:
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(merged_filename, index=False, encoding='utf-8-sig')
    print(f"All data merged into {merged_filename}")
else:
    print("No CSV files found to merge.")
