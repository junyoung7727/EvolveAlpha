import tweepy
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Twitter API 인증 정보 로드
API_KEY = os.getenv('X_API_KEY')
API_SECRET = os.getenv('X_API_KEY_SECRET')
BEARER_TOKEN = os.getenv('X_API_BEARER_TOKEN')
ACCESS_TOKEN = os.getenv('X_API_ACCESS_TOKEN')
ACCESS_TOKEN_SECRET = os.getenv('X_API_ACCESS_TOKEN_SECRET')

# API 키 유효성 검사
if not all([API_KEY, API_SECRET, BEARER_TOKEN, ACCESS_TOKEN, ACCESS_TOKEN_SECRET]):
    raise ValueError("Some API credentials are missing in the .env file. Please check.")

# Tweepy 클라이언트 초기화
client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
    wait_on_rate_limit=True
)

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
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=5*365)  # 약 5년 전부터 현재까지

# 월 단위로 날짜 범위 나누기
def generate_date_ranges(start, end):
    current = start
    while current < end:
        if current.month == 12:
            next_month = current.replace(year=current.year+1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month+1, day=1)
        yield current, min(next_month, end)
        current = next_month

# 트윗 검색 함수
def search_tweets(query, start_time, end_time, max_results=500):
    tweets = []
    next_token = None
    while True:
        try:
            response = client.search_all_tweets(
                query=query,
                start_time=start_time.isoformat("T") + "Z",
                end_time=end_time.isoformat("T") + "Z",
                max_results=500,
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
                    time.sleep(1)
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
        tweets_data.extend(tweets)
        if tweets:
            temp_df = pd.DataFrame(tweets, columns=columns)
            output_dir = 'tweets_data'
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f'{output_dir}/tweets_data_{keyword}_{start.strftime("%Y%m")}.csv'
            temp_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"Data saved to {output_filename}")
            time.sleep(1)
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
