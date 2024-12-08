import praw
import pandas as pd
import time
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

###########################
# 환경설정
###########################
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = 'junyoung727'
refresh_token = '139178961664279-uYGDI1ij7VloF7Iw7QnNn8Rm7YNVEg'  # refresh_token을 발급받았을 경우

# 5년 전 timestamp
# 5년 전 timestamp
end_time = datetime.now(timezone.utc)

# 5년 전 UTC 시간 계산
start_time = int((end_time - timedelta(days=5*365)).timestamp())

# 키워드 리스트
keywords = ["Bitcoin", "Ethereum", "KOSPI"]

# 데이터 저장 경로
output_dir = "./reddit_data"
os.makedirs(output_dir, exist_ok=True)

###########################
# PRAW 인증 함수
###########################
def get_reddit_instance():
    # refresh_token을 통한 인증
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        refresh_token=refresh_token,
        user_agent=user_agent
    )
    return reddit

reddit = get_reddit_instance()

###########################
# Pushshift API를 이용한 게시물 ID 수집 함수
###########################
def fetch_submission_ids(keyword, start, end, batch_size=500):
    """
    Pushshift를 이용해 start~end 기간 사이 keyword를 포함하는 게시물 ID를 batch_size 단위로 가져옴
    """
    submission_ids = []
    current_end = end
    url_template = "https://api.pushshift.io/reddit/search/submission/?q={keyword}&after={after}&before={before}&sort=asc&size={size}"
    while True:
        url = url_template.format(
            keyword=keyword,
            after=start,
            before=current_end,
            size=batch_size
        )
        resp = requests.get(url)
        if resp.status_code != 200:
            time.sleep(10)
            continue

        data = resp.json().get('data', [])
        if not data:
            # 더 이상 결과가 없으면 종료
            break

        # 데이터에서 submission id만 추출
        for d in data:
            submission_ids.append(d['id'])

        # 마지막 항목의 created_utc를 기준으로 다음 루프에서 after를 갱신
        if data:
            last_utc = data[-1]['created_utc']
            start = last_utc  # 다음 요청 시 이 시각 이후의 게시물을 가져옴
            time.sleep(1)  # Rate Limit 방지를 위해 sleep

        # 결과가 batch_size보다 적다면 종료
        if len(data) < batch_size:
            break

    return submission_ids

###########################
# 데이터 수집 함수
###########################
def fetch_submissions_data(submission_ids, reddit_instance, save_path, batch_save_size=1000):
    """
    주어진 submission_ids 리스트를 바탕으로 PRAW를 사용해 상세 정보를 수집하고,
    일정량(batch_save_size)마다 parquet 파일로 저장.
    """
    all_data = []
    count = 0

    for sid in submission_ids:
        # 인증 만료 등 예외 처리
        try:
            submission = reddit_instance.submission(id=sid)
        except Exception as e:
            # 인증 만료 등 문제가 발생하면 재인증 시도
            print("Error fetching submission:", e, "Trying to re-authenticate...")
            time.sleep(5)
            reddit_instance = get_reddit_instance()
            continue

        # 속성 추출
        # 가능한 많은 정보를 수집
        data = {
            'id': submission.id,
            'title': submission.title,
            'score': submission.score,                     # 점수(좋아요수 유사)
            'num_comments': submission.num_comments,       # 댓글 수
            'upvote_ratio': submission.upvote_ratio,
            'author': str(submission.author) if submission.author else None,
            'created_utc': submission.created_utc,         # 게시물 생성 UTC 타임스탬프
            'selftext': submission.selftext,
            'subreddit': str(submission.subreddit),
            'url': submission.url,
            'num_crossposts': submission.num_crossposts,   # 퍼가기(크로스포스트) 횟수
            'over_18': submission.over_18,
            'is_self': submission.is_self,
            'locked': submission.locked,
            'spoiler': submission.spoiler,
            'stickied': submission.stickied
        }

        all_data.append(data)
        count += 1

        # Rate limit 준수: 1초에 하나 정도 요청하게끔 (상황에 따라 조정)
        time.sleep(1)

        # 일정 수집량마다 저장
        if count % batch_save_size == 0:
            df = pd.DataFrame(all_data)
            # parquet로 저장 (append를 위해 존재 여부 확인)
            if os.path.exists(save_path):
                # append mode
                df_existing = pd.read_parquet(save_path)
                df = pd.concat([df_existing, df], ignore_index=True)
            df.to_parquet(save_path, index=False)
            all_data = []
            print(f"{count} records saved to {save_path}")

    # 남은 데이터 저장
    if all_data:
        df = pd.DataFrame(all_data)
        if os.path.exists(save_path):
            df_existing = pd.read_parquet(save_path)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_parquet(save_path, index=False)
        print(f"Final save: {len(df)} records in {save_path}")


###########################
# 메인 로직
###########################
for keyword in keywords:
    print(f"Collecting data for keyword: {keyword}")
    submission_ids = fetch_submission_ids(keyword, start_time, end_time, batch_size=500)
    print(f"Found {len(submission_ids)} submissions for {keyword}")

    # 키워드별로 다른 파일명 지정
    save_path = os.path.join(output_dir, f"{keyword.lower()}_posts.parquet")

    # 데이터 수집
    fetch_submissions_data(submission_ids, reddit, save_path, batch_save_size=1000)
    print(f"Data collection for {keyword} completed.\n")
