import os
import yfinance as yf
import pandas as pd
import requests
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# FRED API 키 가져오기
fred_api_key = os.getenv("FRED_API_KEY")

# 1. S&P 500 주요 기업 주가 데이터 수집 (예: AAPL)
ticker = 'AAPL'  # Apple Inc.
start_date = '2020-01-01'
end_date = '2023-10-01'
stock_data = yf.download(ticker, start=start_date, end=end_date)

# 2. FRED API로 금리 데이터 수집 (10년 만기 국채 금리)
interest_rate_id = 'IR14200'  # 10-Year Treasury Constant Maturity Rate
url = f'https://api.stlouisfed.org/fred/series/observations?series_id={interest_rate_id}&api_key={fred_api_key}&file_type=json'
response = requests.get(url)
interest_rate_data = response.json()

# 3. 금리 데이터 프레임으로 변환
interest_rates = pd.DataFrame(interest_rate_data['observations'])
interest_rates['date'] = pd.to_datetime(interest_rates['date'])
interest_rates['value'] = pd.to_numeric(interest_rates['value'])
interest_rates.set_index('date', inplace=True)

# 4. 데이터 전처리: AAPL 종가 이동 평균 추가
stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()

# 5. 통계량 산출
correlation = stock_data['Close'].corr(interest_rates['value'])
print(f'Correlation between {ticker} closing price and interest rate: {correlation:.2f}')
print(stock_data.describe())
print(interest_rates.describe())

# 6. 그래프 생성
plt.figure(figsize=(14, 7))

# AAPL 주가 시각화
plt.subplot(2, 1, 1)
plt.plot(stock_data.index, stock_data['Close'], label='AAPL Closing Price', color='blue')
plt.plot(stock_data.index, stock_data['MA_20'], label='20-Day Moving Average', color='orange')
plt.title(f'{ticker} Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# 금리 시각화
plt.subplot(2, 1, 2)
plt.plot(interest_rates.index, interest_rates['value'], label='Interest Rate', color='red')
plt.title('Interest Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Interest Rate (%)')
plt.legend()

plt.tight_layout()
plt.show()

# 7. 데이터베이스에 저장하는 코드 (예: Pinecone)
# Pinecone에 연결하기 위한 예시 (Pinecone API 키 필요)
# import pinecone
# pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment='us-west1-gcp')
# index = pinecone.Index("financial-data")

# # 데이터 저장 예시
# for i, row in stock_data.iterrows():
#     index.upsert(vectors=[(str(i), row.values.tolist())])  # 데이터 저장
# print("Stock data saved to Pinecone.")

# 데이터베이스에서 데이터 조회하는 예시
# saved_data = index.fetch(ids=[str(stock_data.index[0]), str(stock_data.index[1])])  # 첫 두 개의 데이터 조회
# print("Retrieved data from Pinecone:", saved_data)
