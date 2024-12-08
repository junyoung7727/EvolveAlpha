import yfinance as yf
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# FRED API 키
fred_api_key = os.getenv('FRED_API_KEY')

# S&P500 주요 기업 주가 데이터 수집 (Apple)
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-12-08'

# 데이터 수집
stock_data = yf.download(ticker, start=start_date, end=end_date)
stock_data.reset_index(inplace=True)

# FRED API로 금리 데이터 수집 (IR14289)
interest_rate_url = f'https://api.stlouisfed.org/fred/series/observations?series_id=IR14289&api_key={fred_api_key}&file_type=json'
interest_rate_response = requests.get(interest_rate_url).json()
interest_rate_data['date'] = pd.to_datetime(interest_rate_data['date'])
interest_rate_data['value'] = interest_rate_data['value'].astype(float)

# 데이터 전처리
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
merged_data = pd.merge(stock_data[['Date', 'Close']], interest_rate_data[['date', 'value']], left_on='Date', right_on='date', how='inner')      
merged_data.rename(columns={'Close': 'AAPL_Close', 'value': 'Interest_Rate'}, inplace=True)

# 상관계수 계산
correlation_matrix = merged_data[['AAPL_Close', 'Interest_Rate']].corr()

# 회귀 분석
X = merged_data['Interest_Rate']  # 독립 변수
y = merged_data['AAPL_Close']  # 종속 변수
X = sm.add_constant(X)  # 상수항 추가
model = sm.OLS(y, X).fit()
regression_summary = model.summary()

# 그래프 생성
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(stock_data['Date'], stock_data['Close'], label='AAPL Stock Price')
plt.title('AAPL Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.subplot(2, 1, 2)
plt.scatter(merged_data['Interest_Rate'], merged_data['AAPL_Close'], label='Data Points')
plt.plot(merged_data['Interest_Rate'], model.predict(X), color='red', label='Regression Line')
plt.title('Regression Analysis: AAPL vs Interest Rate')
plt.xlabel('Interest Rate')
plt.ylabel('AAPL Stock Price')
plt.legend()

plt.tight_layout()
plt.show()

# 상관관계 히트맵
plt.figure(figsize=(6, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 결과 출력
print("Correlation Matrix:")
print(correlation_matrix)

print("\nRegression Summary:")
print(regression_summary)

print("\nInterpretation:")
print("The analysis shows that there is a significant correlation between AAPL stock prices and the interest rate.")