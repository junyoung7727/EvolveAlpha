import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# NVIDIA 주가 데이터 다운로드
df = yf.download('NVDA', start='2020-01-01', end='2024-12-29')

# 1. Candlestick 차트 with 거래량
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.03, subplot_titles=('NVDA Price', 'Volume'),
                    row_width=[0.7, 0.3])

fig.add_trace(go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name='OHLC'),
                            row=1, col=1)

fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
                     name='Volume'),
                     row=2, col=1)

# 20일, 50일 이동평균선 추가
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

fig.add_trace(go.Scatter(x=df.index, y=df['MA20'],
                        line=dict(color='orange', width=1),
                        name='MA20'),
                        row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['MA50'],
                        line=dict(color='blue', width=1),
                        name='MA50'),
                        row=1, col=1)

fig.update_layout(
    title='NVIDIA Stock Price (2020-2024)',
    yaxis_title='Stock Price (USD)',
    yaxis2_title='Volume',
    xaxis_rangeslider_visible=False,
    height=800
)

fig.show()

# 2. 수익률 분포 시각화
plt.figure(figsize=(15, 6))

# 일간 수익률 계산
returns = df['Close'].pct_change().dropna()

# 커널 밀도 추정과 히스토그램 결합
sns.histplot(returns, bins=100, stat='density', alpha=0.5)
sns.kdeplot(returns, color='red', lw=2)

# 정규분포 추가
mu = returns.mean()
sigma = returns.std()
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
plt.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2)), 
         'g--', lw=2, label='Normal Distribution')

plt.title('NVIDIA Daily Returns Distribution')
plt.xlabel('Returns')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 3. 변동성 클러스터링 시각화
plt.figure(figsize=(15, 8))

# 20일 롤링 변동성 계산
rolling_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100

plt.plot(rolling_vol, color='purple', linewidth=2)
plt.fill_between(rolling_vol.index, rolling_vol, alpha=0.3, color='purple')
plt.title('NVIDIA 20-Day Rolling Volatility (Annualized)')
plt.xlabel('Date')
plt.ylabel('Volatility (%)')
plt.grid(True, alpha=0.3)
plt.show()