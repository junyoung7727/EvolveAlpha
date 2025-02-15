import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 데이터 다운로드
df = yf.download('NVDA', start='2020-01-01', end='2024-12-29')

# 수익률 계산
returns = df['Close'].pct_change().dropna()

# 정규분포 추가
mu = returns.mean()
sigma = returns.std()
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
plt.hist(returns, bins=100, density=True, alpha=0.5, label='Data Distribution')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
plt.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), 
         'r--', lw=2, label='Normal Distribution')

plt.title('Density Histogram + Normal Dist\n(density=True, alpha=0.5)')
plt.legend()

plt.tight_layout()
plt.show()