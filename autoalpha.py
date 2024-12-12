import yfinance as yf
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances

###############################################################################
# SP100 티커 목록
###############################################################################
SP100_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "BRK-B", "UNH", "JNJ", "V", "XOM",
    "JPM", "WMT", "NVDA", "PG", "HD", "MA", "BAC", "CVX", "PFE", "ABBV",
    "KO", "TMO", "PEP", "COST", "MRK", "DIS", "MCD", "AVGO", "DHR",
    "ABT", "CSCO", "VZ", "ADBE", "WFC", "ACN", "CRM", "NFLX", "INTC", "LIN",
    "TXN", "PM", "HON", "LOW", "IBM", "UNP", "NEE", "UPS", "SCHW",
    "MDT", "INTU", "T", "CVS", "MS", "AMT", "RTX", "QCOM", "BMY", "AMGN",
    "C", "CME", "LMT", "CAT", "BA", "GS", "SPG", "BLK", "AXP", "PLD",
    "DE", "SYK", "BDX", "BK", "ADP", "NOW", "MU", "MDLZ", "TMUS", "GILD",
    "ADI", "ISRG", "ZTS", "COP", "MO", "REGN", "CCI", "MMC", "USB", "TGT",
    "MMM", "DUK", "SO", "CI", "EQIX", "CSX", "PNC", "ICE", "FDX",
    "NSC", "APD", "ITW", "SPGI"
]
SP100_TICKERS = list(set(SP100_TICKERS))  # 중복 제거

###############################################################################
# 데이터 가져오기
###############################################################################
def fetch_data(tickers, start, end):
    print("Fetching data from yfinance...")
    df = yf.download(tickers, start=start, end=end)
    print(df.head())
    if isinstance(df.columns, pd.MultiIndex):
        print(df.columns)
        df.columns = [f"{c[0].lower()}_{c[1].lower()}" for c in df.columns]
        print(df.columns)
    else:
        df.columns = [col.lower() for col in df.columns]
    print("Data fetched with shape:", df.shape)
    return df

panel_data = fetch_data(SP100_TICKERS, "2000-01-01", "2023-01-01")
print(panel_data.head())
###############################################################################
# 데이터 필드 예시:
# open, high, low, close, adjusted close, vwap, volume, turnover, adv20, adv60 등
# panel_data에는 ticker_field 형태의 컬럼 존재.
# 예: 'aapl_close', 'msft_volume', ...
# 여기서는 이미 panel_data에 OHLCV 형태로 존재.
# turnover, adv20, adv60 등은 rolling mean/합으로 구현 가능.
###############################################################################

# 예시: adv20 = rolling 20일 volume 평균
# 필요 시 factor 계산 과정에서 만듦
def adv(series, window=20):
    return series.rolling(window).mean()


###############################################################################
# 연산자 구현
#
# 1. Arithmetic Operators
#    log_diff, s_log_1p, densify, max, min, multiply, nan_mask, replace,
#    fraction, signed_power, nan_out, power, purify
###############################################################################
def safe_div(x, y):
    return x / (y.replace(0, np.nan))

def log_diff(x):
    return np.log(x) - np.log(x.shift(1))

def s_log_1p(x):
    return np.sign(x)*np.log1p(np.abs(x))

def densify(x):
    return x.ffill().fillna(0)

def arith_max(*args):
    df = pd.concat(args, axis=1)
    return df.max(axis=1)

def arith_min(*args):
    df = pd.concat(args, axis=1)
    return df.min(axis=1)

def multiply(*args, filter=True):
    df = pd.concat(args, axis=1)
    res = df.prod(axis=1)
    if filter:
        res[np.isnan(res)] = 0
    return res

def nan_mask(x, y):
    return x.where(~y.isna())

def replace(x, target, dest):
    # target, dest 모두 scalar나 series일 수 있지만 여기선 단순화
    return x.replace(target, dest)

def fraction(x):
    return (x - x.floor())*np.sign(x)

def signed_power(x, y):
    return np.sign(x)*np.abs(x)**y

def nan_out(x, lower, upper):
    return x.where((x>=lower)&(x<=upper))

def power(x, y):
    return x**y

def purify(x):
    x = x.copy()
    x[np.isinf(x)] = np.nan
    return x

###############################################################################
# 2. Transformational Operators
#    arc_tan, keep, clamp, filter, bucket, kth_element, trade_when,
#    left_tail, right_tail, tail
###############################################################################

def arc_tan(x):
    return np.arctan(x)

def keep(x, f, period=5):
    # 여기선 단순히: f==True인 구간에서 period일 유지
    x_array = x.values
    f_array = f.values
    out = np.full_like(x_array, np.nan, dtype=float)
    idxs = np.where(f_array==True)[0]
    for i in idxs:
        end = min(len(x_array), i+period)
        out[i:end] = x_array[i:end]
    return pd.Series(out, index=x.index)

def clamp(x, lower, upper, inverse=False):
    if not inverse:
        return x.clip(lower, upper)
    else:
        # inverse: [lower, upper] 범위 내면 NaN
        return x.where((x<lower)|(x>upper))

# filter: 이미 pandas where 사용 가능, 여기선 별도 구현 생략 가능
# bucket: 구간 나누기
def bucket(x, num_buckets=5):
    ranks = x.rank(pct=True)
    return np.ceil(ranks*num_buckets)

def kth_element(x, k, ignore=[]):
    arr = x.dropna().values
    arr = [a for a in arr if a not in ignore]
    arr = np.sort(arr)
    if len(arr)<k:
        return pd.Series(np.nan, index=x.index)
    val = arr[k-1]
    return pd.Series(val, index=x.index)

def trade_when(trigger, value, exit_condition):
    # 단순 구현: trigger True일 때 value, exit_condition True되면 NaN
    res = np.full_like(value.values, np.nan, dtype=float)
    holding = False
    for i in range(len(value)):
        if trigger.iloc[i]:
            holding = True
        if exit_condition.iloc[i]:
            holding = False
        if holding:
            res[i] = value.iloc[i]
    return pd.Series(res, index=value.index)

def left_tail(x, maximum):
    return x.where(x<=maximum)

def right_tail(x, minimum):
    return x.where(x>=minimum)

def tail(x, lower, upper, newval):
    return np.where((x>lower)&(x<upper), newval, x)

###############################################################################
# 3. Time Series Operators
#    hump, hump_decay, jump_decay, ts_regression, ts_co_skewness, ts_co_kurtosis,
#    ts_triple_corr, ts_partial_corr, ts_moment, ts_skewness, ts_kurtosis,
#    ts_decay_exp_window, ts_percentage, ts_weighted_delay, ts_arg_max,
#    ts_av_diff, ts_returns, ts_scale, ts_entropy, ts_rank_gmean_amean_diff
###############################################################################

def hump(x, hump=0.01):
    # 단순: x에 hump값 더하기
    return x + hump

def hump_decay(x, hump=0.01, factor=0.9):
    # window없이 단순: x에 점진적 hump 적용
    return x + hump * factor

def jump_decay(x, jump=0.01, factor=0.9):
    # 유사하게 단순 구현
    return x + jump * factor

def ts_regression(y, x, d, lag=0, rettype=0):
    # 단순히 slope 반환 예시
    from scipy.stats import linregress
    out = pd.Series(np.nan, index=y.index)
    for i in range(d, len(y)):
        yy = y.iloc[i-d:i]
        xx = x.iloc[i-d:i].shift(lag)
        valid = yy.notna() & xx.notna()
        if valid.sum()<d:
            continue
        slope, intercept, r, p, std = linregress(xx[valid], yy[valid])
        if rettype==2:
            out.iloc[i]=slope
        else:
            out.iloc[i]=intercept
    return out

def ts_co_skewness(y, x, d):
    # y,x로부터 rolling window d에서 co-skewness 계산 (간단히 y-x의 skew)
    diff = y - x
    return diff.rolling(d).skew()

def ts_co_kurtosis(y, x, d):
    diff = y - x
    return diff.rolling(d).kurt()

def ts_triple_corr(x, y, z, d):
    # 단순히 (x,y,z) 3개 rolling d window에서 상관관계 합
    out = pd.Series(np.nan, index=x.index)
    for i in range(d,len(x)):
        xx = x.iloc[i-d:i]
        yy = y.iloc[i-d:i]
        zz = z.iloc[i-d:i]
        # 3개 변수간 상관행렬 계산
        mat = pd.concat([xx,yy,zz],axis=1)
        if mat.dropna().shape[0]<d:
            continue
        c = mat.corr().values
        # triple corr 가령 (x,y)+(y,z)+(z,x) 합
        val = c[0,1]+c[1,2]+c[0,2]
        out.iloc[i]=val
    return out

def ts_partial_corr(x, y, z, d):
    # 단순 구현: partial correlation ~ 상관행렬에서 partialcorr 계산 복잡하므로 생략
    # 여기서는 x,y를 z에 대해 회귀 후 residual 상관
    out = pd.Series(np.nan, index=x.index)
    for i in range(d,len(x)):
        xx = x.iloc[i-d:i]
        yy = y.iloc[i-d:i]
        zz = z.iloc[i-d:i]
        df = pd.concat([xx,yy,zz],axis=1)
        df = df.dropna()
        if len(df)<d:
            continue
        # partial corr 계산: z에 대해 x,y를 회귀후 잔차 상관
        # 간단히 x->z, y->z 회귀 후 잔차 corr
        from scipy.stats import pearsonr
        Xz = np.polyfit(zz, xx, 1)
        Yz = np.polyfit(zz, yy, 1)
        x_res = xx - (Xz[0]*zz+Xz[1])
        y_res = yy - (Yz[0]*zz+Yz[1])
        rr,_ = pearsonr(x_res.dropna(), y_res.dropna())
        out.iloc[i]=rr
    return out

def ts_moment(x, d, k):
    mean = x.rolling(d).mean()
    return ((x-mean)**k).rolling(d).mean()

def ts_skewness(x, d):
    return x.rolling(d).skew()

def ts_kurtosis(x, d):
    return x.rolling(d).kurt()

def ts_decay_exp_window(x, d, factor):
    w = factor**np.arange(d)
    w = w/w.sum()
    return x.rolling(d).apply(lambda arr: np.sum(arr*w), raw=True)

def ts_percentage(x, d, percentage=0.5):
    # rolling d window에서 quantile
    return x.rolling(d).quantile(percentage)

def ts_weighted_delay(x, k):
    # 단순히 shift(k)로 대체
    return x.shift(k)

def ts_arg_max(x, d):
    return x.rolling(d).apply(np.argmax, raw=True)

def ts_av_diff(x, d):
    return x.rolling(d).mean().diff()

def ts_returns(x, d, mode='simple'):
    if mode=='simple':
        return x.pct_change(d)
    else:
        return np.log(x) - np.log(x.shift(d))

def ts_scale(x, d, constant=1):
    return ((x - x.rolling(d).mean())/x.rolling(d).std())*constant

def ts_entropy(x, d, buckets=10):
    out = pd.Series(np.nan, index=x.index)
    for i in range(d, len(x)):
        window = x.iloc[i-d:i].dropna()
        if len(window)<buckets:
            continue
        hist, _ = np.histogram(window, bins=buckets)
        p = hist/hist.sum()
        p = p[p>0]
        ent = -np.sum(p*np.log(p))
        out.iloc[i]=ent
    return out

def ts_rank_gmean_amean_diff(inputs, d):
    # inputs는 여러 시리즈일 경우 concat 필요
    df = pd.concat(inputs,axis=1)
    rank_df = df.rolling(d).apply(lambda arr: pd.Series(arr).rank(pct=True).iloc[-1], raw=False)
    gmean = rank_df.rolling(d).apply(lambda arr: np.exp(np.mean(np.log(arr[arr>0]))), raw=True)
    amean = rank_df.rolling(d).mean()
    return gmean-amean

###############################################################################
# 4. Cross-Sectional Operators
#    rank(x,rate), rank_by_side, generalized_rank, regression_neut(Y,X),
#    regression_proj(Y,X), scale, scale_down, truncate, vector_neut, quantile,
#    normalize, rank_gmean_amean_diff
###############################################################################

def cs_rank(x, rate=2):
    # row-wise rank 필요하지만, 여기선 단순 series라 가정
    return x.rank(pct=True)

# 아래 연산자들은 개념적으로만 구현(실제로는 cross-sectional 필요)
def rank_by_side(x, rate, scale):
    return x.rank(pct=True)*scale

def generalized_rank(open, m):
    return open.rank(pct=True)*m

def regression_neut(Y, X):
    # Y를 X에 회귀 후 residual 반환
    # 단순 구현
    X = sm.add_constant(X)
    model = sm.OLS(Y,X, missing='drop').fit()
    return Y - model.predict(X)

def regression_proj(Y, X):
    # Y를 X에 회귀 후 추정값 반환
    X = sm.add_constant(X)
    model = sm.OLS(Y,X, missing='drop').fit()
    return model.predict(X)

def scale(x, scale=1, longscale=None, shortscale=None):
    return (x - x.mean())/x.std()*scale

def scale_down(x, constant):
    return x/constant

def truncate(x, maxPercent):
    lower = x.quantile((1-maxPercent)/2)
    upper = x.quantile(1-(1-maxPercent)/2)
    return x.clip(lower, upper)

def vector_neut(x, y):
    proj = (x*y).sum()/ (y*y).sum() * y
    return x - proj

def quantile(x, driver, sigma):
    # driver 기반 quantile threshold
    q = driver.rank(pct=True)
    return x.where(q>sigma)

def normalize(x, useStd=False, limit=0.0):
    n = x - x.mean()
    if useStd:
        n = n/x.std()
    if limit>0:
        n = n.clip(-limit, limit)
    return n

# rank_gmean_amean_diff는 위 TS연산자에서 예제로 구현됨

###############################################################################
# 5. Group Operators
#    group_backfill, group_coalesce, group_neutralize, group_normalize
###############################################################################

def group_backfill(x, group, d, std):
    # 단순히 그룹별 결측값 평균으로 대체
    df = pd.DataFrame({'x':x,'group':group})
    out = df.groupby('group')['x'].apply(lambda g: g.fillna(g.mean()))
    return out

def group_coalesce(original_group, group2):
    # 그룹 2개를 병합하는 예
    return original_group.combine_first(group2)

def group_neutralize(x, group):
    df = pd.DataFrame({'x':x,'group':group})
    out = df.groupby('group')['x'].transform(lambda g: g - g.mean())
    return out

def group_normalize(x, group, constantCheck=True, tolerance=0.01, scale=1):
    df = pd.DataFrame({'x':x,'group':group})
    def norm_func(g):
        n = g - g.mean()
        s = g.std()
        if s< tolerance:
            return g
        return (n/s)*scale
    return df.groupby('group')['x'].transform(norm_func)

###############################################################################
# 함수 딕셔너리에 모두 등록
###############################################################################

functions = {
    # Arithmetic
    'log_diff': (log_diff, 1),
    's_log_1p': (s_log_1p, 1),
    'densify': (densify, 1),
    'max': (arith_max, 'vararg'),
    'min': (arith_min, 'vararg'),
    'multiply': (multiply, 'vararg'),
    'nan_mask': (nan_mask, 2),
    'replace': (replace, 3),
    'fraction': (fraction, 1),
    'signed_power': (signed_power, 2),
    'nan_out': (nan_out, 3),
    'power': (power, 2),
    'purify': (purify, 1),

    # Transformational
    'arc_tan': (arc_tan, 1),
    'keep': (keep, 3),
    'clamp': (clamp, 3),
    'bucket': (bucket, 2),
    'kth_element': (kth_element, 2),
    'trade_when': (trade_when, 3),
    'left_tail': (left_tail, 2),
    'right_tail': (right_tail, 2),
    'tail': (tail, 4),

    # Time Series (일부만 완전 구현)
    'hump': (hump, 1),
    'hump_decay': (hump_decay, 2),
    'jump_decay': (jump_decay, 2),
    'ts_regression': (ts_regression, 5),
    'ts_co_skewness': (ts_co_skewness, 3),
    'ts_co_kurtosis': (ts_co_kurtosis, 3),
    'ts_triple_corr': (ts_triple_corr, 4),
    'ts_partial_corr': (ts_partial_corr, 4),
    'ts_moment': (ts_moment, 3),
    'ts_skewness': (ts_skewness, 2),
    'ts_kurtosis': (ts_kurtosis, 2),
    'ts_decay_exp_window': (ts_decay_exp_window, 3),
    'ts_percentage': (ts_percentage, 3),
    'ts_weighted_delay': (ts_weighted_delay, 2),
    'ts_arg_max': (ts_arg_max, 2),
    'ts_av_diff': (ts_av_diff, 2),
    'ts_returns': (ts_returns, 3),
    'ts_scale': (ts_scale, 3),
    'ts_entropy': (ts_entropy, 3),
    'ts_rank_gmean_amean_diff': (ts_rank_gmean_amean_diff, 'vararg'),

    # Cross-Sectional
    'cs_rank': (cs_rank, 1),
    'rank_by_side': (rank_by_side, 3),
    'generalized_rank': (generalized_rank, 2),
    'regression_neut': (lambda Y,X: Y, 2), # stub
    'regression_proj': (lambda Y,X: Y, 2), # stub
    'scale': (scale, 1),
    'scale_down': (scale_down, 2),
    'truncate': (truncate, 2),
    'vector_neut': (vector_neut, 2),
    'quantile': (quantile, 3),
    'normalize': (normalize, 3),
    # 'rank_gmean_amean_diff' 위에 TS로 구현 (vararg)

    # Group
    'group_backfill': (group_backfill, 4),
    'group_coalesce': (group_coalesce, 2),
    'group_neutralize': (group_neutralize, 2),
    'group_normalize': (group_normalize, 4),
}

binary_ops = ['+', '-', '*', '/']

###############################################################################
# 노드 평가
###############################################################################
def eval_node(node, df):
    if isinstance(node, str):
        if node in df.columns:
            return df[node]
        else:
            try:
                val = float(node)
                return pd.Series(val, index=df.index)
            except:
                return pd.Series(np.nan, index=df.index)
    elif isinstance(node, (int, float)):
        return pd.Series(node, index=df.index)
    elif isinstance(node, tuple):
        op = node[0]
        if op in functions:
            func_spec = functions[op]
            func = func_spec[0]
            arg_count = func_spec[1]
            if arg_count=='vararg':
                args = [eval_node(a, df) for a in node[1:]]
                return func(*args)
            else:
                args = [eval_node(a, df) for a in node[1:]]
                return func(*args)
        elif op in binary_ops:
            left = eval_node(node[1], df)
            right = eval_node(node[2], df)
            if op == '+':
                return left + right
            elif op == '-':
                return left - right
            elif op == '*':
                return left * right
            elif op == '/':
                return safe_div(left, right)

    return pd.Series(np.nan, index=df.index)

###############################################################################
# 알파 평가
# SP100 중 변동성 있는 티커를 하나 선택해서 그 티커로 IC 계산
###############################################################################
def get_volatile_ticker(df):
    # 티커 중 close pct_change 변동성 있는 것 찾아 반환
    tickers = [c.split('_')[0] for c in df.columns if c.endswith('_close')]
    tickers = list(set(tickers))
    random.shuffle(tickers)
    for t in tickers:
        col = f"{t}_close"
        if col in df.columns:
            r = df[col].pct_change().dropna()
            if r.nunique()>5: # 어느정도 변동성
                return t
    return None

def evaluate_alpha(alpha_tree, df):
    ticker = get_volatile_ticker(df)
    if ticker is None:
        # 변동성 있는 티커 없음
        return None, None
    returns = df[f'{ticker}_close'].pct_change().dropna()
    factor = eval_node(alpha_tree, df)
    factor = factor.loc[returns.index].dropna()
    if len(factor)<10:
        return None, None
    factor = factor.dropna()

    if factor.nunique() <=1:
        return None, None

    ic, _ = spearmanr(returns, factor)
    if np.isnan(ic):
        return None, None
    return ic, {ticker: factor}

###############################################################################
# PCA-QD 기반 유전 알고리즘
###############################################################################
def get_alpha_vector(alpha_values):
    if len(alpha_values)==0:
        return None
    s = list(alpha_values.values())[0].dropna()
    if len(s)==0:
        return None
    return s.values

def pca_diversity_score(population_vectors):
    valid_vecs = [v for v in population_vectors if v is not None and len(v)>0]
    if len(valid_vecs)<2:
        return [1.0]*len(population_vectors)
    min_len = min(map(len, valid_vecs))
    valid_vecs = [v[-min_len:] for v in valid_vecs]

    mat = np.array(valid_vecs)
    if mat.shape[0]<2 or mat.shape[1]<2:
        return [1.0]*len(population_vectors)

    pca = PCA(n_components=2)
    emb = pca.fit_transform(mat)
    dist = pairwise_distances(emb, emb)
    diversity = dist.mean(axis=1)

    div_scores=[]
    idx=0
    for v in population_vectors:
        if v is not None and len(v)>=min_len:
            div_scores.append(diversity[idx])
            idx+=1
        else:
            div_scores.append(1.0)
    return div_scores

def correlation_penalty(alpha_values_list):
    vecs=[]
    for av in alpha_values_list:
        v = get_alpha_vector(av)
        vecs.append(v)
    valid_vecs = [vv for vv in vecs if vv is not None and len(vv)>0]
    if len(valid_vecs)<2:
        return [0]*len(alpha_values_list)
    min_len = min(map(len, valid_vecs))
    valid_vecs = [v[-min_len:] for v in valid_vecs]
    mat = np.array(valid_vecs)
    corr_mat = np.corrcoef(mat)
    penalties=[]
    idx=0
    for v in vecs:
        if v is not None and len(v)>=min_len:
            c = corr_mat[idx,:]
            c = np.delete(c, idx)
            penalties.append(np.nanmean(np.abs(c)))
            idx+=1
        else:
            penalties.append(0)
    return penalties

def random_alpha_tree(depth=3, simple_initial=False):
    if depth == 1:
        if random.random()<0.5:
            return str(round(random.uniform(-1,1),2))
        else:
            col = random.choice(panel_data.columns)
            return col
    else:
        if simple_initial:
            op = random.choice(binary_ops)
            return (op, random_alpha_tree(depth-1, True), random_alpha_tree(depth-1, True))
        else:
            if random.random()<0.5 and len(functions)>0:
                f = random.choice(list(functions.keys()))
                arg_count = functions[f][1]
                if arg_count=='vararg':
                    num_args = random.randint(2,3)
                    return tuple([f]+[random_alpha_tree(depth-1) for _ in range(num_args)])
                else:
                    return tuple([f]+[random_alpha_tree(depth-1) for _ in range(arg_count)])
            else:
                op = random.choice(binary_ops)
                return (op, random_alpha_tree(depth-1), random_alpha_tree(depth-1))

def mutate(alpha_tree, mutation_rate=0.1):
    if random.random()<mutation_rate:
        return random_alpha_tree(depth=3, simple_initial=True)
    return alpha_tree

def initialize_population(size=20, simple_initial=True):
    population=[]
    attempts=size*200
    valid_count=0
    for i in range(attempts):
        tree = random_alpha_tree(depth=3, simple_initial=simple_initial)
        ic, vals = evaluate_alpha(tree, panel_data)
        if ic is not None:
            population.append((tree, ic, vals))
            valid_count += 1
            if len(population)>=size:
                break
    if len(population)==0:
        print(f"Attempted {attempts} times but no valid alpha found.")
        print("Possible reasons:")
        print("- Data may not have enough variance in any ticker's close price.")
        print("- The generated random alpha trees may produce constant or NaN factors.")
        print("- Try increasing attempts or simplifying tree generation further.")
        print("- Consider extending the data range.")
        raise ValueError("No valid initial population generated.")
    else:
        print(f"Generated initial population with {valid_count} valid alphas out of {attempts} attempts.")
    return population

def genetic_algorithm(df, generations=5, population_size=20):
    population = initialize_population(population_size, simple_initial=True)

    for gen in range(generations):
        def tournament(pop):
            best=None
            for _ in range(3):
                cand = random.choice(pop)
                if best is None or cand[1]>best[1]:
                    best=cand
            return best
        parents = [tournament(population) for _ in range(population_size)]

        offspring=[]
        for i in range(0,len(parents),2):
            p1 = parents[i][0]
            p2 = parents[(i+1)%len(parents)][0]
            child = random.choice([p1,p2])
            child = mutate(child, mutation_rate=0.2)
            ic, vals = evaluate_alpha(child, df)
            if ic is not None:
                offspring.append((child, ic, vals))

        combined = population + offspring
        alpha_vals_list=[c[2] for c in combined]
        vectors=[get_alpha_vector(av) for av in alpha_vals_list]
        div_scores = pca_diversity_score(vectors)
        corr_penalties = correlation_penalty(alpha_vals_list)

        λ=0.1
        μ=0.1
        final_scores=[]
        for i, ind in enumerate(combined):
            base_ic=ind[1]
            diversity=div_scores[i]
            penalty=corr_penalties[i]
            score=base_ic+λ*diversity-μ*penalty
            final_scores.append(score)

        sorted_indices=np.argsort(final_scores)[::-1]
        population=[combined[i] for i in sorted_indices[:population_size]]

        best=population[0]
        print(f"Gen {gen+1}: Best IC={best[1]:.4f}, Tree={best[0]}")

    return population[0]

# 실행
best_alpha = genetic_algorithm(panel_data, generations=5, population_size=5000)
print("Best Alpha:", best_alpha[0], "IC:", best_alpha[1])
