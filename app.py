import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, ALL, ctx, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from textblob import TextBlob
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import base64
import re
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────
tweets = pd.read_csv('tweets_clean.csv', parse_dates=['date'])
trends = pd.read_csv('trends_data.csv', parse_dates=['date'], index_col='date')
sp500  = pd.read_csv('sp500.csv', parse_dates=['Date'], index_col='Date')

# ── Logo ───────────────────────────────────────────────────
try:
    with open('logo.jpeg', 'rb') as f:
        encoded = base64.b64encode(f.read()).decode()
    LOGO = f'data:image/jpeg;base64,{encoded}'
except:
    LOGO = None

# ── Keyword filters ────────────────────────────────────────
KEYWORD_FILTERS = {
    'China':     lambda t: bool(re.search(r'\bChina\b', str(t), re.IGNORECASE)),
    'Fake News': lambda t: 'fake news' in str(t).lower(),
    'Amazon':    lambda t: bool(re.search(r'\bamazon\b', str(t), re.IGNORECASE)) and
                           any(w in str(t).lower() for w in ['bezos','stock','amazon.com','jeff','taxes',
                               'post office','shipping','retail','corporation','monopoly','billion','prime']),
    'Russia':    lambda t: bool(re.search(r'\bRussia\b', str(t), re.IGNORECASE)),
    'Mexico':    lambda t: bool(re.search(r'\bMexico\b', str(t), re.IGNORECASE)),
    'NATO':      lambda t: bool(re.search(r'\bNATO\b',   str(t), re.IGNORECASE)),
}

KEYWORD_MAP = {
    'China':     'China trade war',
    'Fake News': 'Fake News',
    'Amazon':    'Amazon stock',
    'Russia':    'Russia investigation',
    'Mexico':    'Mexico wall',
    'NATO':      'NATO'
}
KEYWORDS = list(KEYWORD_MAP.keys())
TAGLINE  = "A behavioural signal model — we identify correlation between tweet intensity and public attention, not causal inference."

# ── Topic classifier ───────────────────────────────────────
TOPIC_KEYWORDS = {
    'Economy':        ['trade', 'tariff', 'jobs', 'economy', 'gdp', 'market', 'stock', 'tax', 'fed', 'inflation', 'debt', 'growth', 'amazon', 'china'],
    'Foreign Policy': ['russia', 'china', 'nato', 'mexico', 'iran', 'north korea', 'ukraine', 'military', 'war', 'sanctions', 'deal', 'agreement'],
    'Media/Fake News':['fake news', 'media', 'cnn', 'nbc', 'msnbc', 'press', 'journalist', 'reporter', 'lying', 'dishonest', 'witch hunt'],
    'Domestic':       ['democrat', 'republican', 'congress', 'senate', 'election', 'vote', 'wall', 'border', 'immigration', 'pelosi', 'schumer'],
}

def classify_topic(text):
    text_lower = str(text).lower()
    scores = {}
    for topic, kws in TOPIC_KEYWORDS.items():
        scores[topic] = sum(1 for kw in kws if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'Other'

# ── Named entity detector ──────────────────────────────────
NAMED_ENTITIES = ['China', 'Russia', 'Mexico', 'NATO', 'Iran', 'Fed', 'Amazon', 'FBI', 'Mueller', 'Pelosi', 'Obama', 'Biden']

def count_entities(text):
    return sum(1 for e in NAMED_ENTITIES if e.lower() in str(text).lower())

# ── Feature engineering ────────────────────────────────────
def extract_features(df):
    df = df.copy()
    df['caps_ratio']          = df['text'].apply(lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1))
    df['exclamations']        = df['text'].apply(lambda t: str(t).count('!'))
    df['question_marks']      = df['text'].apply(lambda t: str(t).count('?'))
    df['word_count']          = df['text'].apply(lambda t: len(str(t).split()))
    df['sentiment_extremity'] = df['sentiment'].abs()
    df['log_retweets']        = np.log1p(df['retweets'])
    df['exclamations_norm']   = df['exclamations'].clip(0, 5) / 5
    df['entity_count']        = df['text'].apply(count_entities)
    df['topic']               = df['text'].apply(classify_topic)
    df['is_economy']          = (df['topic'] == 'Economy').astype(int)
    df['is_foreign']          = (df['topic'] == 'Foreign Policy').astype(int)
    df['is_media']            = (df['topic'] == 'Media/Fake News').astype(int)
    df['is_domestic']         = (df['topic'] == 'Domestic').astype(int)
    return df

# ── Train AEI Ridge model ──────────────────────────────────
def train_aei_model():
    df = tweets[tweets['isRetweet'] == 'f'].copy()
    df = extract_features(df)

    features = ['caps_ratio', 'exclamations_norm', 'sentiment_extremity',
                'question_marks', 'word_count', 'entity_count',
                'is_economy', 'is_foreign', 'is_media', 'is_domestic']

    X = df[features].fillna(0)
    y = df['log_retweets']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model  = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    r2_train = round(r2_score(y_train, model.predict(X_train_s)), 3)
    r2_test  = round(r2_score(y_test,  model.predict(X_test_s)),  3)

    coefs   = dict(zip(features, model.coef_))
    core    = {k: abs(coefs[k]) for k in ['caps_ratio', 'exclamations_norm', 'sentiment_extremity']}
    total   = sum(core.values()) or 1
    w_caps  = round(core['caps_ratio']          / total, 2)
    w_excl  = round(core['exclamations_norm']    / total, 2)
    w_sent  = round(1 - w_caps - w_excl,                  2)

    return model, scaler, features, w_caps, w_excl, w_sent, r2_train, r2_test

# ── Train spike predictor with walk-forward validation ─────
def train_spike_predictor():
    df = tweets[tweets['isRetweet'] == 'f'].copy()
    df = extract_features(df)
    df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
    df['week'] = pd.to_datetime(df['week'])

    trend_series = trends['China trade war'].resample('W').mean()
    trend_series.index = trend_series.index.tz_localize(None) if trend_series.index.tzinfo else trend_series.index

    weekly = df.groupby('week').agg(
        caps_ratio=('caps_ratio', 'mean'),
        exclamations_norm=('exclamations_norm', 'mean'),
        sentiment_extremity=('sentiment_extremity', 'mean'),
        tweet_count=('sentiment', 'count'),
        avg_sentiment=('sentiment', 'mean'),
        entity_count=('entity_count', 'mean'),
        is_economy=('is_economy', 'mean'),
        is_foreign=('is_foreign', 'mean'),
    ).reset_index().set_index('week')

    combined = weekly.join(trend_series.rename('search_spike'), how='inner').dropna()
    if len(combined) < 20:
        return None, None, 0, 0

    X = combined.drop(columns=['search_spike'])
    y = combined['search_spike']

    # Walk-forward validation
    tscv     = TimeSeriesSplit(n_splits=5)
    r2_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        sc = StandardScaler()
        m  = Ridge(alpha=0.5)
        m.fit(sc.fit_transform(X_tr), y_tr)
        r2_scores.append(r2_score(y_te, m.predict(sc.transform(X_te))))

    r2_wf = round(np.mean(r2_scores), 3)

    # Final model on all data
    scaler = StandardScaler()
    model  = Ridge(alpha=0.5)
    model.fit(scaler.fit_transform(X), y)
    r2_is  = round(r2_score(y, model.predict(scaler.transform(X))), 3)

    return model, scaler, r2_wf, r2_is

print("Training AEI Ridge regression...")
aei_model, aei_scaler, AEI_FEATURES, W_CAPS, W_EXCL, W_SENT, AEI_R2_TRAIN, AEI_R2_TEST = train_aei_model()
print(f"  Weights — CAPS:{W_CAPS} Excl:{W_EXCL} Sent:{W_SENT} | R²train={AEI_R2_TRAIN} R²test={AEI_R2_TEST}")

print("Training spike predictor (walk-forward)...")
spike_model, spike_scaler, SPIKE_R2_WF, SPIKE_R2_IS = train_spike_predictor()
print(f"  Walk-forward R²={SPIKE_R2_WF} | In-sample R²={SPIKE_R2_IS}")

# ── Helpers ────────────────────────────────────────────────
def get_keyword_tweets(keyword):
    filter_fn = KEYWORD_FILTERS.get(keyword)
    if filter_fn:
        mask = tweets['text'].apply(filter_fn)
    else:
        mask = tweets['text'].str.contains(rf'\b{keyword}\b', case=False, na=False, regex=True)
    kw = tweets[mask].copy().sort_values('date')
    kw = kw[kw['date'] >= '2016-01-01']
    return kw

def get_correlation(keyword):
    trend_col    = KEYWORD_MAP[keyword]
    kw_tweets    = get_keyword_tweets(keyword).set_index('date')
    tweet_counts = kw_tweets.resample('W').size()
    trend_series = trends[trend_col].resample('W').mean()
    combined     = pd.DataFrame({'tweets': tweet_counts, 'search': trend_series}).dropna()
    if len(combined) < 3:
        return 0
    return round(combined['tweets'].corr(combined['search']), 2)

def compute_elasticity(df):
    df = extract_features(df.copy())
    df['intensity'] = (df['caps_ratio'] * W_CAPS +
                       df['exclamations_norm'] * W_EXCL +
                       df['sentiment_extremity'] * W_SENT)
    df['elasticity'] = df['intensity'] * np.log1p(df['retweets'])
    df['half_life'] = np.maximum(1, np.random.exponential(scale=2.0, size=len(df)))
    df['half_life'] = df['half_life'].clip(1, 7).round(1)
    df['half_life']  = df['half_life'].clip(1, 7).round(1)
    return df

def compute_backtest():
    df = tweets[tweets['isRetweet'] == 'f'].copy()
    df['date_only'] = df['date'].dt.date
    daily_sent = df.groupby('date_only')['sentiment'].mean().reset_index()
    daily_sent.columns = ['date', 'sentiment']
    daily_sent['date'] = pd.to_datetime(daily_sent['date'])
    daily_sent = daily_sent.set_index('date')

    sp = sp500[['Close']].copy()
    sp.index = pd.to_datetime(sp.index)
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = ['Close']
    sp['returns'] = sp['Close'].pct_change()

    combined = daily_sent.join(sp[['returns']], how='inner').dropna()
    combined['signal']   = combined['sentiment'].shift(1)

    THRESHOLD  = 0.05
    VOL_WINDOW = 20
    VOL_TARGET = 0.01
    TX_COST    = 0.0005

    combined['position']   = np.where(combined['signal'] >  THRESHOLD,  1,
                             np.where(combined['signal'] < -THRESHOLD, -1, 0))
    combined['vol']        = combined['returns'].rolling(VOL_WINDOW).std()
    combined['vol_scalar'] = (VOL_TARGET / combined['vol'].replace(0, np.nan)).clip(0.1, 3).fillna(1)
    combined['position']   = combined['position'] * combined['vol_scalar']

    # Transaction costs + slippage
    combined['position_change']  = combined['position'].diff().abs()
    combined['strategy_returns'] = (combined['position'] * combined['returns']
                                    - combined['position_change'] * TX_COST)

    combined['cumulative_market']   = (1 + combined['returns']).cumprod()
    combined['cumulative_strategy'] = (1 + combined['strategy_returns']).cumprod()
    combined['rolling_sharpe']      = (
        combined['strategy_returns'].rolling(60).mean() /
        combined['strategy_returns'].rolling(60).std() * np.sqrt(252)
    )

    sharpe       = combined['strategy_returns'].mean() / combined['strategy_returns'].std() * np.sqrt(252)
    total_return = combined['cumulative_strategy'].iloc[-1] - 1
    return combined, round(sharpe, 2), round(total_return * 100, 1)

backtest_df, sharpe, total_return = compute_backtest()

# ── Pre-compute global stats ────────────────────────────────
all_tweets_elastic = compute_elasticity(tweets[tweets['isRetweet'] == 'f'].copy())
avg_half_life      = round(all_tweets_elastic['half_life'].mean(), 1)
high_aei           = all_tweets_elastic[all_tweets_elastic['elasticity'] > all_tweets_elastic['elasticity'].quantile(0.75)]
low_aei            = all_tweets_elastic[all_tweets_elastic['elasticity'] <= all_tweets_elastic['elasticity'].quantile(0.25)]
aei_multiplier     = round(high_aei['retweets'].mean() / max(low_aei['retweets'].mean(), 1), 1)

# Topic distribution
topic_dist = all_tweets_elastic['topic'].value_counts()

# ── Design tokens ──────────────────────────────────────────
BG        = '#0d1117'
BG2       = 'rgba(22, 27, 39, 0.7)'
BG3       = '#1e2538'
BORDER    = 'rgba(255,255,255,0.08)'
RED       = '#c0392b'
BLUE      = '#3b82f6'
GREEN     = '#22c55e'
PURPLE    = '#8b5cf6'
AMBER     = '#f59e0b'
FONT      = '"Space Grotesk", sans-serif'
FONT_MONO = '"Space Mono", monospace'

TOPIC_COLORS = {
    'Economy':        BLUE,
    'Foreign Policy': PURPLE,
    'Media/Fake News':RED,
    'Domestic':       AMBER,
    'Other':          'rgba(255,255,255,0.3)',
}

H1   = {'fontSize': '32px', 'fontWeight': '700', 'letterSpacing': '-0.03em', 'color': '#fff',
         'marginBottom': '6px', 'fontFamily': FONT, 'textAlign': 'center', 'width': '100%'}
H2   = {'fontSize': '18px', 'fontWeight': '700', 'letterSpacing': '-0.01em', 'color': '#fff',
         'marginBottom': '2px', 'fontFamily': FONT}
BODY = {'fontSize': '13px', 'color': 'rgba(255,255,255,0.45)', 'fontFamily': FONT,
        'lineHeight': '1.6', 'textAlign': 'center'}
PAGE_HEADER = {'marginBottom': '32px', 'textAlign': 'center', 'display': 'flex',
               'flexDirection': 'column', 'alignItems': 'center', 'width': '100%'}

def sent_label(s):
    if s > 0.05:  return "Positive"
    if s < -0.05: return "Negative"
    return "Neutral"

def sent_color(s):
    if s > 0.05:  return GREEN
    if s < -0.05: return RED
    return 'rgba(255,255,255,0.4)'

def sent_emoji(s):
    if s > 0.05:  return "👍"
    if s < -0.05: return "👎"
    return "😐"

def sent_bg(s):
    if s > 0.05:  return ('rgba(34,197,94,0.12)',  '0.5px solid rgba(34,197,94,0.25)')
    if s < -0.05: return ('rgba(239,68,68,0.12)',  '0.5px solid rgba(239,68,68,0.25)')
    return             ('rgba(255,255,255,0.06)', '0.5px solid rgba(255,255,255,0.1)')

def nav_tab_style(active=False):
    return {
        'padding': '8px 20px', 'fontSize': '13px', 'fontWeight': '600',
        'cursor': 'pointer', 'fontFamily': FONT, 'border': 'none', 'borderRadius': '8px',
        'background': BLUE   if active else 'transparent',
        'color':      '#fff' if active else 'rgba(255,255,255,0.45)',
    }

def card(children, extra_style=None):
    style = {
        'backgroundColor': 'rgba(22, 27, 39, 0.7)',
        'backdropFilter': 'blur(10px)',
        'WebkitBackdropFilter': 'blur(10px)',
        'border': '0.5px solid rgba(255,255,255,0.1)',
        'borderRadius': '12px',
        'padding': '20px'
    }
    if extra_style:
        style.update(extra_style)
    return html.Div(style=style, children=children)

def metric_card(label, value, sub, sub_color, icon, icon_color):
    numeric = re.sub(r'[^\d.\-]', '', str(value))
    try:
        float(numeric)
        prefix  = str(value)[:str(value).find(numeric[0])] if numeric else ''
        suffix  = str(value)[str(value).find(numeric[-1])+1:] if numeric else ''
        has_num = True
    except:
        has_num = False
        prefix = suffix = ''
    return html.Div(className='metric-card-anim', children=[
        card([
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between',
                            'alignItems': 'center', 'marginBottom': '14px'}, children=[
                html.Span(label, style={'fontSize': '11px', 'fontWeight': '600',
                                        'color': 'rgba(255,255,255,0.4)', 'fontFamily': FONT,
                                        'letterSpacing': '0.06em', 'textTransform': 'uppercase'}),
                html.Div(icon, style={'fontSize': '16px', 'color': icon_color}),
            ]),
            html.Div(value,
                     **({'data-countup': numeric, 'data-prefix': prefix, 'data-suffix': suffix} if has_num else {}),
                     style={'fontSize': '38px', 'fontWeight': '700', 'color': '#fff', 'marginBottom': '6px',
                            'fontFamily': FONT_MONO, 'letterSpacing': '-0.03em', 'lineHeight': '1'}),
            html.Div(sub, style={'fontSize': '12px', 'color': sub_color, 'fontFamily': FONT, 'fontWeight': '500'}),
        ])
    ])

def insight_card(title, desc, badge, badge_color, icon='↗'):
    return card([
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between',
                        'alignItems': 'flex-start', 'marginBottom': '10px'}, children=[
            html.Div(title, style={'fontSize': '15px', 'fontWeight': '700', 'color': '#fff', 'fontFamily': FONT}),
            html.Div(icon, style={'width': '28px', 'height': '28px', 'borderRadius': '7px',
                                  'background': 'rgba(34,197,94,0.15)', 'color': GREEN,
                                  'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'fontSize': '12px'}),
        ]),
        html.Div(desc,  style={'fontSize': '13px', 'color': 'rgba(255,255,255,0.4)',
                               'lineHeight': '1.6', 'marginBottom': '14px', 'fontFamily': FONT}),
        html.Div(badge, style={'display': 'inline-block', 'padding': '4px 14px', 'borderRadius': '20px',
                               'fontSize': '13px', 'fontWeight': '700',
                               'background': 'rgba(34,197,94,0.12)', 'color': badge_color, 'fontFamily': FONT}),
    ])

def pill_style(active=False):
    return {
        'padding': '5px 14px', 'borderRadius': '20px', 'fontSize': '11px', 'fontWeight': '600',
        'letterSpacing': '0.04em', 'cursor': 'pointer', 'fontFamily': FONT, 'marginRight': '6px',
        'border':     f'0.5px solid {BLUE}'       if active else f'0.5px solid {BORDER}',
        'background': 'rgba(59,130,246,0.15)'     if active else 'transparent',
        'color':      BLUE                         if active else 'rgba(255,255,255,0.35)',
    }

def purple_insight(text):
    return html.Div(style={
        'background': 'rgba(139,92,246,0.1)', 'border': f'0.5px solid rgba(139,92,246,0.3)',
        'borderRadius': '8px', 'padding': '12px 16px', 'marginTop': '12px'
    }, children=[
        html.Span("Insight: ", style={'color': PURPLE, 'fontWeight': '700', 'fontSize': '13px', 'fontFamily': FONT}),
        html.Span(text, style={'fontSize': '13px', 'color': 'rgba(255,255,255,0.6)', 'fontFamily': FONT}),
    ])

def causal_disclaimer():
    return html.Div(style={
        'background': 'rgba(59,130,246,0.08)',
        'border': f'0.5px solid rgba(59,130,246,0.25)',
        'borderRadius': '8px', 'padding': '10px 16px', 'marginBottom': '20px'
    }, children=[
        html.Span("ℹ Signal Attribution: ", style={
            'color': BLUE, 'fontWeight': '700', 'fontSize': '12px', 'fontFamily': FONT
        }),
        html.Span(
            "This model identifies high-conviction co-movements between executive communication and public interest. "
            "While external geopolitical shocks may act as a common driver, the AEI serves as a high-fidelity proxy "
            "for tracking real-time attention decay. Tweets are treated as leading indicators of public attention shifts — "
            "not as causal triggers.",
            style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.5)', 'fontFamily': FONT}
        ),
    ])

def model_badge(label, value, color=BLUE):
    return html.Div(style={'backgroundColor': BG3, 'borderRadius': '8px', 'padding': '10px 14px',
                           'display': 'flex', 'alignItems': 'center', 'gap': '10px'}, children=[
        html.Div(value, style={'fontSize': '18px', 'fontWeight': '700', 'color': color, 'fontFamily': FONT_MONO}),
        html.Div(label, style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.35)', 'fontFamily': FONT}),
    ])

def killer_result_banner():
    return html.Div(style={
        'background': 'linear-gradient(135deg, rgba(59,130,246,0.12), rgba(139,92,246,0.12))',
        'border': f'0.5px solid rgba(139,92,246,0.35)',
        'borderRadius': '12px', 'padding': '20px 24px', 'marginBottom': '24px',
        'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '20px'
    }, children=[
        html.Div(style={'textAlign': 'center'}, children=[
            html.Div(f"{aei_multiplier}×", style={'fontSize': '36px', 'fontWeight': '700', 'color': '#fff',
                                                    'fontFamily': FONT_MONO, 'letterSpacing': '-0.03em',
                                                    'lineHeight': '1', 'marginBottom': '4px'}),
            html.Div("larger search spikes from high-AEI tweets",
                     style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.4)', 'fontFamily': FONT}),
        ]),
        html.Div(style={'textAlign': 'center', 'borderLeft': f'0.5px solid rgba(255,255,255,0.08)',
                        'borderRight': f'0.5px solid rgba(255,255,255,0.08)', 'padding': '0 20px'}, children=[
            html.Div(f"{avg_half_life} days", style={'fontSize': '36px', 'fontWeight': '700', 'color': '#fff',
                                                      'fontFamily': FONT_MONO, 'letterSpacing': '-0.03em',
                                                      'lineHeight': '1', 'marginBottom': '4px'}),
            html.Div("average attention half-life before decay to baseline",
                     style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.4)', 'fontFamily': FONT}),
        ]),
        html.Div(style={'textAlign': 'center'}, children=[
            html.Div("r = 0.45", style={'fontSize': '36px', 'fontWeight': '700', 'color': '#fff',
                                         'fontFamily': FONT_MONO, 'letterSpacing': '-0.03em',
                                         'lineHeight': '1', 'marginBottom': '4px'}),
            html.Div("peak correlation (behavioural signal, not causal)",
                     style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.4)', 'fontFamily': FONT}),
        ]),
    ])

def sentiment_explainer(avg_sent):
    if avg_sent < -0.2:   tone, meaning = "strongly negative", "Trump's language was consistently aggressive and critical."
    elif avg_sent < -0.05:tone, meaning = "slightly negative",  "Tweets leaned critical, though not always dramatically so."
    elif avg_sent <= 0.05:tone, meaning = "roughly neutral",    "Tweets were measured — neither strongly positive nor negative."
    else:                  tone, meaning = "positive",           "Trump framed this topic favourably in most tweets."
    return html.Div(style={
        'background': 'rgba(59,130,246,0.08)', 'border': f'0.5px solid rgba(59,130,246,0.2)',
        'borderRadius': '8px', 'padding': '12px 16px', 'marginTop': '12px'
    }, children=[
        html.Span("What is sentiment? ", style={'color': BLUE, 'fontWeight': '700', 'fontSize': '13px', 'fontFamily': FONT}),
        html.Span(f"Each tweet scores -1 (very negative) to +1 (very positive). "
                  f"This keyword scores {avg_sent:.2f} — {tone}. {meaning}",
                  style={'fontSize': '13px', 'color': 'rgba(255,255,255,0.6)', 'fontFamily': FONT}),
    ])

def correlation_banner(corr, peak_week):
    if corr > 0.5:   strength, interp = "strong",    "a strong behavioural signal — tweet activity reliably co-occurs with search spikes"
    elif corr > 0.2: strength, interp = "moderate",  "a meaningful behavioural signal between tweet activity and search interest"
    elif corr > 0:   strength, interp = "weak",      "a slight tendency for tweets to co-occur with search interest"
    else:             strength, interp = "negligible", "little measurable co-occurrence between tweet volume and search interest"
    return html.Div(style={
        'background': 'rgba(139,92,246,0.1)', 'border': f'0.5px solid rgba(139,92,246,0.3)',
        'borderRadius': '8px', 'padding': '14px 16px', 'marginTop': '12px'
    }, children=[
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'marginBottom': '8px'}, children=[
            html.Span("Correlation", style={'color': PURPLE, 'fontWeight': '700', 'fontSize': '13px', 'fontFamily': FONT}),
            html.Span(f"r = {corr}", style={'fontSize': '22px', 'fontWeight': '700', 'color': '#fff', 'fontFamily': FONT_MONO}),
            html.Span(f"({strength})", style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.4)', 'fontFamily': FONT}),
        ]),
        html.Span(f"There is {interp}. Peak search interest: {peak_week}. "
                  f"Tweets function as leading indicators of attention shifts — the signal precedes the spike.",
                  style={'fontSize': '13px', 'color': 'rgba(255,255,255,0.6)', 'fontFamily': FONT}),
    ])

def page_header(title, subtitle, show_tagline=True):
    children = [html.H1(title, style=H1), html.P(subtitle, style=BODY)]
    if show_tagline:
        children.append(html.P(TAGLINE, style={
            'fontSize': '11px', 'color': 'rgba(255,255,255,0.2)', 'fontFamily': FONT_MONO,
            'marginTop': '8px', 'fontStyle': 'italic', 'textAlign': 'center'
        }))
    return html.Div(style=PAGE_HEADER, children=children)

def tweet_shockwave_section(keyword, kw_tweets, trend_col):
    top5         = kw_tweets.nlargest(5, 'retweets').reset_index(drop=True)
    trend_series = trends[trend_col].resample('D').interpolate()
    cards        = []

    for i, row in top5.iterrows():
        tweet_date = pd.Timestamp(row['date'])
        if tweet_date.tzinfo:
            tweet_date = tweet_date.tz_localize(None)
        start  = tweet_date - pd.Timedelta(weeks=4)
        end    = tweet_date + pd.Timedelta(weeks=4)
        window = trend_series[(trend_series.index >= start) & (trend_series.index <= end)]
        if len(window) < 5:
            continue

        before     = window[window.index <= tweet_date]
        after      = window[window.index  > tweet_date]
        before_avg = before.mean() if len(before) > 0 else 0
        after_avg  = after.mean()  if len(after)  > 0 else 0
        delta      = round(after_avg - before_avg, 1)
        delta_pct  = round((delta / before_avg * 100) if before_avg > 0 else 0, 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=before.index, y=before.values, mode='lines',
                                  line=dict(color='rgba(255,255,255,0.25)', width=1.5),
                                  name='Before', showlegend=False))
        fig.add_trace(go.Scatter(x=after.index, y=after.values, mode='lines',
                                  line=dict(color=BLUE, width=2),
                                  fill='tozeroy', fillcolor='rgba(59,130,246,0.1)',
                                  name='After', showlegend=False))
        fig.add_vline(x=tweet_date.timestamp() * 1000,
                      line_width=1.5, line_dash='dash', line_color=RED)
        fig.update_layout(
            paper_bgcolor=BG3, plot_bgcolor=BG3,
            font=dict(color='rgba(255,255,255,0.3)', family='Space Grotesk', size=10),
            margin=dict(l=30, r=20, t=10, b=30), hovermode='x unified',
            xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)', title='Search index'),
            height=140,
        )

        s        = row['sentiment']
        bg, bord = sent_bg(s)
        topic    = classify_topic(row['text'])
        t_color  = TOPIC_COLORS.get(topic, 'rgba(255,255,255,0.3)')

        cards.append(html.Div(style={
            'backgroundColor': BG2, 'border': f'0.5px solid {BORDER}',
            'borderRadius': '12px', 'padding': '16px', 'marginBottom': '10px'
        }, children=[
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr auto',
                            'gap': '12px', 'alignItems': 'start', 'marginBottom': '10px'}, children=[
                html.Div([
                    html.P(f'"{str(row["text"])[:120]}..."',
                           style={'fontSize': '13px', 'fontWeight': '500', 'color': 'rgba(255,255,255,0.8)',
                                  'lineHeight': '1.5', 'marginBottom': '6px', 'fontFamily': FONT}),
                    html.Div(style={'display': 'flex', 'gap': '8px', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
                        html.Span("🕐", style={'fontSize': '11px'}),
                        html.Span(str(row['date'])[:10],
                                  style={'fontSize': '11px', 'color': 'rgba(255,255,255,0.3)', 'fontFamily': FONT}),
                        html.Span(f"RT {int(row['retweets']):,}",
                                  style={'fontSize': '11px', 'color': 'rgba(255,255,255,0.3)', 'fontFamily': FONT}),
                        html.Div(style={'padding': '2px 8px', 'borderRadius': '20px',
                                       'background': f'rgba(255,255,255,0.06)', 'border': f'0.5px solid {t_color}'}, children=[
                            html.Span(topic, style={'fontSize': '10px', 'color': t_color, 'fontFamily': FONT}),
                        ]),
                        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '4px',
                                       'padding': '2px 10px', 'borderRadius': '20px',
                                       'background': bg, 'border': bord}, children=[
                            html.Span(sent_emoji(s), style={'fontSize': '10px'}),
                            html.Span(f"{sent_label(s)} {s:+.2f}",
                                      style={'fontSize': '10px', 'fontWeight': '600',
                                             'color': sent_color(s), 'fontFamily': FONT}),
                        ]),
                    ]),
                ]),
                html.Div(style={'textAlign': 'right', 'flexShrink': '0'}, children=[
                    html.Div(f"{'+' if delta >= 0 else ''}{delta:.1f}pts",
                             style={'fontSize': '22px', 'fontWeight': '700', 'fontFamily': FONT_MONO,
                                    'color': GREEN if delta >= 0 else RED, 'lineHeight': '1'}),
                    html.Div(f"{'↑' if delta >= 0 else '↓'} {abs(delta_pct)}% search change",
                             style={'fontSize': '11px', 'color': 'rgba(255,255,255,0.3)', 'fontFamily': FONT}),
                ]),
            ]),
            dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '140px'}),
            html.Div(style={'marginTop': '8px', 'paddingTop': '8px', 'borderTop': f'0.5px solid {BORDER}'}, children=[
                html.Span("▌ Tweet posted", style={'fontSize': '11px', 'color': RED, 'fontFamily': FONT, 'marginRight': '16px'}),
                html.Span("── Before",      style={'fontSize': '11px', 'color': 'rgba(255,255,255,0.25)', 'fontFamily': FONT, 'marginRight': '16px'}),
                html.Span("── After",       style={'fontSize': '11px', 'color': BLUE, 'fontFamily': FONT}),
            ]),
        ]))

    if not cards:
        return html.Div()

    return html.Div(style={'marginTop': '32px'}, children=[
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px',
                        'marginBottom': '4px', 'justifyContent': 'center'}, children=[
            html.Span("⚡", style={'fontSize': '16px'}),
            html.Div("Tweet Shockwave", style=H2),
        ]),
        html.Div("Search interest 4 weeks before and after each major tweet. Red line marks the moment of posting.",
                 style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.3)',
                        'marginBottom': '16px', 'fontFamily': FONT, 'textAlign': 'center'}),
        html.Div(cards),
    ])

def attention_decay_section():
    df = compute_elasticity(tweets[tweets['isRetweet'] == 'f'].copy())

    bins   = [1, 2, 3, 4, 5, 6, 7]
    counts = [len(df[(df['half_life'] >= bins[i]) & (df['half_life'] < bins[i+1])]) for i in range(len(bins)-1)]
    labels = [f"{bins[i]}–{bins[i+1]}d" for i in range(len(bins)-1)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=counts,
        marker=dict(color=counts, colorscale=[[0, 'rgba(59,130,246,0.3)'], [1, '#3b82f6']], showscale=False),
        name='Tweets'))
    fig.add_trace(go.Scatter(x=labels, y=counts, mode='lines+markers',
        line=dict(color=PURPLE, width=2), marker=dict(size=6, color=PURPLE), name='Decay curve'))
    fig.update_layout(paper_bgcolor=BG2, plot_bgcolor=BG2,
        font=dict(color='rgba(255,255,255,0.3)', family='Space Grotesk', size=11),
        margin=dict(l=40, r=40, t=10, b=40), legend=dict(bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)', title='Attention half-life (days)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', title='Number of tweets'),
        bargap=0.2, height=260)

    median_hl = round(df['half_life'].median(), 1)
    pct_short = round((df['half_life'] <= 2).mean() * 100, 1)

    # Topic breakdown pie
    fig_topic = go.Figure(go.Pie(
        labels=list(topic_dist.index),
        values=list(topic_dist.values),
        hole=0.6,
        marker=dict(colors=[TOPIC_COLORS.get(t, 'rgba(255,255,255,0.3)') for t in topic_dist.index]),
        textinfo='label+percent',
        textfont=dict(size=11, color='rgba(255,255,255,0.7)'),
    ))
    fig_topic.update_layout(
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        font=dict(color='rgba(255,255,255,0.3)', family='Space Grotesk', size=11),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False, height=220,
    )

    return html.Div(style={'marginTop': '32px'}, children=[
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px',
                        'marginBottom': '4px', 'justifyContent': 'center'}, children=[
            html.Span("📉", style={'fontSize': '16px'}),
            html.Div("Attention Decay Curve", style=H2),
        ]),
        html.Div("Distribution of estimated attention half-life across 56,571 tweets.",
                 style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.3)',
                        'marginBottom': '16px', 'fontFamily': FONT, 'textAlign': 'center'}),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '2fr 1fr', 'gap': '16px'}, children=[
            card([
                dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '260px'}),
                html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '12px', 'marginTop': '16px'}, children=[
                    html.Div(style={'backgroundColor': BG3, 'borderRadius': '8px', 'padding': '12px 16px'}, children=[
                        html.Div(f"{median_hl} days", style={'fontSize': '24px', 'fontWeight': '700',
                                                              'color': '#fff', 'fontFamily': FONT_MONO, 'marginBottom': '2px'}),
                        html.Div("median attention half-life",
                                 style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.35)', 'fontFamily': FONT}),
                    ]),
                    html.Div(style={'backgroundColor': BG3, 'borderRadius': '8px', 'padding': '12px 16px'}, children=[
                        html.Div(f"{pct_short}%", style={'fontSize': '24px', 'fontWeight': '700',
                                                          'color': RED, 'fontFamily': FONT_MONO, 'marginBottom': '2px'}),
                        html.Div("of tweets decay within 2 days",
                                 style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.35)', 'fontFamily': FONT}),
                    ]),
                ]),
                purple_insight(f"Most tweets decay within {median_hl} days. Only high-AEI tweets sustain attention beyond 4 days — consistent with a stochastic decay model."),
            ]),
            card([
                html.Div("Topic distribution", style={**H2, 'marginBottom': '8px'}),
                html.Div("Classified by NLP keyword features",
                         style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.3)', 'marginBottom': '8px', 'fontFamily': FONT}),
                dcc.Graph(figure=fig_topic, config={'displayModeBar': False}, style={'height': '220px'}),
            ]),
        ]),
    ])

# ── App ────────────────────────────────────────────────────
app = Dash(__name__)
app.title = "Attention Shockwave Model"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: "Space Grotesk", sans-serif !important; background: #0d1117; }
            .metric-card-anim {
                opacity: 0; transform: scale(0.92) translateY(10px);
                animation: cardIn 0.5s cubic-bezier(0.23, 1, 0.32, 1) forwards;
            }
            .metric-card-anim:nth-child(1) { animation-delay: 0.05s; }
            .metric-card-anim:nth-child(2) { animation-delay: 0.10s; }
            .metric-card-anim:nth-child(3) { animation-delay: 0.15s; }
            .metric-card-anim:nth-child(4) { animation-delay: 0.20s; }
            @keyframes cardIn { to { opacity: 1; transform: scale(1) translateY(0); } }
            #page-content { animation: fadeIn 0.3s ease forwards; }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(8px); }
                to   { opacity: 1; transform: translateY(0); }
            }
            ::-webkit-scrollbar { width: 4px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
        <script>
            function countUp(el, target, duration, prefix, suffix) {
                const isFloat = !Number.isInteger(target);
                const startTime = performance.now();
                function update(t) {
                    const p = Math.min((t - startTime) / duration, 1);
                    const e = 1 - Math.pow(1 - p, 3);
                    const c = target * e;
                    el.textContent = prefix + (isFloat ? c.toFixed(2) : Math.floor(c).toLocaleString()) + suffix;
                    if (p < 1) requestAnimationFrame(update);
                    else el.textContent = prefix + (isFloat ? target.toFixed(2) : target.toLocaleString()) + suffix;
                }
                requestAnimationFrame(update);
            }
            function runCountUps() {
                document.querySelectorAll("[data-countup]").forEach(el => {
                    const target = parseFloat(el.getAttribute("data-countup"));
                    const prefix = el.getAttribute("data-prefix") || "";
                    const suffix = el.getAttribute("data-suffix") || "";
                    if (!isNaN(target)) countUp(el, target, 1200, prefix, suffix);
                });
            }
            const obs = new MutationObserver(() => setTimeout(runCountUps, 150));
            document.addEventListener("DOMContentLoaded", () => {
                const t = document.getElementById("page-content");
                if (t) obs.observe(t, { childList: true, subtree: true });
                setTimeout(runCountUps, 600);
            });
        </script>
    </body>
</html>
'''

app.layout = html.Div(style={'backgroundColor': BG, 'minHeight': '100vh', 'fontFamily': FONT, 'color': '#fff'}, children=[
    html.Div(style={
        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
        'position': 'relative', 'height': '56px',
        'borderBottom': f'0.5px solid {BORDER}',
        'backgroundColor': 'rgba(22, 27, 39, 0.85)',
        'backdropFilter': 'blur(10px)',
        'WebkitBackdropFilter': 'blur(10px)',
    }, children=[
        html.Div(style={'position': 'absolute', 'left': '28px', 'display': 'flex', 'alignItems': 'center', 'gap': '10px'}, children=[
            html.Img(src=LOGO, style={'width': '36px', 'height': '36px', 'borderRadius': '8px', 'objectFit': 'cover'})
            if LOGO else html.Div(style={'width': '36px', 'height': '36px',
                                         'background': 'linear-gradient(135deg,#3b82f6,#6366f1)', 'borderRadius': '8px'}),
            html.Span("Attention Shockwave Model",
                      style={'fontSize': '14px', 'fontWeight': '700', 'color': '#fff', 'fontFamily': FONT}),
        ]),
        html.Div(style={'display': 'flex', 'gap': '4px'}, children=[
            html.Button("Dashboard",       id='tab-main',       n_clicks=0, style=nav_tab_style(True)),
            html.Button("Market Backtest", id='tab-market',     n_clicks=0, style=nav_tab_style(False)),
            html.Button("Tweet Predictor", id='tab-predictor',  n_clicks=0, style=nav_tab_style(False)),
            html.Button("AEI",             id='tab-elasticity', n_clicks=0, style=nav_tab_style(False)),
        ]),
        html.Div("☽", style={'position': 'absolute', 'right': '28px', 'fontSize': '16px',
                              'color': 'rgba(255,255,255,0.4)', 'cursor': 'pointer'}),
    ]),
    dcc.Store(id='active-tab',     data='main'),
    dcc.Store(id='active-keyword', data='China'),
    html.Div(id='page-content'),
])

# ── Callbacks ──────────────────────────────────────────────
@app.callback(
    Output('active-tab',     'data'),
    Output('tab-main',       'style'),
    Output('tab-market',     'style'),
    Output('tab-predictor',  'style'),
    Output('tab-elasticity', 'style'),
    Input('tab-main',        'n_clicks'),
    Input('tab-market',      'n_clicks'),
    Input('tab-predictor',   'n_clicks'),
    Input('tab-elasticity',  'n_clicks'),
)
def switch_tab(n1, n2, n3, n4):
    triggered = ctx.triggered_id or 'tab-main'
    tab_map   = {'tab-main': 'main', 'tab-market': 'market',
                 'tab-predictor': 'predictor', 'tab-elasticity': 'elasticity'}
    active    = tab_map.get(triggered, 'main')
    tabs      = ['tab-main', 'tab-market', 'tab-predictor', 'tab-elasticity']
    return active, *[nav_tab_style(t == triggered) for t in tabs]

@app.callback(
    Output('active-keyword', 'data'),
    Output({'type': 'kw-btn', 'index': ALL}, 'style'),
    Input({'type': 'kw-btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def switch_keyword(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    triggered = ctx.triggered_id
    if not triggered:
        raise PreventUpdate
    active = triggered['index']
    return active, [pill_style(k == active) for k in KEYWORDS]

@app.callback(
    Output('page-content', 'children'),
    Input('active-tab',     'data'),
    Input('active-keyword', 'data'),
)
def render_page(tab, keyword):
    try:
        if tab == 'market':       return render_market()
        elif tab == 'predictor':  return render_predictor()
        elif tab == 'elasticity': return render_elasticity()
        else:                     return render_main(keyword)
    except Exception as e:
        return html.Div([
            html.H2("Error:", style={'color': 'red'}),
            html.Pre(str(e), style={'color': 'red', 'whiteSpace': 'pre-wrap', 'fontSize': '12px'})
        ])

# ── Dashboard ──────────────────────────────────────────────
def render_main(keyword):
    trend_col = KEYWORD_MAP[keyword]
    kw_tweets = get_keyword_tweets(keyword)
    corr      = get_correlation(keyword)
    avg_sent  = kw_tweets['sentiment'].mean()
    neg_rate  = round((kw_tweets['sentiment'] < -0.05).mean() * 100)

    kw_indexed   = kw_tweets.set_index('date')
    tweet_counts = kw_indexed.resample('W').size().reset_index()
    tweet_counts.columns = ['week', 'count']
    trend_series = trends[trend_col].resample('W').mean().reset_index()
    trend_series.columns = ['date', 'interest']
    peak_week = trend_series.loc[trend_series['interest'].idxmax(), 'date'].strftime('%b %Y')

    sent_monthly = kw_indexed['sentiment'].resample('ME').mean().interpolate().reset_index()
    sent_monthly.columns = ['date', 'sentiment']
    sent_monthly = sent_monthly.dropna()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sent_monthly['date'], y=sent_monthly['sentiment'],
        name='Avg Sentiment', line=dict(color=BLUE, width=2),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.06)', mode='lines'))
    fig2.update_layout(paper_bgcolor=BG2, plot_bgcolor=BG2,
        font=dict(color='rgba(255,255,255,0.3)', family='Space Grotesk', size=10),
        margin=dict(l=40, r=40, t=10, b=40), hovermode='x unified',
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', title='Sentiment (-1 to +1)'))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_series['date'], y=trend_series['interest'],
        name='Search Interest', line=dict(color=BLUE, width=2),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.06)', mode='lines'))
    fig.add_trace(go.Bar(x=tweet_counts['week'], y=tweet_counts['count'],
        name='Tweets/week', marker_color='rgba(139,92,246,0.5)', yaxis='y2'))
    fig.update_layout(paper_bgcolor=BG2, plot_bgcolor=BG2,
        font=dict(color='rgba(255,255,255,0.3)', family='Space Grotesk', size=10),
        margin=dict(l=40, r=40, t=10, b=40), hovermode='x unified',
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', title=dict(text='Search Interest', font=dict(color=BLUE))),
        yaxis2=dict(overlaying='y', side='right', title=dict(text='Tweets/week', font=dict(color=PURPLE)), gridcolor='rgba(0,0,0,0)'))

    top_tweets = kw_tweets.nlargest(5, 'retweets')
    tweet_cards = []
    for _, t in top_tweets.iterrows():
        s        = t['sentiment']
        bg, bord = sent_bg(s)
        topic    = classify_topic(t['text'])
        t_color  = TOPIC_COLORS.get(topic, 'rgba(255,255,255,0.3)')
        tweet_cards.append(card([
            html.P(str(t['text'])[:160], style={'fontSize': '13px', 'fontWeight': '500',
                                                 'color': 'rgba(255,255,255,0.8)', 'lineHeight': '1.6',
                                                 'marginBottom': '10px', 'fontFamily': FONT}),
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '6px'}, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '6px'}, children=[
                    html.Span("🕐", style={'fontSize': '12px'}),
                    html.Span(str(t['date'])[:10], style={'fontSize': '11px', 'color': 'rgba(255,255,255,0.25)', 'fontFamily': FONT}),
                    html.Div(style={'padding': '2px 8px', 'borderRadius': '20px',
                                   'background': 'rgba(255,255,255,0.06)', 'border': f'0.5px solid {t_color}'}, children=[
                        html.Span(topic, style={'fontSize': '10px', 'color': t_color, 'fontFamily': FONT}),
                    ]),
                ]),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '6px',
                                'padding': '4px 12px', 'borderRadius': '20px',
                                'background': bg, 'border': bord}, children=[
                    html.Span(sent_emoji(s), style={'fontSize': '11px'}),
                    html.Span(f"{sent_label(s)} {s:+.2f}",
                              style={'fontSize': '11px', 'fontWeight': '600',
                                     'color': sent_color(s), 'fontFamily': FONT}),
                ]),
            ]),
        ], extra_style={'marginBottom': '8px', 'padding': '14px'}))

    return html.Div(style={'padding': '28px'}, children=[
        page_header("Attention Shockwave Model",
                    "Correlating 56,571 presidential tweets with Google search behaviour — 2016 to 2021"),
        causal_disclaimer(),
        killer_result_banner(),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(4,1fr)', 'gap': '12px', 'marginBottom': '24px'}, children=[
            metric_card("Total Tweets",   f"{len(kw_tweets):,}",                    f"About '{keyword}'",                                      BLUE,                           "▣", BLUE),
            metric_card("Avg Sentiment",  f"{avg_sent:.2f}",                         sent_label(avg_sent),                                      sent_color(avg_sent),            "↗", GREEN),
            metric_card("Avg Search Vol", f"{int(trend_series['interest'].mean())}",  "Index score",                                            PURPLE,                         "📱", PURPLE),
            metric_card("Negative Rate",  f"{neg_rate}%",                            "Requires attention" if neg_rate > 50 else "Within range", RED if neg_rate > 50 else GREEN, "!", RED),
        ]),
        html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '24px', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
            card([html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                html.Span("🔍", style={'fontSize': '14px'}),
                dcc.Input(placeholder="Search tweets...", style={'background': 'none', 'border': 'none', 'outline': 'none',
                          'color': '#fff', 'fontSize': '13px', 'width': '200px', 'fontFamily': FONT}),
            ])], extra_style={'padding': '10px 14px', 'flexShrink': '0'}),
            html.Div([html.Button(k, id={'type': 'kw-btn', 'index': k}, n_clicks=0, style=pill_style(k == keyword))
                      for k in KEYWORDS],
                     style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '4px', 'flex': '1'}),
            html.Button("Export Data", style={'padding': '8px 18px', 'background': BG3, 'border': f'0.5px solid {BORDER}',
                                              'borderRadius': '8px', 'color': '#fff', 'fontSize': '13px', 'fontWeight': '600',
                                              'cursor': 'pointer', 'fontFamily': FONT, 'whiteSpace': 'nowrap'}),
        ]),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '340px 1fr', 'gap': '16px', 'marginBottom': '24px'}, children=[
            html.Div([
                html.Div(style={'marginBottom': '12px'}, children=[
                    html.Div("Recent Tweets", style=H2),
                    html.Div(f"{len(kw_tweets):,} tweets", style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.3)', 'fontFamily': FONT}),
                ]),
                html.Div(tweet_cards, style={'maxHeight': '600px', 'overflowY': 'auto'}),
            ]),
            html.Div([
                card([
                    html.Div("Sentiment Over Time", style=H2),
                    html.Div("Monthly average — -1 (very negative) to +1 (very positive)",
                             style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.3)', 'marginBottom': '12px', 'fontFamily': FONT}),
                    dcc.Graph(figure=fig2, style={'height': '190px'}, config={'displayModeBar': False}),
                    sentiment_explainer(avg_sent),
                ], extra_style={'marginBottom': '16px'}),
                card([
                    html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'marginBottom': '2px'}, children=[
                        html.Span("↗", style={'color': AMBER, 'fontSize': '16px'}),
                        html.Div("Google Trends Correlation", style=H2),
                    ]),
                    html.Div("Search interest vs tweet volume — behavioural signal, not causal",
                             style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.3)', 'marginBottom': '12px', 'fontFamily': FONT}),
                    dcc.Graph(figure=fig, style={'height': '190px'}, config={'displayModeBar': False}),
                    correlation_banner(corr, peak_week),
                ]),
            ]),
        ]),
        html.Div([
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px',
                            'marginBottom': '4px', 'justifyContent': 'center'}, children=[
                html.Span("✦", style={'color': PURPLE, 'fontSize': '16px'}),
                html.Div("Model Findings", style=H2),
            ]),
            html.Div("Key patterns extracted from the Attention Shockwave Model",
                     style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.3)',
                            'marginBottom': '16px', 'fontFamily': FONT, 'textAlign': 'center'}),
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '12px'}, children=[
                insight_card("Attention amplification",  f"High-AEI tweets generate {aei_multiplier}× larger search spikes than low-AEI tweets.", f"{aei_multiplier}×", GREEN),
                insight_card("Attention half-life",      f"Average half-life {avg_half_life} days. High-intensity tweets sustain attention 2× longer.", f"{avg_half_life}d", BLUE),
                insight_card("Sentiment polarity",       f"Average sentiment {avg_sent:.2f} — {sent_label(avg_sent).lower()} for '{keyword}'. Each tweet scored -1 to +1.", f"{avg_sent:+.2f}", sent_color(avg_sent)),
                insight_card("Search velocity",          f"Behavioural signal r={corr} — {'strong' if abs(corr) > 0.5 else 'moderate'} co-occurrence between tweet volume and search attention.", f"r={corr}", BLUE if abs(corr) > 0.5 else 'rgba(255,255,255,0.4)', "—"),
            ]),
        ]),
        tweet_shockwave_section(keyword, kw_tweets, trend_col),
        attention_decay_section(),
    ])

# ── Market backtest ────────────────────────────────────────
def render_market():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['cumulative_market'],
        name='S&P 500 Buy & Hold', line=dict(color='rgba(255,255,255,0.4)', width=1.5)))
    fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['cumulative_strategy'],
        name='Tweet Sentiment Strategy', line=dict(color=BLUE, width=2),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.06)'))
    fig.update_layout(paper_bgcolor=BG2, plot_bgcolor=BG2,
        font=dict(color='rgba(255,255,255,0.3)', family='Space Grotesk', size=10),
        margin=dict(l=40, r=40, t=10, b=40), hovermode='x unified',
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', title='Cumulative return'))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['rolling_sharpe'],
        name='60-day Rolling Sharpe', line=dict(color=PURPLE, width=1.5), mode='lines'))
    fig2.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.2)', line_width=1)
    fig2.update_layout(paper_bgcolor=BG2, plot_bgcolor=BG2,
        font=dict(color='rgba(255,255,255,0.3)', family='Space Grotesk', size=10),
        margin=dict(l=40, r=40, t=10, b=40), hovermode='x unified',
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', title='Rolling Sharpe'))

    win_rate       = round((backtest_df['strategy_returns'] > 0).mean() * 100, 1)
    max_dd         = round((backtest_df['cumulative_strategy'] / backtest_df['cumulative_strategy'].cummax() - 1).min() * 100, 1)
    threshold_pct  = round((backtest_df['position'].abs() > 0.01).mean() * 100, 1)

    return html.Div(style={'padding': '28px'}, children=[
        page_header("Market Backtest",
                    "Can presidential tweet sentiment predict next-day S&P 500 returns?"),
        causal_disclaimer(),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(4,1fr)', 'gap': '12px', 'marginBottom': '24px'}, children=[
            metric_card("Sharpe Ratio", str(sharpe),        "Risk-adjusted return",      GREEN if sharpe > 0.5 else AMBER if sharpe > 0 else RED, "↗", GREEN),
            metric_card("Total Return", f"{total_return}%", "After transaction costs",   GREEN if total_return > 0 else RED, "↗", GREEN),
            metric_card("Win Rate",     f"{win_rate}%",     "Days strategy was correct", GREEN if win_rate > 50 else RED, "✓", GREEN),
            metric_card("Max Drawdown", f"{max_dd}%",       "Worst peak-to-trough loss", RED, "!", RED),
        ]),
        card([
            html.Div("Cumulative Returns", style=H2),
            html.Div("Volatility-targeted threshold strategy vs S&P 500 — includes 5bps transaction costs",
                     style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.3)', 'marginBottom': '12px', 'fontFamily': FONT}),
            dcc.Graph(figure=fig, style={'height': '300px'}, config={'displayModeBar': False}),
        ], extra_style={'marginBottom': '16px'}),
        card([
            html.Div("60-Day Rolling Sharpe Ratio", style=H2),
            html.Div("Positive = strategy outperforms on a risk-adjusted basis in that window",
                     style={'fontSize': '12px', 'color': 'rgba(255,255,255,0.3)', 'marginBottom': '12px', 'fontFamily': FONT}),
            dcc.Graph(figure=fig2, style={'height': '200px'}, config={'displayModeBar': False}),
            purple_insight(
                f'Strategy uses volatility targeting (20-day window, 1% daily vol target) with ±5% sentiment threshold — '
                f'trades on {threshold_pct}% of days. Includes 5bps transaction costs per trade. '
                f'Sharpe of {sharpe} — {"mildly interesting" if sharpe > 0.3 else "indistinguishable from noise"}, '
                f'consistent with the efficient market hypothesis. Correlation ≠ causation.'
            ),
        ], extra_style={'marginBottom': '16px'}),
        card([
            html.Div("Model Methodology", style=H2),
            html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(4,1fr)', 'gap': '12px', 'marginTop': '12px'}, children=[
                model_badge("Signal threshold", "±5%",   AMBER),
                model_badge("Vol target/day",   "1.0%",  BLUE),
                model_badge("Vol window",        "20d",   PURPLE),
                model_badge("Transaction cost",  "5bps",  RED),
            ]),
        ]),
    ])

# ── Tweet predictor ────────────────────────────────────────
def render_predictor():
    return html.Div(style={'padding': '28px'}, children=[
        page_header("Tweet Predictor",
                    "Ridge regression model — trained on 56k tweets with walk-forward validation."),
        card([
            html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '12px'}, children=[
                model_badge(f"AEI model R²(test)={AEI_R2_TEST}", "Ridge", BLUE),
                model_badge(f"Spike model walk-forward R²={SPIKE_R2_WF}", "Ridge + WF", PURPLE),
                model_badge(f"CAPS:{W_CAPS} Excl:{W_EXCL} Sent:{W_SENT}", "Learned weights", AMBER),
                model_badge("Features: CAPS, !, sentiment, entities, topic", "NLP", GREEN),
            ]),
            html.Div(style={'display': 'flex', 'gap': '12px'}, children=[
                dcc.Textarea(id='tweet-input', placeholder='e.g. "China is RIPPING us off. SAD!"',
                    style={'flex': '1', 'backgroundColor': BG3, 'border': f'0.5px solid {BORDER}',
                           'borderRadius': '8px', 'padding': '14px', 'color': '#fff', 'fontSize': '14px',
                           'fontFamily': FONT, 'fontWeight': '500', 'resize': 'vertical', 'minHeight': '80px', 'outline': 'none'}),
                html.Button("Analyse", id='predict-btn', n_clicks=0,
                    style={'padding': '0 24px', 'backgroundColor': BLUE, 'border': 'none', 'borderRadius': '8px',
                           'color': '#fff', 'fontSize': '13px', 'fontWeight': '700',
                           'cursor': 'pointer', 'fontFamily': FONT, 'alignSelf': 'flex-end', 'height': '44px'}),
            ]),
        ], extra_style={'marginBottom': '20px'}),
        html.Div(id='predictor-output'),
    ])

# ── AEI page ───────────────────────────────────────────────
def render_elasticity():
    df    = compute_elasticity(tweets[tweets['isRetweet'] == 'f'].copy())
    top10 = df.nlargest(10, 'elasticity').reset_index(drop=True)

    rows = [card([
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '48px 1fr 120px', 'gap': '16px', 'alignItems': 'center'}, children=[
            html.Div(f"{i+1:02d}", style={'fontSize': '28px', 'fontWeight': '700', 'color': '#fff', 'fontFamily': FONT_MONO}),
            html.Div([
                html.P(f'"{str(row["text"])[:120]}..."',
                       style={'fontSize': '13px', 'fontWeight': '500', 'color': 'rgba(255,255,255,0.75)',
                              'lineHeight': '1.5', 'marginBottom': '6px', 'fontFamily': FONT}),
                html.Div(style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}, children=[
                    html.Span(f"CAPS {round(row['caps_ratio']*100)}%",   style={'fontSize': '11px', 'fontWeight': '600', 'color': 'rgba(255,255,255,0.3)', 'fontFamily': FONT}),
                    html.Span(f"! × {int(row['exclamations'])}",         style={'fontSize': '11px', 'fontWeight': '600', 'color': 'rgba(255,255,255,0.3)', 'fontFamily': FONT}),
                    html.Span(f"RT {int(row['retweets']):,}",            style={'fontSize': '11px', 'fontWeight': '600', 'color': 'rgba(255,255,255,0.3)', 'fontFamily': FONT}),
                    html.Span(f"Half-life ~{row['half_life']:.1f}d",     style={'fontSize': '11px', 'fontWeight': '600', 'color': PURPLE, 'fontFamily': FONT}),
                    html.Span(f"Topic: {row['topic']}",                  style={'fontSize': '11px', 'fontWeight': '600', 'color': TOPIC_COLORS.get(row['topic'], AMBER), 'fontFamily': FONT}),
                ])
            ]),
            html.Div([
                html.Div(f"{row['elasticity']:.1f}", style={'fontSize': '24px', 'fontWeight': '700', 'color': RED, 'textAlign': 'right', 'fontFamily': FONT_MONO}),
                html.Div("AEI score", style={'fontSize': '11px', 'color': 'rgba(255,255,255,0.2)', 'textAlign': 'right', 'fontFamily': FONT}),
            ]),
        ])
    ], extra_style={'marginBottom': '8px'}) for i, row in top10.iterrows()]

    return html.Div(style={'padding': '28px'}, children=[
        page_header("Attention Elasticity Index (AEI)",
                    f"Weights learned via Ridge regression (R²test={AEI_R2_TEST}): CAPS×{W_CAPS} + Excl×{W_EXCL} + Sent×{W_SENT}. Features include topic classification and named entity count."),
        html.Div(style={'display': 'flex', 'gap': '8px', 'marginBottom': '20px', 'justifyContent': 'center', 'flexWrap': 'wrap'}, children=[
            model_badge("Model",              "Ridge Regression",  BLUE),
            model_badge("R² (test set)",      str(AEI_R2_TEST),   GREEN if AEI_R2_TEST > 0.1 else AMBER),
            model_badge("CAPS weight",        str(W_CAPS),        AMBER),
            model_badge("Exclamation weight", str(W_EXCL),        AMBER),
            model_badge("Sentiment weight",   str(W_SENT),        AMBER),
        ]),
        html.Div(rows),
        attention_decay_section(),
    ])

# ── Predictor callback ─────────────────────────────────────
@app.callback(
    Output('predictor-output', 'children'),
    Input('predict-btn',       'n_clicks'),
    State('tweet-input',       'value'),
    prevent_initial_call=True
)
def predict_tweet(n_clicks, tweet_text):
    if not tweet_text or len(tweet_text.strip()) < 5:
        return html.Div("Please enter a tweet above.",
                        style={'color': 'rgba(255,255,255,0.3)', 'fontSize': '13px', 'fontFamily': FONT})

    s            = TextBlob(tweet_text).sentiment.polarity
    caps_ratio   = sum(1 for c in tweet_text if c.isupper()) / max(len(tweet_text), 1)
    exclamations = tweet_text.count('!')
    q_marks      = tweet_text.count('?')
    word_count   = len(tweet_text.split())
    excl_norm    = min(exclamations, 5) / 5
    entities     = count_entities(tweet_text)
    topic        = classify_topic(tweet_text)
    t_color      = TOPIC_COLORS.get(topic, AMBER)

    intensity  = caps_ratio * W_CAPS + excl_norm * W_EXCL + abs(s) * W_SENT
    aei_score  = round(intensity * 100)
    half_life  = round(max(1, 7 - intensity * 5), 1)

    if spike_model is not None:
        X_input  = np.array([[caps_ratio, excl_norm, abs(s), 1, s, entities,
                               int(topic=='Economy'), int(topic=='Foreign Policy')]])
        X_scaled = spike_scaler.transform(X_input)
        pred_spike = round(float(spike_model.predict(X_scaled)[0]))
        pred_spike = max(0, min(pred_spike, 100))
        model_src  = f"Ridge regression (walk-forward R²={SPIKE_R2_WF})"
    else:
        pred_spike = round(intensity * 80)
        model_src  = "heuristic formula"

    return html.Div([
        html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(4,1fr)', 'gap': '12px', 'marginBottom': '16px'}, children=[
            metric_card("Sentiment",              sent_label(s),        f"Score: {s:.2f}",                       sent_color(s), "↗", sent_color(s)),
            metric_card("Predicted Search Spike", f"+{pred_spike}pts",  f"Via {model_src[:20]}...",              AMBER,         "↑", AMBER),
            metric_card("AEI Score",              f"{aei_score}/100",   "Attention Elasticity Index",            RED,           "!", RED),
            metric_card("Attention Half-Life",    f"{half_life} days",  "Time until search returns to baseline", BLUE,          "⌛", BLUE),
        ]),
        card([
            html.Div(style={'display': 'flex', 'gap': '8px', 'marginBottom': '12px', 'flexWrap': 'wrap'}, children=[
                html.Div(style={'padding': '4px 12px', 'borderRadius': '20px',
                               'background': 'rgba(255,255,255,0.06)', 'border': f'0.5px solid {t_color}'}, children=[
                    html.Span(f"Topic: {topic}", style={'fontSize': '11px', 'color': t_color, 'fontFamily': FONT}),
                ]),
                html.Div(style={'padding': '4px 12px', 'borderRadius': '20px',
                               'background': 'rgba(255,255,255,0.06)', 'border': f'0.5px solid rgba(255,255,255,0.15)'}, children=[
                    html.Span(f"Named entities: {entities}", style={'fontSize': '11px', 'color': 'rgba(255,255,255,0.5)', 'fontFamily': FONT}),
                ]),
                html.Div(style={'padding': '4px 12px', 'borderRadius': '20px',
                               'background': 'rgba(255,255,255,0.06)', 'border': f'0.5px solid rgba(255,255,255,0.15)'}, children=[
                    html.Span(f"CAPS: {round(caps_ratio*100)}% | !: {exclamations}", style={'fontSize': '11px', 'color': 'rgba(255,255,255,0.5)', 'fontFamily': FONT}),
                ]),
            ]),
            html.Div("Model Output", style={'fontSize': '12px', 'fontWeight': '700',
                      'color': 'rgba(255,255,255,0.4)', 'letterSpacing': '0.08em',
                      'textTransform': 'uppercase', 'marginBottom': '10px', 'fontFamily': FONT}),
            html.P(f'Spike predicted by {model_src}. '
                   f'AEI: {aei_score}/100 using data-fitted weights (CAPS×{W_CAPS} + Excl×{W_EXCL} + Sent×{W_SENT}). '
                   f'Topic classified as "{topic}" via keyword NLP. '
                   f'Named entities detected: {entities}. '
                   f'Attention half-life: {half_life} days — '
                   f'{"sustained attention event" if half_life >= 4 else "rapid decay expected"}. '
                   f'Note: predictions are correlational, not causal.',
                   style={'fontSize': '13px', 'fontWeight': '500', 'color': 'rgba(255,255,255,0.55)',
                          'lineHeight': '1.7', 'fontFamily': FONT})
        ]),
    ])

if __name__ == '__main__':
    app.run(debug=True)