import os, uuid
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Optional

# =========================================================
# Page Config (한 번만!)
# =========================================================
st.set_page_config(page_title="Google Merchandise Sale Prediction", layout="wide")

# =========================================================
# 메인 타이틀
# =========================================================
st.markdown(
    """
    <h1 style="font-size:40px; font-weight:bold;">
        <span style="color:#4285F4;">G</span>
        <span style="color:#DB4437;">o</span>
        <span style="color:#F4B400;">o</span>
        <span style="color:#4285F4;">g</span>
        <span style="color:#0F9D58;">l</span>
        <span style="color:#DB4437;">e</span>
        Merchandise Sale Prediction
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("<hr style='margin: 40px 0; border: 0; border-top: 1px solid #444;'>", unsafe_allow_html=True)

st.title("Overview")

# =========================================================
# 데이터 로드
# =========================================================
import io, zipfile, requests

CSV_PATH = r".\google_with_cluster3_aha.csv"
ZIP_URL  = "https://github.com/gigigilt/Team-Project2/raw/main/google_with_cluster3_aha.zip"

@st.cache_data(show_spinner=True, ttl=3600, max_entries=1)
def load_data() -> pd.DataFrame:
    # 1) GitHub zip 시도
    try:
        r = requests.get(ZIP_URL, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            # zip 안 첫 번째 파일 가져오기
            fname = z.namelist()[0]
            with z.open(fname) as f:
                return pd.read_csv(f, engine="python", on_bad_lines="skip")
    except Exception as e:
        st.warning(f"GitHub zip 로드 실패 → 로컬 CSV 시도 ({e})")

    # 2) 로컬 CSV fallback
    try:
        return pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")
    except Exception as e:
        st.error(f"CSV 로드 실패: {e}")
        return pd.DataFrame()

df = load_data()


# =========================================================
# 공통 상수/유틸
# =========================================================
GOOGLE_BLUE   = "#4285F4"
GOOGLE_YELLOW = "#F4B400"
GOOGLE_RED    = "#DB4437"
GOOGLE_GREY   = "#9AA0A6"

CLUSTER_COLORS = {"0": GOOGLE_BLUE, "1": GOOGLE_YELLOW, "2": GOOGLE_RED}
DESIRED_ORDER = ["0", "1", "2"]

def ukey(name): 
    return f"{name}-{uuid.uuid4().hex[:8]}"

def to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    try:
        return (pd.to_numeric(s, errors="coerce").fillna(0) > 0)
    except Exception:
        ss = s.astype(str).str.lower().str.strip()
        return ss.isin({"1","true","t","y","yes"})

def ensure_dt(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    # is_datetime64tz_dtype 대체 (경고 제거)
    if isinstance(out.dtype, pd.DatetimeTZDtype):
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out

# =========================================================
# Overview 카드 (세션/유저/체류/페이지뷰)
# =========================================================
df["sessionId"] = df["fullVisitorId"].astype(str) + "_" + df["visitStartTime"].astype(str)

total_sessions = df["sessionId"].nunique()
unique_users   = df["fullVisitorId"].nunique()
sessions_per_user = total_sessions / unique_users if unique_users else 0

c1, c2, c3 = st.columns(3)
c1.metric("총 세션 수", f"{total_sessions:,}")
c2.metric("총 유저 수", f"{unique_users:,}")
c3.metric("유저당 평균 방문 횟수", f"{sessions_per_user:.2f}")

avg_time_minutes = (pd.to_numeric(df["totalTimeOnSite"], errors="coerce").fillna(0).mean()) / 60
avg_pageviews    = pd.to_numeric(df["totalPageviews"], errors="coerce").fillna(0).mean()

c1, c2 = st.columns(2)
c1.metric("평균 체류시간 (분)", f"{avg_time_minutes:.1f}")
c2.metric("세션당 평균 페이지뷰", f"{avg_pageviews:.2f}")

# 신규/재방문 & Cart 전환율
df["visit_type"] = np.where(pd.to_numeric(df["isFirstVisit"], errors="coerce") == 1, "신규", "재방문")
added_num = pd.to_numeric(df["addedToCart"], errors="coerce")
added_str = df["addedToCart"].astype(str).str.lower()
df["cart_flag"] = (added_num.fillna(0) > 0) | (added_str.isin(["1", "true", "t", "y", "yes"]))

summary = (
    df.groupby("visit_type")
      .agg(sessions=("fullVisitorId", "count"), cart_sessions=("cart_flag", "sum"))
      .reset_index()
)
summary["cart_rate"] = summary["cart_sessions"] / summary["sessions"]

summary_view = summary.copy()
summary_view["session_ratio(%)"] = (summary_view["sessions"] / summary_view["sessions"].sum() * 100).round(1).astype(str) + "%"
summary_view["cart_rate(%)"]     = (summary_view["cart_rate"] * 100).round(1).astype(str) + "%"

st.subheader("신규/재방문 회원별 Cart 전환율")
left, right = st.columns([1, 1])
with left:
    # dataframe 은 width 문자열을 지원하지 않으므로 use_container_width 사용
    st.dataframe(summary_view[["visit_type", "session_ratio(%)", "cart_rate(%)"]], use_container_width=True)
with right:
    fig = px.bar(
        summary, x="visit_type", y="cart_rate",
        text=(summary["cart_rate"] * 100).round(1).astype(str) + "%",
        labels={"visit_type": "방문 유형", "cart_rate": "Cart 전환율"},
        category_orders={"visit_type": ["신규", "재방문"]},
        color="visit_type",
        color_discrete_map={"신규": GOOGLE_BLUE, "재방문": GOOGLE_YELLOW}
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_traces(textposition="outside")
    fig.update_layout(bargap=0.7, bargroupgap=0.1)
    st.plotly_chart(fig, width="stretch", key=ukey("overview-nr-bar"))

# AHA 달성률
st.subheader("전체 AHA 달성률")
aha = to_bool(df["AhaMoment"])
aha_rate = float(aha.mean()) if len(aha) else 0.0
st.metric("Aha 달성률", f"{aha_rate*100:.1f}%")

pie_df = pd.DataFrame({"Aha": ["TRUE", "FALSE"], "count": [int(aha.sum()), int((~aha).sum())]})
fig = px.pie(
    pie_df, names="Aha", values="count", hole=0.35, color="Aha",
    color_discrete_map={"TRUE": GOOGLE_YELLOW, "FALSE": GOOGLE_BLUE}
)
fig.update_traces(textposition="inside", texttemplate="%{label}: %{percent:.1%}")
col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(fig, width="stretch", key=ukey("overview-aha-pie"))
with col2:
    st.markdown(
        """
        **AHA 모먼트 측정 기준**  
        - 처음으로 상품을 장바구니에 담았을 때  
        - 세션 시간(TimePerSession)이 일정 기준(예: 3 이상)에 도달했을 때
        """
    )

# 세션 퍼널
st.subheader("퍼널 분석")
step1 = (pd.to_numeric(df["totalPageviews"], errors="coerce").fillna(0) >= 1)
step2 = (pd.to_numeric(df["productPagesViewed"], errors="coerce").fillna(0) >= 1)
step3 = (pd.to_numeric(df["addedToCart"], errors="coerce").fillna(0) >= 1)
n1 = int(step1.sum()); n2 = int((step1 & step2).sum()); n3 = int((step1 & step2 & step3).sum())
conv12 = n2 / n1 if n1 else 0; conv23 = n3 / n2 if n2 else 0; conv_overall = n3 / n1 if n1 else 0

k1, k2, k3 = st.columns(3)
k1.metric("페이지뷰→제품상세 전환율", f"{conv12*100:.1f}%")
k2.metric("제품상세→카트추가 전환율", f"{conv23*100:.1f}%")
k3.metric("페이지뷰→카트추가 전환율", f"{conv_overall*100:.1f}%")

funnel_df = pd.DataFrame({"Step": ["페이지뷰 ≥1", "제품상세 ≥1", "카트추가 ≥1"], "Sessions": [n1, n2, n3]})
pct_prev = [1.0, (n2 / n1) if n1 else 0, (n3 / n2) if n2 else 0]
funnel_df["Text"] = [f"{v:,.0f} ({p*100:.1f}%)" for v, p in zip(funnel_df["Sessions"], pct_prev)]
fig_funnel = px.funnel(funnel_df, x="Sessions", y="Step")
fig_funnel.update_traces(text=funnel_df["Text"], texttemplate="%{text}", textposition="inside", textinfo="none", marker_color=GOOGLE_BLUE)
st.plotly_chart(fig_funnel, width="stretch", key=ukey("overview-funnel"))

st.markdown("<hr style='margin: 40px 0; border: 0; border-top: 1px solid #444;'>", unsafe_allow_html=True)

# =========================================================
# Cluster Summary
# =========================================================
st.title("Cluster Summary Table")
data = {
    "고객군": ["탐색형 고객 (Explorers)", "비활성 고객 (Visitors)", "충성/핵심 고객 (Core Buyers)"],
    "유저 수": ["260,180", "305,542", "29,126"],
    "Recency (일)": [159.0, 164.4, 150.4],
    "Frequency (방문일수)": [1.21, 1.07, 1.75],
    "Cart Rate": [0.01, 0.00, 0.77],
    "Search Rate": [0.96, 0.01, 0.95],
    "Time/Session (정규화)": [1.19, 0.01, 5.32]
}
df_cluster = pd.DataFrame(data)
st.subheader("클러스터별 요약")
st.dataframe(df_cluster, use_container_width=True)

st.subheader("클러스터별 유저 비중")
data = {
    "Cluster": [0, 1, 2],
    "클러스터링 결과": ["탐색형 고객 (Explorers)", "비활성 고객 (Visitors)", "충성/핵심 고객 (Core Buyers)"],
    "유저 수": [260180, 305542, 29126]
}
df_cluster = pd.DataFrame(data)
df_cluster["label"] = df_cluster["클러스터링 결과"].str.strip()
order = ["탐색형 고객 (Explorers)", "비활성 고객 (Visitors)", "충성/핵심 고객 (Core Buyers)"]
colors = [GOOGLE_BLUE, GOOGLE_YELLOW, GOOGLE_RED]
fig = px.pie(
    df_cluster, names="label", values="유저 수", hole=0.3,
    category_orders={"label": order}, color="label", color_discrete_sequence=colors
)
fig.update_traces(textinfo="label+percent", pull=[0.02, 0.02, 0.05], rotation=90)
fig.update_layout(margin=dict(t=40, b=40, l=40, r=40), legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
st.plotly_chart(fig, width="stretch", key=ukey("cluster-pie"))

st.markdown("<hr style='margin: 40px 0; border: 0; border-top: 1px solid #444;'>", unsafe_allow_html=True)

# =========================================================
# AARRR — USER level
# =========================================================
st.title("AARRR Dashboard — USER level")

user_col     = "fullVisitorId"
date_col     = "date"
cluster_col  = "cluster"
detail_col   = "productPagesViewed"
cart_col     = "addedToCart"
device_col   = "deviceCategory"
country_col  = "country"

# 채널 컬럼 결정
if "trafficMedium" in df.columns:
    medium_col = "trafficMedium"
elif "trafficMed" in df.columns:
    medium_col = "trafficMed"
elif "trafficSource" in df.columns:
    medium_col = "trafficSource"
else:
    st.error("채널 차원을 위한 trafficMedium/trafficMed/trafficSource 중 하나가 필요합니다.")
    st.stop()

# 타입/클린
session_col = "_session_surrogate"
df[user_col] = df[user_col].astype("string").fillna("")
cnum = pd.to_numeric(df[cluster_col], errors="coerce").astype("Int64")
df = df[cnum.isin([0,1,2])].copy()
df[cluster_col] = cnum.astype("Int64").astype(str)
df[date_col] = ensure_dt(df[date_col])
df = df[df[date_col].notna()]
df[detail_col] = pd.to_numeric(df[detail_col], errors="coerce").fillna(0)
df[cart_col]   = to_bool(df[cart_col])
df["AhaMoment"] = to_bool(df["AhaMoment"])

# 세션 surrogate
if df[date_col].dt.time.astype(str).nunique() > 1:
    bucket = df[date_col].dt.floor("30min").astype(str)
else:
    bucket = df[date_col].dt.strftime("%Y-%m-%d")
df[session_col] = (df[user_col].astype(str) + "|" + bucket)

# 기간 필터(메인)
st.markdown("### 기간")
with st.expander("기간 필터 열기 / 닫기", expanded=False):
    min_d, max_d = df[date_col].min().date(), df[date_col].max().date()
    c1, c2 = st.columns(2)
    start = c1.date_input("시작일", value=min_d, key="start_date_main")
    end   = c2.date_input("종료일", value=max_d, key="end_date_main")

start_dt = pd.Timestamp(start)
end_dt   = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
dfp = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)].copy()
if dfp.empty:
    st.warning("선택 구간 데이터 없음"); st.stop()

def rep_mode(s: pd.Series) -> Optional[str]:
    m = pd.to_numeric(s, errors="coerce"); m = m[m.notna()]
    if m.empty: return None
    return str(int(m.mode().iat[0]))

# 최신 스냅샷(유저 최신 행)
df_user_last = (
    dfp.sort_values([user_col, date_col])
       .drop_duplicates(subset=[user_col], keep="last")
       [[user_col, cluster_col, "AhaMoment", cart_col]]
       .copy()
)
df_user_last[cluster_col] = df_user_last[cluster_col].astype(str)
df_user_last["AhaMoment"] = to_bool(df_user_last["AhaMoment"])
df_user_last["cart_bool"] = to_bool(df_user_last[cart_col])

# 기간 전체(any) + 대표 클러스터
rep_cluster_mode = (dfp.groupby(user_col)[cluster_col].apply(rep_mode).reset_index(name=cluster_col))
user_any = (
    dfp.groupby(user_col)
       .agg(detail_any=(detail_col, lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).any())),
            cart_any=(cart_col, "any"))
       .reset_index()
)
uf = user_any.merge(rep_cluster_mode, on=user_col, how="left")
uf["home"] = 1

# =========================================================
# 1) Activation
# =========================================================
st.header("1. Activation")

cluster_pop = (df_user_last.groupby(cluster_col)[user_col].nunique().rename("cluster_users").reset_index())
cluster_pop[cluster_col] = pd.Categorical(cluster_pop[cluster_col], categories=DESIRED_ORDER, ordered=True)
cluster_pop = cluster_pop.sort_values(cluster_col)

aha_true = (df_user_last.groupby(cluster_col)["AhaMoment"].sum().rename("aha_true").reset_index())

aha_rate_den_cluster = (
    cluster_pop.merge(aha_true, on=cluster_col, how="left")
               .fillna({"aha_true":0})
               .assign(aha_rate=lambda t: t["aha_true"] / t["cluster_users"].replace(0, np.nan))
               [[cluster_col,"cluster_users","aha_true","aha_rate"]]
)
aha_rate_den_cluster[cluster_col] = pd.Categorical(aha_rate_den_cluster[cluster_col], categories=DESIRED_ORDER, ordered=True)
aha_rate_den_cluster = aha_rate_den_cluster.sort_values(cluster_col)

aha_cart_true = ((df_user_last["AhaMoment"] & df_user_last["cart_bool"]).groupby(df_user_last[cluster_col]).sum().rename("aha_cart_true").reset_index())

aha_cart_overall_den_cluster = (
    cluster_pop.merge(aha_cart_true, on=cluster_col, how="left")
               .fillna({"aha_cart_true":0})
               .assign(aha_cart_overall=lambda t: t["aha_cart_true"] / t["cluster_users"].replace(0, np.nan))
               [[cluster_col,"cluster_users","aha_cart_true","aha_cart_overall"]]
)
aha_cart_overall_den_cluster[cluster_col] = pd.Categorical(aha_cart_overall_den_cluster[cluster_col], categories=DESIRED_ORDER, ordered=True)
aha_cart_overall_den_cluster = aha_cart_overall_den_cluster.sort_values(cluster_col)

fig_aha = px.bar(
    aha_rate_den_cluster, y=cluster_col, x="aha_rate", orientation="h",
    title="클러스터별 Aha 달성률 (분모=클러스터 인원)",
    color=cluster_col, color_discrete_map=CLUSTER_COLORS,
    labels={"aha_rate":"Aha 달성률", cluster_col:"Cluster"}
)
fig_aha.update_yaxes(categoryorder="array", categoryarray=DESIRED_ORDER)
fig_aha.update_traces(texttemplate="%{x:.1%}", textposition="outside", showlegend=False)
fig_aha.update_layout(xaxis_tickformat=".1%")
st.plotly_chart(fig_aha, width="stretch", key=ukey("act-aha"))

fig_aha_cart = px.bar(
    aha_cart_overall_den_cluster, y=cluster_col, x="aha_cart_overall", orientation="h",
    title="클러스터별 Aha → Cart (분모=클러스터 인원)",
    color=cluster_col, color_discrete_map=CLUSTER_COLORS,
    labels={"aha_cart_overall":"Aha→Cart (전체 대비)", cluster_col:"Cluster"}
)
fig_aha_cart.update_yaxes(categoryorder="array", categoryarray=DESIRED_ORDER)
fig_aha_cart.update_traces(texttemplate="%{x:.1%}", textposition="outside", showlegend=False)
fig_aha_cart.update_layout(xaxis_tickformat=".1%")
st.plotly_chart(fig_aha_cart, width="stretch", key=ukey("act-aha-cart"))

# =========================================================
# 2) Funnel (클러스터별 탭)
# =========================================================
st.subheader("퍼널 전환율")
tabs = st.tabs(["Cluster 0", "Cluster 1", "Cluster 2"])
tab_keys = ["0", "1", "2"]

def draw_home_based_funnel(uv: pd.DataFrame, title: str, key_sfx: str):
    h = int(len(uv))
    d = int((uv["detail_any"] == 1).sum())
    k = int(((uv["detail_any"] == 1) & (uv["cart_any"] == 1)).sum())
    pct_detail = (d / h) if h else np.nan
    pct_cart   = (k / h) if h else np.nan
    seg_text = [f"{h:,}\n100%", f"{d:,}\n{pct_detail:.1%} (Home→Detail)", f"{k:,}\n{pct_cart:.1%} (Home→Cart via Detail)"]
    fig = go.Figure(go.Funnel(
        y=["Home","Detail","Cart(Detail 경유)"], x=[h, d, k],
        text=seg_text, textposition="inside", textinfo="text",
        marker={"color": [GOOGLE_BLUE, GOOGLE_BLUE, GOOGLE_BLUE]},
        hovertemplate="%{y}: %{x:,}<extra></extra>",
    ))
    fig.update_layout(title=title)
    st.plotly_chart(fig, width="stretch", key=ukey(f"funnel-{key_sfx}"))

for t, key in zip(tabs, tab_keys):
    with t:
        uv = uf[uf[cluster_col] == key].copy()
        draw_home_based_funnel(uv, f"Funnel (Cluster {key})", key)

# =========================================================
# 3) Acquisition
# =========================================================
st.header("2. Acquisition")

cA, cB = st.columns([1,1])
with cA:
    min_share = st.slider("최소 유입 비중 제외", 0.0, 0.2, 0.01, 0.005, key=ukey("acq-minshare"))
with cB:
    top_n = st.number_input("TOP N", min_value=3, max_value=30, value=10, step=1, key=ukey("acq-topn"))

def _clean_series_exclude_noise(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.strip()
    bad = {"", "nan", "(none)", "none", "(not set)", "not set", "unavailable"}
    return ss.where(~ss.str.lower().isin(bad), other=np.nan)

def _clean_channel_keep_none(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.strip()
    bad = {"", "nan", "(not set)", "not set", "unavailable"}
    return ss.where(~ss.str.lower().isin(bad), other=np.nan)

def render_acquisition_for_dim(dfp_src: pd.DataFrame, dim_col: str, title_prefix: str):
    dfp_acq = dfp_src.copy()
    rep_dim = (
        dfp_acq[[user_col, dim_col, date_col]]
        .sort_values([user_col, date_col])
        .drop_duplicates(user_col, keep="last")[[user_col, dim_col]]
        .copy()
    )
    rep_dim[dim_col] = _clean_series_exclude_noise(rep_dim[dim_col])

    acq = (
        rep_dim.dropna(subset=[dim_col])[dim_col]
        .value_counts(normalize=True)
        .rename_axis(dim_col)
        .reset_index(name="share")
        .sort_values("share", ascending=False)
    )
    acq = acq[acq["share"] >= float(min_share)].head(int(top_n))
    order = acq[dim_col].tolist()

    fig_share = px.bar(acq, x=dim_col, y="share", title=f"{title_prefix} 상위 {len(acq)} 유입 비중 (USER)", color_discrete_sequence=[GOOGLE_BLUE])
    fig_share.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
    fig_share.update_layout(yaxis_tickformat=".1%")
    fig_share.update_xaxes(categoryorder="array", categoryarray=order)
    st.plotly_chart(fig_share, width="stretch", key=ukey(f"acq-share-{dim_col}"))

    sess_cart = dfp_acq.dropna(subset=[dim_col]).groupby([dim_col, session_col])[cart_col].any().reset_index()
    conv = sess_cart.groupby(dim_col)[cart_col].mean().reset_index(name="conversion")
    conv = conv[conv[dim_col].isin(order)]
    conv[dim_col] = pd.Categorical(conv[dim_col], categories=order, ordered=True)
    conv = conv.sort_values(dim_col)

    fig_conv = px.bar(conv, x=dim_col, y="conversion", title=f"{title_prefix}별 전환율 (세션→장바구니, %)", color_discrete_sequence=[GOOGLE_BLUE])
    fig_conv.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
    fig_conv.update_layout(yaxis_tickformat=".1%")
    fig_conv.update_xaxes(categoryorder="array", categoryarray=order)
    st.plotly_chart(fig_conv, width="stretch", key=ukey(f"acq-conv-{dim_col}"))

tab_channel, tab_device, tab_country, tab_city = st.tabs(["채널", "디바이스", "국가", "도시"])

with tab_channel:
    rep_medium = (
        dfp[[user_col, medium_col, date_col]]
        .sort_values([user_col, date_col])
        .drop_duplicates(user_col, keep="last")[[user_col, medium_col]]
        .copy()
    )
    rep_medium[medium_col] = _clean_channel_keep_none(rep_medium[medium_col])

    channel_share = (
        rep_medium.dropna(subset=[medium_col])[medium_col]
        .value_counts(normalize=True)
        .rename_axis(medium_col)
        .reset_index(name="share")
        .sort_values("share", ascending=False)
    )
    channel_share = channel_share[channel_share["share"] >= float(min_share)].head(int(top_n))
    order_channels = channel_share[medium_col].tolist()

    fig_ch_share = px.bar(channel_share, x=medium_col, y="share", title=f"채널별 유입 비중 (USER) — Top {len(channel_share)}", color_discrete_sequence=[GOOGLE_BLUE])
    fig_ch_share.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
    fig_ch_share.update_layout(yaxis_tickformat=".1%")
    fig_ch_share.update_xaxes(categoryorder="array", categoryarray=order_channels)
    st.plotly_chart(fig_ch_share, width="stretch", key=ukey("acq-channel-share"))

    df_conv = dfp.copy()
    df_conv[medium_col] = _clean_channel_keep_none(df_conv[medium_col])
    df_conv = df_conv.dropna(subset=[medium_col])

    sess_cart = df_conv.groupby([medium_col, session_col])[cart_col].any().reset_index()
    conv = sess_cart.groupby(medium_col)[cart_col].mean().reset_index(name="conversion")
    conv = conv[conv[medium_col].isin(order_channels)]
    conv[medium_col] = pd.Categorical(conv[medium_col], categories=order_channels, ordered=True)
    conv = conv.sort_values(medium_col)

    fig_ch_conv = px.bar(conv, x=medium_col, y="conversion", title="채널별 전환율 (세션→장바구니, %)", color_discrete_sequence=[GOOGLE_BLUE])
    fig_ch_conv.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
    fig_ch_conv.update_layout(yaxis_tickformat=".1%")
    fig_ch_conv.update_xaxes(categoryorder="array", categoryarray=order_channels)
    st.plotly_chart(fig_ch_conv, width="stretch", key=ukey("acq-channel-conv"))

with tab_device:
    render_acquisition_for_dim(dfp, device_col, "디바이스")

with tab_country:
    render_acquisition_for_dim(dfp, country_col, "국가")

with tab_city:
    def _clean_city(s: pd.Series) -> pd.Series:
        ss = s.astype(str).str.strip()
        bad = {"", "nan", "(none)", "none", "(not set)", "not set", "unavailable", "not available in demo dataset"}
        return ss.where(~ss.str.lower().isin(bad), other=np.nan)

    possible_city_cols = ["city", "City", "regionCity"]
    city_col = next((c for c in possible_city_cols if c in dfp.columns), None)

    if city_col is not None:
        rep_city = (
            dfp[[user_col, city_col, date_col]]
              .sort_values([user_col, date_col])
              .drop_duplicates(user_col, keep="last")[[user_col, city_col]]
              .copy()
        )
        rep_city[city_col] = _clean_city(rep_city[city_col])

        top_n_cities = int(top_n)
        city_share = (
            rep_city.dropna(subset=[city_col])[city_col]
                    .value_counts(normalize=True)
                    .rename_axis(city_col).reset_index(name="share")
                    .sort_values("share", ascending=False)
                    .head(top_n_cities)
        )

        if not city_share.empty:
            order_cities = city_share[city_col].tolist()

            fig_city_share = px.bar(city_share, x=city_col, y="share", title="도시별 유입 비중 Top N (USER)", color_discrete_sequence=[GOOGLE_BLUE])
            fig_city_share.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
            fig_city_share.update_layout(yaxis_tickformat=".1%")
            fig_city_share.update_xaxes(categoryorder="array", categoryarray=order_cities)
            st.plotly_chart(fig_city_share, width="stretch", key=ukey("city-share"))

            df_city_conv = dfp.copy()
            df_city_conv[city_col] = _clean_city(df_city_conv[city_col])
            df_city_conv = df_city_conv.dropna(subset=[city_col])

            sess_city = df_city_conv.groupby([city_col, session_col])[cart_col].any().reset_index()
            conv_city = sess_city.groupby(city_col)[cart_col].mean().reset_index(name="conversion")
            conv_city = conv_city[conv_city[city_col].isin(order_cities)]
            conv_city[city_col] = pd.Categorical(conv_city[city_col], categories=order_cities, ordered=True)
            conv_city = conv_city.sort_values(city_col)

            fig_city_conv = px.bar(conv_city, x=city_col, y="conversion", title="도시별 전환율 (세션→장바구니, %)", color_discrete_sequence=[GOOGLE_BLUE])
            fig_city_conv.update_traces(texttemplate="%{y:.1%}", textposition="outside", showlegend=False)
            fig_city_conv.update_layout(yaxis_tickformat=".1%")
            fig_city_conv.update_xaxes(categoryorder="array", categoryarray=order_cities)
            st.plotly_chart(fig_city_conv, width="stretch", key=ukey("city-conv"))
        else:
            st.info("표시할 도시 데이터가 없습니다.")
    else:
        st.warning("도시 컬럼(city/City/regionCity)을 찾을 수 없습니다. (원본 컬럼명 확인)")

# ====================== Stickiness (DAU/WAU ↔ WAU/MAU 토글) ======================
st.header("고착도 (Stickiness)")

# ── 소스 준비: 날짜/유저만 추출 + 정규화
d_stick = dfp[[date_col, user_col]].dropna().copy()
d_stick[date_col] = pd.to_datetime(d_stick[date_col], errors="coerce")
d_stick = d_stick[d_stick[date_col].notna()].copy()
d_stick["date"]  = d_stick[date_col].dt.normalize()
d_stick["week"]  = d_stick[date_col].dt.to_period("W").dt.start_time
d_stick["month"] = d_stick[date_col].dt.to_period("M").dt.start_time

# ── 라디오 필터
mode = st.radio(
    "보기 전환",
    options=["DAU/WAU (일별)", "WAU/MAU (주별)"],
    horizontal=True,
    index=0,  # 기본: 일별
    key=ukey("stickiness-mode")
)

if d_stick.empty:
    st.info("고착도 계산을 위한 데이터가 없습니다.")
else:
    # ── 공통: 일/주/월 집계 준비(한 번에 계산해두고 아래에서 선택 출력)
    # 1) 일별 DAU
    daily = (
        d_stick.groupby("date", as_index=False)[user_col]
               .nunique()
               .rename(columns={user_col: "DAU"})
               .sort_values("date")
    )
    # 날짜 연속성 보장 (빈 날짜 0 채움)
    if not daily.empty:
        full_days = pd.DataFrame({"date": pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")})
        daily = full_days.merge(daily, on="date", how="left").fillna({"DAU": 0})
        daily["DAU"] = daily["DAU"].astype(int)

    # 2) 7일 윈도우 WAU (유저-날짜 원천에서 정확히 distinct 집계)
    if not daily.empty:
        N = 7
        d_idx = d_stick[["date", user_col]].drop_duplicates().set_index("date").sort_index()
        wau_list = []
        for d in daily["date"]:
            left = d - pd.Timedelta(days=N-1)
            # 인덱스 경계 안전 처리
            if d_idx.empty:
                sub = d_idx
            else:
                sub = d_idx.loc[max(left, d_idx.index.min()):min(d, d_idx.index.max())]
            wau_list.append(sub[user_col].nunique())
        daily["WAU"] = wau_list
        daily["DAU/WAU"] = (daily["DAU"] / daily["WAU"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 3) 주별 WAU
    weekly = (
        d_stick.groupby("week", as_index=False)[user_col]
               .nunique()
               .rename(columns={user_col: "WAU"})
               .sort_values("week")
    )
    # 4) 월별 MAU
    monthly = (
        d_stick.groupby("month", as_index=False)[user_col]
               .nunique()
               .rename(columns={user_col: "MAU"})
               .sort_values("month")
    )
    # 주→월 매핑하여 비율
    wk = None
    if not weekly.empty and not monthly.empty:
        weekly["week"]  = pd.to_datetime(weekly["week"], errors="coerce")
        weekly["month"] = weekly["week"].dt.to_period("M").dt.start_time
        wk = weekly.merge(monthly, on="month", how="left")
        wk["WAU/MAU"] = (wk["WAU"] / wk["MAU"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ── 선택 출력
    if mode.startswith("DAU/WAU"):
        if daily.empty:
            st.info("일별 고착도(DAU/WAU)를 계산할 데이터가 부족합니다.")
        else:
            fig_dau_wau = px.line(
                daily, x="date", y="DAU/WAU",
                title="고착도 (DAU/WAU, 일별)",
                markers=True
            )
            fig_dau_wau.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig_dau_wau, width="stretch", key=ukey("stick-dau-wau-toggle"))
    else:
        if wk is None or wk.empty:
            st.info("주별 고착도(WAU/MAU)를 계산할 데이터가 부족합니다.")
        else:
            fig_wau_mau = px.line(
                wk.sort_values("week"), x="week", y="WAU/MAU",
                title="고착도 (WAU/MAU, 주별)",
                markers=True
            )
            fig_wau_mau.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig_wau_mau, width="stretch", key=ukey("stick-wau-mau-toggle"))
# ====================== /Stickiness ======================


# =========================================================
# 4) Retention (30/90일) — Heatmap
# =========================================================
st.header("3. Retention")

cohort = (
    dfp[[user_col, cluster_col, date_col]]
      .drop_duplicates()
      .sort_values([user_col, date_col])
)

def _rep_mode(s: pd.Series):
    m = pd.to_numeric(s, errors="coerce"); m = m[m.notna()]
    return str(int(m.mode().iat[0])) if not m.mode().empty else None

rep_cluster_mode = cohort.groupby(user_col)[cluster_col].apply(_rep_mode).reset_index(name=cluster_col)
first_visit = cohort.groupby(user_col, as_index=False)[date_col].min().rename(columns={date_col: "first_visit"})
cohort2 = cohort.merge(first_visit, on=user_col, how="left")

def _ret_flag(days: int) -> pd.Series:
    limit = cohort2["first_visit"] + pd.to_timedelta(days, "D")
    flag = (cohort2[date_col] > cohort2["first_visit"]) & (cohort2[date_col] <= limit)
    return cohort2.assign(flag=flag).groupby(user_col)["flag"].any().rename(f"ret_{days}")

ret30 = _ret_flag(30)
ret90 = _ret_flag(90)

ur = (
    first_visit.merge(ret30.reset_index(), on=user_col, how="left")
               .merge(ret90.reset_index(), on=user_col, how="left")
               .merge(rep_cluster_mode, on=user_col, how="left")
)
ur["cohort_month"] = ur["first_visit"].dt.to_period("M").dt.start_time

def render_retention_heatmap(ret_col: str, window_label: str, key_suffix: str):
    pivot = (
        ur.groupby(["cohort_month", cluster_col])[ret_col]
          .mean().reset_index()
          .pivot(index="cohort_month", columns=cluster_col, values=ret_col)
    )
    if pivot.empty:
        st.info(f"{window_label} 유지율 집계가 없습니다.")
        return
    pivot.columns = pivot.columns.astype(str)
    col_order = [c for c in DESIRED_ORDER if c in pivot.columns]
    if col_order:
        pivot = pivot[col_order]
    z = pivot.values
    x = list(pivot.columns)
    y = [pd.to_datetime(d).strftime("%Y-%m") for d in pivot.index]
    text = np.where(np.isnan(z), "", (z*100).round(1).astype(str) + "%")

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y, colorscale="Blues",
        hovertemplate="Cohort %{y}<br>Cluster %{x}<br>Retention %{z:.1%}<extra></extra>"
    ))
    fig.update_layout(title=f"클러스터별 코호트 유지율 Heatmap ({window_label}) — (열=Cluster, 행=Cohort Month)")
    fig.add_trace(go.Scatter(
        x=np.repeat(x, len(y)), y=np.tile(y, len(x)),
        mode="text", text=text.flatten(), hoverinfo="skip", showlegend=False
    ))
    st.plotly_chart(fig, width="stretch", key=ukey(f"ret-heatmap-{key_suffix}"))

tab30, tab90 = st.tabs(["30일", "90일"])
with tab30:
    render_retention_heatmap("ret_30", "30일", "30")
with tab90:
    render_retention_heatmap("ret_90", "90일", "90")

# =========================================================
# 5) 장바구니 진입 비중
# =========================================================
cart_users_cluster = (
    uf[uf["cart_any"]]
      .groupby(cluster_col)[user_col]
      .nunique()
      .rename("cart_users")
      .reset_index()
)

total_cart_users = int(cart_users_cluster["cart_users"].sum())
cart_users_cluster["cart_share"] = (cart_users_cluster["cart_users"] / total_cart_users if total_cart_users > 0 else np.nan)

label_map = {"1": "비활성 고객", "0": "탐색 고객","2": "충성 고객" }
color_map = {"1": GOOGLE_YELLOW, "0": GOOGLE_BLUE, "2": GOOGLE_RED}
order_clusters = ["2", "0", "1"]  # y축: 충성→탐색→비활성
order_labels   = [label_map[c] for c in order_clusters]

cart_users_cluster = (
    cart_users_cluster.set_index(cluster_col)
                      .reindex(order_clusters)
                      .fillna(0)
                      .reset_index()
)
cart_users_cluster["세그먼트"] = cart_users_cluster[cluster_col].map(label_map)

fig = px.bar(
    cart_users_cluster, y="세그먼트", x="cart_share", orientation="h",
    color=cluster_col, color_discrete_map=color_map,
    labels={"세그먼트": "세그먼트", "cart_share": "장바구니 진입 비중"},
    title=f"세그먼트별 장바구니 진입 비중 (분모=전체 장바구니 진입자 {total_cart_users:,}명)"
)
fig.update_traces(texttemplate="%{x:.1%}", textposition="outside", showlegend=False)
fig.update_layout(xaxis_tickformat=".1%")
fig.update_yaxes(categoryorder="array", categoryarray=order_labels)
st.plotly_chart(fig, width="stretch", key=ukey("cart-share-hbar"))

# dataframe: width="stretch" 대신 use_container_width=True 유지
st.dataframe(
    cart_users_cluster[["세그먼트", "cart_users", "cart_share"]]
      .rename(columns={"cart_users": "장바구니 진입자 수", "cart_share": "장바구니 진입 비중"}),
    use_container_width=True
)




