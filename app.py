"""
加盟店送客エンジン（インバウンド × DOOH デモ）
百貨店訪問者の前夜ホテル × 来店経路 × DOOH 最適タッチポイント分析
"""
import math
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 定数: Liveboard DOOH 設置場所（東京都心部 モックデータ）
# ─────────────────────────────────────────────────────────────────────────────
DOOH_DF = pd.DataFrame([
    {"id": "D01", "name": "銀座四丁目交差点",       "lat": 35.6714, "lon": 139.7651},
    {"id": "D02", "name": "有楽町マリオン前",        "lat": 35.6753, "lon": 139.7620},
    {"id": "D03", "name": "銀座一丁目駅前",          "lat": 35.6731, "lon": 139.7666},
    {"id": "D04", "name": "東銀座駅前",              "lat": 35.6680, "lon": 139.7659},
    {"id": "D05", "name": "銀座三丁目交差点",        "lat": 35.6706, "lon": 139.7651},
    {"id": "D06", "name": "日比谷公園前",            "lat": 35.6737, "lon": 139.7585},
    {"id": "D07", "name": "新橋駅 SL広場",           "lat": 35.6659, "lon": 139.7575},
    {"id": "D08", "name": "銀座 ITOYA 前",           "lat": 35.6700, "lon": 139.7645},
    {"id": "D09", "name": "築地本願寺前",            "lat": 35.6660, "lon": 139.7708},
    {"id": "D10", "name": "浜離宮前",                "lat": 35.6597, "lon": 139.7631},
    {"id": "D11", "name": "虎ノ門ヒルズ前",          "lat": 35.6673, "lon": 139.7490},
    {"id": "D12", "name": "東京駅丸の内南口",        "lat": 35.6790, "lon": 139.7660},
    {"id": "D13", "name": "京橋交差点",              "lat": 35.6757, "lon": 139.7683},
    {"id": "D14", "name": "銀座柳通り",              "lat": 35.6715, "lon": 139.7636},
    {"id": "D15", "name": "晴海通り 銀座5丁目",      "lat": 35.6695, "lon": 139.7660},
    {"id": "D16", "name": "日比谷交差点",            "lat": 35.6736, "lon": 139.7586},
    {"id": "D17", "name": "銀座中央通り 8丁目",      "lat": 35.6665, "lon": 139.7648},
    {"id": "D18", "name": "汐留メディアタワー前",    "lat": 35.6633, "lon": 139.7589},
    {"id": "D19", "name": "内幸町交差点",            "lat": 35.6714, "lon": 139.7530},
    {"id": "D20", "name": "有楽町交差点",            "lat": 35.6746, "lon": 139.7633},
])

REQUIRED_COLS = {"member_id", "lat", "lon", "stay_datetime", "stay_duration_min"}

# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """2 点間のコンパス方位角 (0=北, 90=東 …)"""
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def load_gps_csv(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"CSV 読み込みエラー: {e}"); return None
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(f"必要なカラムがありません: {missing}"); return None
    df["stay_datetime"] = pd.to_datetime(df["stay_datetime"], errors="coerce")
    df = df.dropna(subset=["stay_datetime", "lat", "lon"])
    df["stay_duration_min"] = pd.to_numeric(df["stay_duration_min"], errors="coerce").fillna(0)
    df["hour"] = df["stay_datetime"].dt.hour
    df["date"]  = df["stay_datetime"].dt.date
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Overpass API ホテル取得
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hotels_overpass(center_lat: float, center_lon: float,
                           radius_m: float = 2000) -> pd.DataFrame:
    """Overpass API でホテル・宿泊施設 POI を取得"""
    dlat = radius_m / 111320
    dlon = radius_m / (111320 * math.cos(math.radians(center_lat)))
    s, n = center_lat - dlat, center_lat + dlat
    w, e = center_lon - dlon, center_lon + dlon

    query = f"""[out:json][timeout:35];
(
  node["tourism"~"hotel|hostel|guest_house|motel"]({s},{w},{n},{e});
  way["tourism"~"hotel|hostel|guest_house|motel"]({s},{w},{n},{e});
);
out center;"""
    url  = "https://overpass-api.de/api/interpreter"
    data = urllib.parse.urlencode({"data": query}).encode()
    req  = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=40) as r:
        result = json.loads(r.read())

    rows = []
    for elem in result.get("elements", []):
        lat = elem.get("lat") or elem.get("center", {}).get("lat")
        lon = elem.get("lon") or elem.get("center", {}).get("lon")
        if not lat: continue
        tags = elem.get("tags", {})
        name = tags.get("name") or tags.get("name:en") or "不明なホテル"
        rows.append({
            "hotel_id":   f"H{elem['id']}",
            "hotel_name": name,
            "lat": lat, "lon": lon,
        })
    if not rows:
        return pd.DataFrame(columns=["hotel_id", "hotel_name", "lat", "lon"])
    df = pd.DataFrame(rows).drop_duplicates("hotel_id").reset_index(drop=True)
    df["dist_m"] = df.apply(lambda r: haversine_m(r.lat, r.lon, center_lat, center_lon), axis=1)
    return df.sort_values("dist_m").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 百貨店訪問者特定
# ─────────────────────────────────────────────────────────────────────────────
def find_store_visitors(gps_df: pd.DataFrame,
                         store_lat: float, store_lon: float,
                         store_radius_m: float,
                         min_stay_min: float = 5.0):
    """
    Returns:
        visitor_ids : set of member_id
        store_visits: DataFrame [member_id, visit_date, arrival_time, stay_duration_min]
    """
    df = gps_df.copy()
    df["dist_store"] = df.apply(
        lambda r: haversine_m(r.lat, r.lon, store_lat, store_lon), axis=1
    )
    in_store = df[(df["dist_store"] <= store_radius_m) &
                  (df["stay_duration_min"] >= min_stay_min)].copy()
    if in_store.empty:
        return set(), pd.DataFrame()

    visits = (
        in_store.sort_values("stay_datetime")
        .groupby(["member_id", "date"])
        .agg(arrival_time=("stay_datetime", "min"),
             stay_duration_min=("stay_duration_min", "max"))
        .reset_index()
        .rename(columns={"date": "visit_date"})
    )
    return set(visits["member_id"]), visits


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 前夜ホテル宿泊特定
# ─────────────────────────────────────────────────────────────────────────────
def find_prev_night_hotel(gps_df: pd.DataFrame,
                           store_visits: pd.DataFrame,
                           hotels_df: pd.DataFrame,
                           hotel_match_r_m: float = 150.0,
                           min_hotel_stay_min: float = 90.0) -> pd.DataFrame:
    """
    百貨店訪問日の前夜（前日18時〜当日12時）にホテルで宿泊したことを GPS から特定。
    Returns: [member_id, hotel_id, hotel_name, hotel_lat, hotel_lon,
               night_date, checkout_time, stay_duration_min, visit_date, arrival_time]
    """
    if hotels_df.empty or store_visits.empty:
        return pd.DataFrame()

    # 夜間・長時間滞在レコード
    night_gps = gps_df[
        (gps_df["stay_duration_min"] >= min_hotel_stay_min) &
        (gps_df["hour"].isin(list(range(18, 24)) + list(range(0, 13))))
    ].copy()
    if night_gps.empty:
        return pd.DataFrame()

    # 最近傍ホテルへのマッチング
    matched = []
    for _, rec in night_gps.iterrows():
        best_d, best_h = float("inf"), None
        for _, h in hotels_df.iterrows():
            d = haversine_m(rec.lat, rec.lon, h.lat, h.lon)
            if d < best_d and d <= hotel_match_r_m:
                best_d, best_h = d, h
        if best_h is None:
            continue
        # 夜付け日の正規化
        if rec.hour >= 18:
            night_date = rec.date
        else:
            night_date = (datetime.combine(rec.date, datetime.min.time()) - timedelta(days=1)).date()
        matched.append({
            "member_id":    rec.member_id,
            "hotel_id":     best_h.hotel_id,
            "hotel_name":   best_h.hotel_name,
            "hotel_lat":    best_h.lat,
            "hotel_lon":    best_h.lon,
            "night_date":   night_date,
            "checkout_time": rec.stay_datetime + timedelta(minutes=rec.stay_duration_min),
            "stay_duration_min": rec.stay_duration_min,
        })

    if not matched:
        return pd.DataFrame()

    hotel_stays = pd.DataFrame(matched)

    # 前夜 × 翌日来店 を結合
    rows = []
    for _, visit in store_visits.iterrows():
        prev_night = (datetime.combine(visit.visit_date, datetime.min.time())
                      - timedelta(days=1)).date()
        prev = hotel_stays[
            (hotel_stays["member_id"] == visit.member_id) &
            (hotel_stays["night_date"] == prev_night)
        ]
        if prev.empty:
            continue
        row = prev.iloc[0].to_dict()
        row["visit_date"]    = visit.visit_date
        row["arrival_time"]  = visit.arrival_time
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .drop_duplicates(subset=["member_id", "visit_date"])
            .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# Step 4a: DOOH 通過判定
# ─────────────────────────────────────────────────────────────────────────────
def compute_dooh_passages(gps_df: pd.DataFrame,
                           prev_night_df: pd.DataFrame,
                           dooh_df: pd.DataFrame,
                           selected_hotel_ids: Optional[list],
                           passage_r_m: float = 500.0) -> pd.DataFrame:
    """
    チェックアウト後〜百貨店到着前の GPS がDOOH 周辺 passage_r_m 以内にあるか判定。
    Returns: [member_id, hotel_name, dooh_id, dooh_name, dooh_lat, dooh_lon,
               passage_time, visit_date]
    """
    target = (prev_night_df[prev_night_df["hotel_id"].isin(selected_hotel_ids)]
              if selected_hotel_ids else prev_night_df).copy()
    if target.empty:
        return pd.DataFrame()

    passages = []
    for _, row in target.iterrows():
        day_gps = gps_df[
            (gps_df["member_id"] == row.member_id) &
            (gps_df["date"] == row.visit_date) &
            (gps_df["stay_datetime"] >= row.checkout_time) &
            (gps_df["stay_datetime"] <= row.arrival_time)
        ].sort_values("stay_datetime")

        for _, gps_row in day_gps.iterrows():
            for _, d in dooh_df.iterrows():
                if haversine_m(gps_row.lat, gps_row.lon, d.lat, d.lon) <= passage_r_m:
                    passages.append({
                        "member_id":   row.member_id,
                        "hotel_name":  row.hotel_name,
                        "hotel_id":    row.hotel_id,
                        "dooh_id":     d.id,
                        "dooh_name":   d["name"],
                        "dooh_lat":    d.lat,
                        "dooh_lon":    d.lon,
                        "passage_time": gps_row.stay_datetime,
                        "visit_date":  row.visit_date,
                    })

    if not passages:
        return pd.DataFrame()
    df = pd.DataFrame(passages)
    return (df.sort_values("passage_time")
              .drop_duplicates(subset=["member_id", "dooh_id", "visit_date"])
              .reset_index(drop=True))


def build_sankey(passages_df: pd.DataFrame,
                 prev_night_df: pd.DataFrame,
                 store_name: str,
                 n_total_visitors: int,
                 threshold_pct: float) -> Optional[go.Figure]:
    """ホテル → DOOH (時系列順) → 百貨店 のサンキーダイアグラム"""
    if passages_df.empty:
        return None

    # threshold 超の DOOH だけ
    dooh_cnt = passages_df.groupby("dooh_id")["member_id"].nunique()
    qualifying = dooh_cnt[dooh_cnt / n_total_visitors * 100 >= threshold_pct].index.tolist()
    if not qualifying:
        return None

    pf = passages_df[passages_df["dooh_id"].isin(qualifying)]

    # 各メンバーのシーケンス: hotel → DOOH(時系列) → store
    sequences = []
    target_members = set(prev_night_df["member_id"])
    for mid in target_members:
        hotel_row = prev_night_df[prev_night_df["member_id"] == mid]
        hotel = hotel_row.iloc[0]["hotel_name"] if not hotel_row.empty else "不明ホテル"
        member_dooh = (pf[pf["member_id"] == mid]
                       .sort_values("passage_time")["dooh_name"].tolist())
        sequences.append([hotel] + member_dooh + [store_name])

    # ノード辞書 & リンク集計
    node_order, node_idx = [], {}
    for seq in sequences:
        for n in seq:
            if n not in node_idx:
                node_idx[n] = len(node_order)
                node_order.append(n)

    link_cnt: dict = {}
    for seq in sequences:
        for i in range(len(seq) - 1):
            key = (node_idx[seq[i]], node_idx[seq[i + 1]])
            link_cnt[key] = link_cnt.get(key, 0) + 1

    hotel_names = set(prev_night_df["hotel_name"])
    dooh_names  = set(pf["dooh_name"])
    colors = [
        "#2980b9" if n in hotel_names else
        "#e74c3c" if n in dooh_names  else
        "#27ae60"
        for n in node_order
    ]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=15, thickness=20, label=node_order, color=colors),
        link=dict(
            source=[k[0] for k in link_cnt],
            target=[k[1] for k in link_cnt],
            value=list(link_cnt.values()),
            color="rgba(180,180,180,0.35)",
        ),
    ))
    fig.update_layout(
        title=f"ホテル → DOOH 通過 → {store_name}  タッチポイントフロー",
        height=520, font_size=11,
        margin=dict(t=50, b=10, l=20, r=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Step 4b: 来店直前経路分析
# ─────────────────────────────────────────────────────────────────────────────
def analyze_pre_arrival_routes(gps_df: pd.DataFrame,
                                store_visits: pd.DataFrame,
                                store_lat: float, store_lon: float,
                                pre_minutes: int = 30,
                                threshold_pct: float = 10.0) -> pd.DataFrame:
    """
    百貨店到着前 pre_minutes 分以内の GPS 点を連結してセグメント化。
    通行人数が threshold_pct % 以上のセグメントを返す。
    Returns: [lat1,lon1,lat2,lon2,count,pct,approaching,bearing]
    """
    if store_visits.empty:
        return pd.DataFrame()

    n_visitors = store_visits["member_id"].nunique()
    segs = []

    for _, visit in store_visits.iterrows():
        t0 = visit.arrival_time - timedelta(minutes=pre_minutes)
        pts = (gps_df[
            (gps_df["member_id"] == visit.member_id) &
            (gps_df["stay_datetime"] >= t0) &
            (gps_df["stay_datetime"] <= visit.arrival_time)
        ].sort_values("stay_datetime")[["lat", "lon"]].values)

        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            la1, lo1 = round(pts[i][0], 4),   round(pts[i][1], 4)
            la2, lo2 = round(pts[i+1][0], 4), round(pts[i+1][1], 4)
            if la1 == la2 and lo1 == lo2:
                continue
            brg = bearing_deg(la1, lo1, la2, lo2)
            mid_lat, mid_lon = (la1+la2)/2, (lo1+lo2)/2
            brg_store = bearing_deg(mid_lat, mid_lon, store_lat, store_lon)
            diff = abs(brg - brg_store)
            if diff > 180: diff = 360 - diff
            segs.append({
                "member_id":  visit.member_id,
                "lat1": la1, "lon1": lo1,
                "lat2": la2, "lon2": lo2,
                "bearing":    brg,
                "approaching": diff < 90,
            })

    if not segs:
        return pd.DataFrame()

    df = pd.DataFrame(segs)
    grp = (df.groupby(["lat1","lon1","lat2","lon2","approaching"])["member_id"]
             .nunique().reset_index()
             .rename(columns={"member_id": "count"}))

    threshold = max(1, n_visitors * threshold_pct / 100.0)
    grp = grp[grp["count"] >= threshold].copy()
    if grp.empty:
        return pd.DataFrame()

    grp["bearing"] = grp.apply(
        lambda r: bearing_deg(r.lat1, r.lon1, r.lat2, r.lon2), axis=1
    )
    grp["pct"] = grp["count"] / n_visitors * 100
    return grp.sort_values("count", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="加盟店送客エンジン",
    page_icon="🏬",
    layout="wide",
)

st.title("🏬 加盟店送客エンジン（インバウンド × DOOH デモ）")
st.caption("百貨店訪問者の前夜ホテル × 来店経路 × DOOH 最適タッチポイント分析")

# ── セッション初期化 ──────────────────────────────────────────────────────────
for k, v in [
    ("gps_df", None), ("hotels_df", None),
    ("store_visitors", None), ("store_visits_df", None),
    ("prev_night_df", None), ("analysis_mode", None),
    ("dooh_passages_df", None), ("dooh_n_sel", 0),
    ("route_segs_df", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: 百貨店設定 & GPS アップロード
# ═════════════════════════════════════════════════════════════════════════════
st.header("① 百貨店設定 & GPS データ入力")

col_cfg, col_up = st.columns(2)
with col_cfg:
    st.subheader("百貨店中心点")
    store_name   = st.text_input("百貨店名",  value="松屋銀座")
    store_lat    = st.number_input("緯度",     value=35.6722, format="%.6f", step=0.0001)
    store_lon    = st.number_input("経度",     value=139.7658, format="%.6f", step=0.0001)
    store_radius = st.slider("訪問判定半径 (m)", 50, 500, 150, step=25)
    min_visit    = st.slider("最低滞在時間 (分)", 1, 60, 5, step=1)

with col_up:
    st.subheader("GPS データアップロード")
    with st.expander("必要カラム仕様", expanded=False):
        st.markdown("""
| カラム名 | 型 | 説明 |
|---|---|---|
| `member_id` | String | メンバー ID |
| `lat` | Float | 緯度 (WGS84) |
| `lon` | Float | 経度 (WGS84) |
| `stay_datetime` | DateTime | 滞在日時 |
| `stay_duration_min` | Float | 滞在時間（分） |

サンプルデータ: `generate_sample_gps.py` で生成してください。
""")
    uploaded = st.file_uploader("GPS データ CSV", type=["csv"])
    if uploaded:
        df_tmp = load_gps_csv(uploaded)
        if df_tmp is not None:
            st.session_state.update({
                "gps_df": df_tmp, "hotels_df": None,
                "prev_night_df": None, "store_visitors": None,
                "dooh_passages_df": None, "route_segs_df": None,
            })
            st.success(f"✅ {len(df_tmp):,} レコード / {df_tmp['member_id'].nunique():,} メンバー")

if st.session_state["gps_df"] is None:
    st.info("GPS データ CSV をアップロードしてください。")
    st.stop()

gps_df = st.session_state["gps_df"]

# 百貨店訪問者特定
store_visitors, store_visits_df = find_store_visitors(
    gps_df, store_lat, store_lon, store_radius, min_visit
)
st.session_state["store_visitors"]  = store_visitors
st.session_state["store_visits_df"] = store_visits_df

m1, m2, m3 = st.columns(3)
m1.metric("GPS レコード総数",       f"{len(gps_df):,}")
m2.metric("ユニークメンバー数",     f"{gps_df['member_id'].nunique():,}")
m3.metric(f"{store_name} 訪問者数", f"{len(store_visitors):,}")

if not store_visitors:
    st.warning("百貨店訪問者が検出されませんでした。中心点・半径を調整してください。")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: 前夜ホテル分析
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("② 前夜宿泊ホテル分析")

col_h1, col_h2, col_h3 = st.columns(3)
with col_h1:
    hotel_search_r = st.slider("ホテル検索範囲 (m)", 500, 5000, 2000, step=100)
with col_h2:
    hotel_match_r  = st.slider("ホテル GPS マッチ半径 (m)", 50, 300, 150, step=25)
with col_h3:
    min_hotel_stay = st.slider("最低ホテル滞在時間 (分)", 60, 360, 90, step=30)

fetch_btn = st.button("🏨 ホテルデータ取得 (Overpass API)", type="primary")
if fetch_btn:
    st.session_state.update({"hotels_df": None, "prev_night_df": None})
    with st.spinner("Overpass API からホテルを取得中..."):
        try:
            hotels_df = fetch_hotels_overpass(store_lat, store_lon, hotel_search_r)
            st.session_state["hotels_df"] = hotels_df
            if hotels_df.empty:
                st.warning("ホテルが見つかりませんでした。検索範囲を広げてください。")
            else:
                st.success(f"✅ {len(hotels_df)} 件のホテルを取得")
        except Exception as e:
            st.error(f"Overpass API エラー: {e}")

if st.session_state["hotels_df"] is None:
    st.info("「ホテルデータ取得」ボタンを押してください。")
    st.stop()

hotels_df: pd.DataFrame = st.session_state["hotels_df"]
if hotels_df.empty:
    st.warning("ホテルデータが空です。")
    st.stop()

# 前夜ホテル宿泊の特定
if st.session_state["prev_night_df"] is None:
    with st.spinner("前夜ホテル宿泊を解析中..."):
        prev_night_df = find_prev_night_hotel(
            gps_df, store_visits_df, hotels_df,
            hotel_match_r_m=hotel_match_r,
            min_hotel_stay_min=min_hotel_stay,
        )
    st.session_state["prev_night_df"] = prev_night_df

prev_night_df: pd.DataFrame = st.session_state["prev_night_df"]

if prev_night_df.empty:
    st.warning("前夜ホテル宿泊が検出されませんでした。"
               "GPS データに夜間の長時間滞在レコードがあるか確認してください。")
    st.stop()

# ── ホテル統計 ────────────────────────────────────────────────────────────────
hotel_stats = (
    prev_night_df
    .groupby(["hotel_id", "hotel_name", "hotel_lat", "hotel_lon"])["member_id"]
    .nunique().reset_index()
    .rename(columns={"member_id": "guests"})
    .sort_values("guests", ascending=False)
    .reset_index(drop=True)
)
hotel_stats["pct"] = hotel_stats["guests"] / len(store_visitors) * 100

st.metric(
    "前夜ホテル宿泊が確認できた訪問者",
    f"{prev_night_df['member_id'].nunique():,} 人 / {len(store_visitors):,} 人",
)

# 地図
st.subheader("🗺️ 前夜宿泊ホテル分布")
angles = np.linspace(0, 2 * math.pi, 60)
lat_sc = 111320.0
lon_sc = 111320.0 * math.cos(math.radians(store_lat))

fig_h = go.Figure()
# 百貨店マーカー
fig_h.add_trace(go.Scattermapbox(
    lat=[store_lat], lon=[store_lon], mode="markers+text",
    marker=dict(size=20, color="#e74c3c", symbol="star"),
    text=[store_name], textposition="top right", name=store_name,
))
# ホテル（サイズ = 来訪者数）
fig_h.add_trace(go.Scattermapbox(
    lat=hotel_stats["hotel_lat"], lon=hotel_stats["hotel_lon"],
    mode="markers",
    marker=dict(
        size=[max(8, min(40, int(v * 2.5))) for v in hotel_stats["guests"]],
        color="#2980b9", opacity=0.85,
    ),
    text=[f"<b>{r.hotel_name}</b><br>{r.guests} 人 ({r.pct:.1f}%)"
          for _, r in hotel_stats.iterrows()],
    hovertemplate="%{text}<extra></extra>",
    name="前夜宿泊ホテル",
))
fig_h.update_layout(
    mapbox=dict(style="open-street-map",
                center=dict(lat=store_lat, lon=store_lon), zoom=14),
    height=460, margin=dict(r=0,t=0,l=0,b=0),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.88)"),
)
st.plotly_chart(fig_h, use_container_width=True)

# TOP10 棒グラフ
st.subheader("🏆 前夜宿泊ホテル TOP10（来訪者数）")
top10 = hotel_stats.head(10)
fig_bar = px.bar(
    top10, x="guests", y="hotel_name", orientation="h",
    labels={"guests": "来訪者数 (人)", "hotel_name": ""},
    text=top10.apply(lambda r: f"{r.guests} ({r.pct:.1f}%)", axis=1),
    color="guests", color_continuous_scale="Blues",
)
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(
    height=max(250, len(top10) * 35),
    yaxis=dict(autorange="reversed"),
    margin=dict(t=10, b=10, l=10, r=80),
    coloraxis_showscale=False,
)
st.plotly_chart(fig_bar, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: 分析モード選択
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("③ 分析モード選択")

mode = st.radio(
    "分析モードを選択してください",
    ["📺 DOOH 訴求推薦", "🗺️ 来店直前の経路特定"],
    horizontal=True,
)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4a: DOOH 訴求推薦
# ═════════════════════════════════════════════════════════════════════════════
if mode == "📺 DOOH 訴求推薦":
    st.divider()
    st.header("④ DOOH 訴求推薦")

    st.subheader("分析対象ホテルを選択")
    all_hotel_names = hotel_stats["hotel_name"].tolist()
    default_sel = all_hotel_names[:min(5, len(all_hotel_names))]
    sel_hotels = st.multiselect(
        "対象ホテル（空白 = 全て）",
        options=all_hotel_names,
        default=default_sel,
    )

    col_da, col_db = st.columns(2)
    with col_da:
        dooh_r    = st.slider("DOOH 検知半径 (m)", 100, 1000, 500, step=50)
    with col_db:
        dooh_thr  = st.slider("表示閾値（訪問者の %）", 0.5, 20.0, 5.0, step=0.5)

    run_dooh = st.button("▶ DOOH 分析実行", type="primary")
    if run_dooh:
        sel_ids = (hotel_stats[hotel_stats["hotel_name"].isin(sel_hotels)]["hotel_id"].tolist()
                   if sel_hotels else None)
        n_sel = (prev_night_df[prev_night_df["hotel_id"].isin(sel_ids)]["member_id"].nunique()
                 if sel_ids else prev_night_df["member_id"].nunique())
        with st.spinner("DOOH 通過分析中..."):
            passages = compute_dooh_passages(
                gps_df, prev_night_df, DOOH_DF,
                selected_hotel_ids=sel_ids,
                passage_r_m=dooh_r,
            )
        st.session_state["dooh_passages_df"] = passages
        st.session_state["dooh_n_sel"] = n_sel

    if st.session_state["dooh_passages_df"] is None:
        st.info("分析実行ボタンを押してください。")
        st.stop()

    passages_df: pd.DataFrame = st.session_state["dooh_passages_df"]
    n_sel_vis = st.session_state["dooh_n_sel"]

    if passages_df.empty:
        st.warning("DOOH 設置場所付近の通過記録が見つかりませんでした。"
                   "検知半径または閾値を調整してください。")
        st.stop()

    # DOOH 統計
    dooh_stats = (
        passages_df
        .groupby(["dooh_id", "dooh_name", "dooh_lat", "dooh_lon"])["member_id"]
        .nunique().reset_index()
        .rename(columns={"member_id": "passers"})
    )
    dooh_stats["pct"] = dooh_stats["passers"] / len(store_visitors) * 100
    qualifying = (dooh_stats[dooh_stats["pct"] >= dooh_thr]
                  .sort_values("pct", ascending=False)
                  .reset_index(drop=True))

    st.metric(f"閾値 {dooh_thr}% 超の DOOH 設置場所",
              f"{len(qualifying)} 箇所 / {len(DOOH_DF)} 箇所")

    if qualifying.empty:
        st.warning("閾値を超える DOOH 設置場所がありません。閾値を下げてください。")
        st.stop()

    # ── DOOH 地図 ──────────────────────────────────────────────────────────────
    st.subheader("🗺️ DOOH 通過ポイントマップ")
    fig_dm = go.Figure()

    # 百貨店
    fig_dm.add_trace(go.Scattermapbox(
        lat=[store_lat], lon=[store_lon], mode="markers+text",
        marker=dict(size=20, color="#27ae60", symbol="star"),
        text=[store_name], textposition="top right", name=store_name,
    ))
    # 非通過 DOOH（薄く）
    non_q = dooh_stats[~dooh_stats["dooh_id"].isin(qualifying["dooh_id"])]
    if not non_q.empty:
        fig_dm.add_trace(go.Scattermapbox(
            lat=non_q["dooh_lat"], lon=non_q["dooh_lon"], mode="markers",
            marker=dict(size=7, color="#bdc3c7", opacity=0.5),
            name="DOOH（閾値未満）", hoverinfo="skip",
        ))
    # 推奨 DOOH（赤、サイズ ∝ pct）
    fig_dm.add_trace(go.Scattermapbox(
        lat=qualifying["dooh_lat"], lon=qualifying["dooh_lon"],
        mode="markers+text",
        marker=dict(
            size=[max(14, min(45, int(p * 2.5))) for p in qualifying["pct"]],
            color="#e74c3c", opacity=0.90,
        ),
        text=qualifying["dooh_name"], textposition="top right",
        customdata=qualifying[["passers", "pct"]].values,
        hovertemplate="<b>%{text}</b><br>通過者: %{customdata[0]} 人 (%{customdata[1]:.1f}%)<extra></extra>",
        name="📺 推奨 DOOH",
    ))
    fig_dm.update_layout(
        mapbox=dict(style="open-street-map",
                    center=dict(lat=store_lat, lon=store_lon), zoom=14),
        height=480, margin=dict(r=0,t=0,l=0,b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.88)"),
    )
    st.plotly_chart(fig_dm, use_container_width=True)

    # DOOH 統計テーブル
    st.dataframe(
        qualifying[["dooh_name","passers","pct"]].rename(columns={
            "dooh_name": "DOOH 設置場所", "passers": "通過者数", "pct": "訪問者割合 (%)",
        }).assign(**{"訪問者割合 (%)": qualifying["pct"].round(1)}),
        use_container_width=False, hide_index=True,
    )

    # ── サンキーダイアグラム ────────────────────────────────────────────────────
    st.subheader("🔀 サンキーダイアグラム（ホテル → DOOH 通過 → 百貨店）")
    sankey_fig = build_sankey(
        passages_df, prev_night_df,
        store_name, len(store_visitors), dooh_thr,
    )
    if sankey_fig:
        st.plotly_chart(sankey_fig, use_container_width=True)
        st.caption("🔵 ホテル　🔴 DOOH 設置場所　🟢 百貨店")
    else:
        st.info("サンキーダイアグラムの生成に必要なデータが不足しています。")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4b: 来店直前の経路特定
# ═════════════════════════════════════════════════════════════════════════════
elif mode == "🗺️ 来店直前の経路特定":
    st.divider()
    st.header("④ 来店直前の経路特定")
    st.caption(
        "百貨店到着前の GPS 移動軌跡を可視化し、チラシ・ポスター配布の最適場所を特定します。\n"
        "※ 本デモでは GPS 滞在点の時系列連結による簡易経路近似を使用しています。"
        "Plateau 道路データへのマップマッチングは別途実装可能です。"
    )

    col_ra, col_rb = st.columns(2)
    with col_ra:
        pre_min   = st.slider("来店前の分析時間窓 (分)", 10, 60, 30, step=5)
    with col_rb:
        route_thr = st.slider("表示閾値（訪問者の %）", 0.5, 30.0, 5.0, step=0.5)

    run_route = st.button("▶ 経路分析実行", type="primary")
    if run_route:
        with st.spinner("経路分析中..."):
            segs = analyze_pre_arrival_routes(
                gps_df, store_visits_df, store_lat, store_lon,
                pre_minutes=pre_min, threshold_pct=route_thr,
            )
        st.session_state["route_segs_df"] = segs

    if st.session_state["route_segs_df"] is None:
        st.info("分析実行ボタンを押してください。")
        st.stop()

    segs_df: pd.DataFrame = st.session_state["route_segs_df"]

    if segs_df.empty:
        st.warning("閾値を超える経路セグメントが見つかりませんでした。閾値を下げてください。")
        st.stop()

    n_total = store_visits_df["member_id"].nunique()
    n_app   = segs_df[segs_df["approaching"]]["count"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("有効セグメント数",        f"{len(segs_df):,}")
    c2.metric("店舗方向 延べ通行数",     f"{n_app:,}")
    c3.metric("分析対象来訪者",          f"{n_total:,}")

    # ── 経路地図 ──────────────────────────────────────────────────────────────
    st.subheader("🗺️ 来店前経路マップ")
    st.caption("🟠 オレンジ太線: 店舗方向  ｜  🔵 ブルー細線: 逆方向・横断  ｜  線幅 = 通行者数に比例")

    fig_rt = go.Figure()
    max_cnt = segs_df["count"].max()

    # 方向別に 2 トレース（凡例用）
    for approaching, color, label in [
        (True,  "rgba(230,126,34,0.80)", "→ 店舗方向"),
        (False, "rgba(52,152,219,0.65)", "← 逆方向・横断"),
    ]:
        grp = segs_df[segs_df["approaching"] == approaching]
        if grp.empty:
            continue
        # 各セグメントを個別トレースとして追加（線幅を変えるため）
        for _, seg in grp.iterrows():
            w = max(1.5, min(10, seg["count"] / max_cnt * 10))
            fig_rt.add_trace(go.Scattermapbox(
                lat=[seg.lat1, seg.lat2],
                lon=[seg.lon1, seg.lon2],
                mode="lines",
                line=dict(color=color, width=w),
                hovertemplate=(
                    f"{label}<br>{seg['count']} 人 ({seg['pct']:.1f}%)<extra></extra>"
                ),
                showlegend=False,
            ))
        # 凡例ダミー
        fig_rt.add_trace(go.Scattermapbox(
            lat=[store_lat], lon=[store_lon], mode="markers",
            marker=dict(size=0, opacity=0), name=label, showlegend=True,
        ))

    # 百貨店
    fig_rt.add_trace(go.Scattermapbox(
        lat=[store_lat], lon=[store_lon], mode="markers+text",
        marker=dict(size=20, color="#e74c3c", symbol="star"),
        text=[store_name], textposition="top right", name=store_name,
    ))

    fig_rt.update_layout(
        mapbox=dict(style="open-street-map",
                    center=dict(lat=store_lat, lon=store_lon), zoom=15),
        height=560, margin=dict(r=0,t=0,l=0,b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.9)"),
    )
    st.plotly_chart(fig_rt, use_container_width=True)

    # ── 配布推奨ポイント ──────────────────────────────────────────────────────
    st.subheader("💡 チラシ・ポスター配布 推奨ポイント（店舗方向通行 上位 5 箇所）")
    top5 = segs_df[segs_df["approaching"]].head(5)
    if top5.empty:
        st.info("店舗方向の有効セグメントが見つかりませんでした。")
    else:
        for rank, (_, seg) in enumerate(top5.iterrows(), 1):
            mid_lat = (seg.lat1 + seg.lat2) / 2
            mid_lon = (seg.lon1 + seg.lon2) / 2
            st.markdown(
                f"**{rank}位** 座標 ({mid_lat:.5f}, {mid_lon:.5f})"
                f" ― **{seg['count']} 人通行** ({seg['pct']:.1f}%) ｜ "
                f"方位 {seg['bearing']:.0f}° → 店舗方向"
            )

    # セグメント一覧
    with st.expander("全セグメント一覧（上位 30）"):
        show = segs_df.head(30).copy()
        show["方向"] = show["approaching"].map({True: "→ 店舗方向", False: "← 逆/横断"})
        show["通行率 (%)"] = show["pct"].round(1)
        st.dataframe(
            show[["lat1","lon1","lat2","lon2","count","通行率 (%)","方向"]],
            use_container_width=True, hide_index=True,
        )
