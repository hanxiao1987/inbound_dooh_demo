"""
加盟店送客エンジン用 擬似GPSデータ生成スクリプト
実行: python generate_sample_gps.py
出力: sample_gps_data.csv

想定シナリオ:
  - 対象百貨店: 松屋銀座 (35.6722, 139.7658)
  - 銀座・新橋エリアのホテルに滞在するインバウンド旅行者 300 名
  - 7 日間 (2024-03-01 〜 2024-03-07)
  - 各メンバー: ホテル宿泊 (夜間長時間滞在) + 百貨店訪問 + 移動途中の滞在点
"""

import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RANDOM_SEED   = 42
BASE_DATE_STR = "2024-03-01"
N_MEMBERS     = 300
N_DAYS        = 7

# ── 対象百貨店 ────────────────────────────────────────────────────────────────
STORE_LAT = 35.6722
STORE_LON = 139.7658

# ── ホテルリスト（銀座・新橋エリア） ─────────────────────────────────────────
HOTELS = [
    {"id": "H01", "name": "ザ・ペニンシュラ東京（仮）",   "lat": 35.6762, "lon": 139.7582},
    {"id": "H02", "name": "東横INN 銀座一丁目",           "lat": 35.6731, "lon": 139.7696},
    {"id": "H03", "name": "メルキュール 東京銀座",         "lat": 35.6694, "lon": 139.7671},
    {"id": "H04", "name": "パークホテル東京（仮）",        "lat": 35.6633, "lon": 139.7588},
    {"id": "H05", "name": "コンラッド東京（仮）",          "lat": 35.6599, "lon": 139.7585},
    {"id": "H06", "name": "ホテル モントレ 銀座",          "lat": 35.6726, "lon": 139.7678},
    {"id": "H07", "name": "ドーミーイン 銀座",             "lat": 35.6686, "lon": 139.7667},
    {"id": "H08", "name": "Villa Fontaine 新橋",           "lat": 35.6659, "lon": 139.7601},
    {"id": "H09", "name": "ホテルグレイスリー 銀座",       "lat": 35.6706, "lon": 139.7661},
    {"id": "H10", "name": "東急ステイ銀座（仮）",          "lat": 35.6718, "lon": 139.7640},
]

# ── DOOH 設置場所（経由地候補） ───────────────────────────────────────────────
DOOH_WAYPOINTS = [
    {"id": "D01", "lat": 35.6714, "lon": 139.7651},  # 銀座四丁目
    {"id": "D02", "lat": 35.6753, "lon": 139.7620},  # 有楽町
    {"id": "D03", "lat": 35.6731, "lon": 139.7666},  # 銀座一丁目
    {"id": "D04", "lat": 35.6680, "lon": 139.7659},  # 東銀座
    {"id": "D05", "lat": 35.6706, "lon": 139.7651},  # 銀座三丁目
    {"id": "D06", "lat": 35.6737, "lon": 139.7585},  # 日比谷
    {"id": "D07", "lat": 35.6659, "lon": 139.7575},  # 新橋 SL広場
    {"id": "D08", "lat": 35.6700, "lon": 139.7645},  # 銀座 ITOYA前
    {"id": "D13", "lat": 35.6757, "lon": 139.7683},  # 京橋
    {"id": "D14", "lat": 35.6715, "lon": 139.7636},  # 銀座柳通り
    {"id": "D15", "lat": 35.6695, "lon": 139.7660},  # 晴海通り
    {"id": "D20", "lat": 35.6746, "lon": 139.7633},  # 有楽町交差点
]

# ── 途中立ち寄り場所（百貨店以外の日中スポット） ──────────────────────────────
DAYTIME_SPOTS = [
    {"lat": 35.6693, "lon": 139.7635, "dur": (20, 60),   "name": "カフェ"},
    {"lat": 35.6710, "lon": 139.7620, "dur": (10, 30),   "name": "コンビニ"},
    {"lat": 35.6740, "lon": 139.7600, "dur": (30, 90),   "name": "レストラン"},
    {"lat": 35.6720, "lon": 139.7690, "dur": (15, 45),   "name": "ドラッグストア"},
    {"lat": 35.6680, "lon": 139.7640, "dur": (20, 50),   "name": "カフェ"},
    {"lat": 35.6700, "lon": 139.7610, "dur": (10, 25),   "name": "コンビニ"},
    {"lat": 35.6750, "lon": 139.7650, "dur": (25, 70),   "name": "土産物店"},
    {"lat": 35.6730, "lon": 139.7630, "dur": (30, 80),   "name": "ランチ"},
]


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def generate() -> pd.DataFrame:
    rng    = random.Random(RANDOM_SEED)
    np_rng = np.random.default_rng(RANDOM_SEED)
    base   = datetime.strptime(BASE_DATE_STR, "%Y-%m-%d")

    records = []

    for m_idx in range(N_MEMBERS):
        mid = f"M{m_idx+1:05d}"
        hotel = rng.choice(HOTELS)

        # 滞在日数（3〜7 日）
        stay_days = rng.randint(3, N_DAYS)
        checkin_day = rng.randint(0, N_DAYS - stay_days)

        # 百貨店訪問日（ホテル滞在中に 1〜3 日）
        hotel_days = list(range(checkin_day, checkin_day + stay_days))
        n_store_visits = rng.randint(1, min(3, len(hotel_days)))
        store_days = sorted(rng.sample(hotel_days, n_store_visits))

        for day_off in hotel_days:
            date = base + timedelta(days=day_off)

            # ── ホテル宿泊レコード（夜間） ────────────────────────────────────
            checkin_h   = rng.randint(20, 23)
            checkin_min = rng.randint(0, 59)
            dur_hotel   = rng.uniform(360, 540)  # 6〜9 時間
            jlat = hotel["lat"] + np_rng.uniform(-0.0001, 0.0001)
            jlon = hotel["lon"] + np_rng.uniform(-0.0001, 0.0001)
            records.append({
                "member_id":         mid,
                "lat":               round(jlat, 6),
                "lon":               round(jlon, 6),
                "stay_datetime":     (date + timedelta(hours=checkin_h, minutes=checkin_min)
                                      ).strftime("%Y-%m-%d %H:%M:%S"),
                "stay_duration_min": round(dur_hotel, 1),
            })

            if day_off not in store_days:
                # 百貨店訪問なし: 日中に観光スポット 1〜2 件
                n_spots = rng.randint(1, 2)
                spots   = rng.sample(DAYTIME_SPOTS, n_spots)
                hour    = rng.randint(10, 16)
                for sp in spots:
                    dur = rng.uniform(*sp["dur"])
                    jl  = sp["lat"] + np_rng.uniform(-0.0002, 0.0002)
                    jlo = sp["lon"] + np_rng.uniform(-0.0002, 0.0002)
                    records.append({
                        "member_id":         mid,
                        "lat":               round(jl, 6),
                        "lon":               round(jlo, 6),
                        "stay_datetime":     (date + timedelta(hours=hour,
                                              minutes=rng.randint(0,59))
                                              ).strftime("%Y-%m-%d %H:%M:%S"),
                        "stay_duration_min": round(dur, 1),
                    })
                    hour += int(dur / 60) + 1
                continue

            # ── 百貨店訪問日 ─────────────────────────────────────────────────
            # ① ホテル出発後の移動: DOOH 周辺スポットを 1〜3 件経由
            # ホテルからの距離でなるべく近い DOOH を選ぶ
            near_dooh = sorted(
                DOOH_WAYPOINTS,
                key=lambda d: haversine_m(hotel["lat"], hotel["lon"], d["lat"], d["lon"])
            )
            n_waypoints = rng.randint(1, 3)
            waypoints   = rng.sample(near_dooh[:8], min(n_waypoints, 8))

            hour = rng.randint(9, 11)
            minute = rng.randint(0, 59)

            for wp in waypoints:
                dur = rng.uniform(10, 40)
                jl  = wp["lat"] + np_rng.uniform(-0.0001, 0.0001)
                jlo = wp["lon"] + np_rng.uniform(-0.0001, 0.0001)
                records.append({
                    "member_id":         mid,
                    "lat":               round(jl, 6),
                    "lon":               round(jlo, 6),
                    "stay_datetime":     (date + timedelta(hours=hour, minutes=minute)
                                          ).strftime("%Y-%m-%d %H:%M:%S"),
                    "stay_duration_min": round(dur, 1),
                })
                minute += int(dur) + rng.randint(5, 20)
                if minute >= 60:
                    hour  += minute // 60
                    minute = minute % 60

            # ② 百貨店到着
            store_dur = rng.uniform(60, 180)
            jl  = STORE_LAT + np_rng.uniform(-0.0002, 0.0002)
            jlo = STORE_LON + np_rng.uniform(-0.0002, 0.0002)
            store_arrive = date + timedelta(hours=hour, minutes=minute + rng.randint(5, 20))
            records.append({
                "member_id":         mid,
                "lat":               round(jl, 6),
                "lon":               round(jlo, 6),
                "stay_datetime":     store_arrive.strftime("%Y-%m-%d %H:%M:%S"),
                "stay_duration_min": round(store_dur, 1),
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("擬似 GPS データを生成中...")
    df = generate()
    out = "sample_gps_data.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    sz = len(df.to_csv(index=False).encode("utf-8")) / 1024
    print(f"✅ {len(df):,} レコード / {df['member_id'].nunique():,} メンバー")
    print(f"   期間: {df['stay_datetime'].min()} 〜 {df['stay_datetime'].max()}")
    print(f"   ファイルサイズ: {sz:.0f} KB → {out}")
