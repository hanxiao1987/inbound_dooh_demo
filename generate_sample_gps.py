"""
加盟店送客エンジン用 擬似GPSデータ生成スクリプト
実行: python generate_sample_gps.py
出力: sample_gps_data.csv

想定シナリオ:
  - 対象百貨店: 新宿高島屋タイムズスクエア (35.6886, 139.7024)
  - 新宿エリアのホテルに滞在するインバウンド旅行者 300 名
  - 7 日間 (2024-03-01 〜 2024-03-07)
  - 各メンバー: ホテル宿泊 (夜間長時間滞在) → 翌日 DOOH 周辺を経由 → 百貨店訪問
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

# ── 対象百貨店: 新宿高島屋タイムズスクエア ───────────────────────────────────
STORE_LAT  = 35.6886
STORE_LON  = 139.7024
STORE_NAME = "新宿高島屋"

# ── ホテルリスト（新宿エリア）──────────────────────────────────────────────────
HOTELS = [
    {"id": "H01", "name": "ハイアット リージェンシー 東京（仮）", "lat": 35.6908, "lon": 139.6937},
    {"id": "H02", "name": "京王プラザホテル",                     "lat": 35.6946, "lon": 139.6951},
    {"id": "H03", "name": "新宿ワシントンホテル（仮）",           "lat": 35.6893, "lon": 139.6973},
    {"id": "H04", "name": "ホテルグレイスリー新宿",               "lat": 35.6930, "lon": 139.7008},
    {"id": "H05", "name": "小田急センチュリーサザンタワー",       "lat": 35.6879, "lon": 139.7022},
    {"id": "H06", "name": "新宿グランベルホテル",                 "lat": 35.6938, "lon": 139.7025},
    {"id": "H07", "name": "東横INN 新宿歌舞伎町（仮）",           "lat": 35.6952, "lon": 139.7006},
    {"id": "H08", "name": "コンフォートホテル新宿（仮）",         "lat": 35.6898, "lon": 139.7053},
    {"id": "H09", "name": "ホテルサンルート 新宿（仮）",          "lat": 35.6868, "lon": 139.6976},
    {"id": "H10", "name": "ベストウェスタン 新宿（仮）",          "lat": 35.6872, "lon": 139.7066},
]

# ── DOOH 設置場所（新宿エリア）─ 経由地候補 ──────────────────────────────────
# ※ app.py の DOOH_DF と同一座標を使用（500m 以内判定が通るよう設計）
DOOH_WAYPOINTS = [
    {"id": "D21", "lat": 35.6877, "lon": 139.7033, "name": "新宿駅南口"},
    {"id": "D22", "lat": 35.6905, "lon": 139.7013, "name": "新宿駅東口"},
    {"id": "D23", "lat": 35.6896, "lon": 139.6993, "name": "新宿駅西口"},
    {"id": "D24", "lat": 35.6887, "lon": 139.7053, "name": "新宿三丁目交差点"},
    {"id": "D25", "lat": 35.6884, "lon": 139.7023, "name": "新宿タイムズスクエア前"},
    {"id": "D26", "lat": 35.6875, "lon": 139.7015, "name": "甲州街道 新宿"},
    {"id": "D27", "lat": 35.6854, "lon": 139.7063, "name": "新宿四丁目"},
    {"id": "D28", "lat": 35.6826, "lon": 139.7023, "name": "代々木駅前"},
    {"id": "D30", "lat": 35.6938, "lon": 139.7016, "name": "新宿歌舞伎町"},
    {"id": "D31", "lat": 35.6918, "lon": 139.6980, "name": "新宿モード学園前"},
    {"id": "D32", "lat": 35.6879, "lon": 139.7006, "name": "新宿サザンテラス"},
]

# ── 日中の途中立ち寄りスポット（新宿周辺） ───────────────────────────────────
DAYTIME_SPOTS = [
    {"lat": 35.6910, "lon": 139.7000, "dur": (20, 60),  "name": "カフェ（西新宿）"},
    {"lat": 35.6882, "lon": 139.7050, "dur": (10, 30),  "name": "コンビニ（新宿三丁目）"},
    {"lat": 35.6900, "lon": 139.7035, "dur": (30, 90),  "name": "ランチ（新宿東口）"},
    {"lat": 35.6862, "lon": 139.7010, "dur": (15, 40),  "name": "ドラッグストア（南新宿）"},
    {"lat": 35.6930, "lon": 139.7020, "dur": (20, 50),  "name": "カフェ（歌舞伎町）"},
    {"lat": 35.6870, "lon": 139.7040, "dur": (10, 25),  "name": "コンビニ（新宿南口）"},
    {"lat": 35.6850, "lon": 139.7070, "dur": (25, 70),  "name": "土産物店（新宿四丁目）"},
    {"lat": 35.6920, "lon": 139.6970, "dur": (30, 80),  "name": "レストラン（西新宿）"},
    {"lat": 35.6895, "lon": 139.7010, "dur": (15, 45),  "name": "コーヒーショップ（新宿）"},
    {"lat": 35.6840, "lon": 139.7025, "dur": (20, 60),  "name": "カフェ（代々木）"},
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
        mid   = f"M{m_idx+1:05d}"
        hotel = rng.choice(HOTELS)

        # 滞在日数（3〜7 日）と百貨店訪問日
        stay_days   = rng.randint(3, N_DAYS)
        checkin_day = rng.randint(0, N_DAYS - stay_days)
        hotel_days  = list(range(checkin_day, checkin_day + stay_days))
        n_visits    = rng.randint(1, min(3, len(hotel_days)))
        store_days  = sorted(rng.sample(hotel_days, n_visits))

        # ホテルから各 DOOH までの距離で近い順にソート（経由優先度付け）
        near_dooh = sorted(
            DOOH_WAYPOINTS,
            key=lambda d: haversine_m(hotel["lat"], hotel["lon"], d["lat"], d["lon"])
        )

        for day_off in hotel_days:
            date = base + timedelta(days=day_off)

            # ── ホテル宿泊レコード（夜間 20〜23 時チェックイン、6〜9 時間） ──
            ci_h   = rng.randint(20, 23)
            ci_min = rng.randint(0, 59)
            dur_h  = rng.uniform(360, 540)
            jlat   = hotel["lat"] + np_rng.uniform(-0.0001, 0.0001)
            jlon   = hotel["lon"] + np_rng.uniform(-0.0001, 0.0001)
            records.append({
                "member_id":         mid,
                "lat":               round(jlat, 6),
                "lon":               round(jlon, 6),
                "stay_datetime":     (date + timedelta(hours=ci_h, minutes=ci_min)
                                      ).strftime("%Y-%m-%d %H:%M:%S"),
                "stay_duration_min": round(dur_h, 1),
            })

            if day_off not in store_days:
                # 百貨店訪問なし: 日中スポット 1〜2 件
                spots = rng.sample(DAYTIME_SPOTS, rng.randint(1, 2))
                hour  = rng.randint(10, 16)
                for sp in spots:
                    dur = rng.uniform(*sp["dur"])
                    jl  = sp["lat"] + np_rng.uniform(-0.0002, 0.0002)
                    jlo = sp["lon"] + np_rng.uniform(-0.0002, 0.0002)
                    records.append({
                        "member_id":         mid,
                        "lat":               round(jl, 6),
                        "lon":               round(jlo, 6),
                        "stay_datetime":     (date + timedelta(hours=hour,
                                              minutes=rng.randint(0, 59))
                                              ).strftime("%Y-%m-%d %H:%M:%S"),
                        "stay_duration_min": round(dur, 1),
                    })
                    hour += max(1, int(dur / 60) + 1)
                continue

            # ── 百貨店訪問日 ──────────────────────────────────────────────────
            hour   = rng.randint(9, 11)
            minute = rng.randint(0, 59)

            # ① ホテル近くの DOOH 1〜3 箇所を経由（GPS 滞在点として記録）
            n_wp  = rng.randint(1, 3)
            wayps = near_dooh[:6]                  # 近い 6 候補から
            wayps = rng.sample(wayps, min(n_wp, len(wayps)))

            # ホテル→百貨店方向に並べ替え（百貨店に近い順）
            wayps = sorted(
                wayps,
                key=lambda d: haversine_m(d["lat"], d["lon"], STORE_LAT, STORE_LON),
                reverse=True,  # 遠い順（ホテル寄りから）
            )

            for wp in wayps:
                dur = rng.uniform(10, 35)
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
                minute += int(dur) + rng.randint(5, 15)
                if minute >= 60:
                    hour  += minute // 60
                    minute  = minute % 60

            # ② 直前にもう 1 スポット（近隣カフェ等）
            if rng.random() < 0.5:
                sp  = rng.choice(DAYTIME_SPOTS)
                dur = rng.uniform(10, 30)
                jl  = sp["lat"] + np_rng.uniform(-0.0002, 0.0002)
                jlo = sp["lon"] + np_rng.uniform(-0.0002, 0.0002)
                records.append({
                    "member_id":         mid,
                    "lat":               round(jl, 6),
                    "lon":               round(jlo, 6),
                    "stay_datetime":     (date + timedelta(hours=hour, minutes=minute)
                                          ).strftime("%Y-%m-%d %H:%M:%S"),
                    "stay_duration_min": round(dur, 1),
                })
                minute += int(dur) + rng.randint(5, 15)
                if minute >= 60:
                    hour  += minute // 60
                    minute  = minute % 60

            # ③ 百貨店到着
            dur_store = rng.uniform(60, 180)
            jl  = STORE_LAT + np_rng.uniform(-0.0003, 0.0003)
            jlo = STORE_LON + np_rng.uniform(-0.0003, 0.0003)
            arrive = date + timedelta(hours=hour, minutes=minute + rng.randint(3, 15))
            records.append({
                "member_id":         mid,
                "lat":               round(jl, 6),
                "lon":               round(jlo, 6),
                "stay_datetime":     arrive.strftime("%Y-%m-%d %H:%M:%S"),
                "stay_duration_min": round(dur_store, 1),
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print(f"擬似 GPS データを生成中（対象: {STORE_NAME}）...")
    df = generate()
    out = "sample_gps_data.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    sz = len(df.to_csv(index=False).encode("utf-8")) / 1024
    print(f"✅ {len(df):,} レコード / {df['member_id'].nunique():,} メンバー")
    print(f"   期間: {df['stay_datetime'].min()} 〜 {df['stay_datetime'].max()}")
    print(f"   緯度範囲: {df['lat'].min():.4f} 〜 {df['lat'].max():.4f}")
    print(f"   経度範囲: {df['lon'].min():.4f} 〜 {df['lon'].max():.4f}")
    print(f"   ファイルサイズ: {sz:.0f} KB → {out}")
    print(f"\nアプリ設定:")
    print(f"   百貨店名: {STORE_NAME}")
    print(f"   緯度: {STORE_LAT} / 経度: {STORE_LON}")
    print(f"   訪問判定半径: 150m 推奨")
