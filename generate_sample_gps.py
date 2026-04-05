"""
加盟店送客エンジン用 擬似GPSデータ生成スクリプト
実行: python generate_sample_gps.py
出力: sample_gps_data.csv

想定シナリオ:
  - 対象百貨店: 新宿高島屋タイムズスクエア (35.6886, 139.7024)
  - インバウンド旅行者 300 名 / 7 日間 (2024-03-01 〜 2024-03-07)
  - 70% (210 名): 新宿エリアのホテル泊 → DOOH 周辺を経由 → 百貨店訪問
  - 30% ( 90 名): 遠方エリア（銀座・渋谷・池袋・浅草・六本木）のホテル泊
                  → 新宿駅に到着 → DOOH 周辺を経由 → 百貨店訪問
"""

import csv
import math
import random
from datetime import datetime, timedelta

import numpy as np

RANDOM_SEED      = 42
BASE_DATE_STR    = "2024-03-01"
N_MEMBERS        = 300
N_DAYS           = 7
REMOTE_RATIO     = 0.30   # 遠方ホテル泊の割合

# ── 対象百貨店: 新宿高島屋タイムズスクエア ───────────────────────────────────
STORE_LAT  = 35.6886
STORE_LON  = 139.7024
STORE_NAME = "新宿高島屋"

# ── ホテルリスト（新宿エリア・近距離）────────────────────────────────────────
HOTELS_LOCAL = [
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

# ── ホテルリスト（遠方エリア・新宿高島屋から 3km 以上）────────────────────────
HOTELS_REMOTE = [
    # 銀座・丸の内エリア (~5km)
    {"id": "R01", "name": "ホテルモントレ銀座（仮）",             "lat": 35.6714, "lon": 139.7651},
    {"id": "R02", "name": "東急ステイ銀座（仮）",                 "lat": 35.6730, "lon": 139.7620},
    {"id": "R03", "name": "パレスホテル東京（仮）",               "lat": 35.6847, "lon": 139.7594},
    # 渋谷エリア (~4km)
    {"id": "R04", "name": "セルリアンタワー東急ホテル（仮）",     "lat": 35.6573, "lon": 139.7015},
    {"id": "R05", "name": "渋谷エクセルホテル東急（仮）",         "lat": 35.6591, "lon": 139.7007},
    {"id": "R06", "name": "コンラッド東京（仮）",                 "lat": 35.6658, "lon": 139.7575},
    # 池袋エリア (~5km)
    {"id": "R07", "name": "メトロポリタンホテル池袋（仮）",       "lat": 35.7295, "lon": 139.7109},
    {"id": "R08", "name": "池袋東武ホテル（仮）",                 "lat": 35.7308, "lon": 139.7121},
    # 浅草エリア (~7km)
    {"id": "R09", "name": "アパホテル浅草（仮）",                 "lat": 35.7106, "lon": 139.7963},
    {"id": "R10", "name": "浅草ビューホテル（仮）",               "lat": 35.7147, "lon": 139.7972},
    # 六本木エリア (~4km)
    {"id": "R11", "name": "グランドハイアット東京（仮）",         "lat": 35.6601, "lon": 139.7292},
    {"id": "R12", "name": "ザ・リッツ・カールトン東京（仮）",     "lat": 35.6656, "lon": 139.7314},
]

# 後方互換: app.py の Overpass マッチングは HOTELS_LOCAL で十分
HOTELS = HOTELS_LOCAL

# ── 新宿駅周辺の乗降/経由スポット（遠方客の到着点として使用）──────────────────
SHINJUKU_ARRIVAL_SPOTS = [
    {"lat": 35.6896, "lon": 139.7006, "name": "新宿駅南口改札前"},
    {"lat": 35.6905, "lon": 139.7013, "name": "新宿駅東口"},
    {"lat": 35.6916, "lon": 139.6999, "name": "新宿駅西口"},
    {"lat": 35.6877, "lon": 139.7033, "name": "新宿駅南口ルミネ前"},
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


def _add_rec(records, mid, lat, lon, dt, dur):
    records.append({
        "member_id":         mid,
        "lat":               lat,
        "lon":               lon,
        "stay_datetime":     dt.strftime("%Y-%m-%d %H:%M:%S"),
        "stay_duration_min": round(dur, 1),
    })


def _store_visit_day(records, mid, hotel, near_dooh, rng, np_rng, date, is_remote):
    """百貨店訪問日の経路を生成（近距離・遠方共通）"""
    hour   = rng.randint(9, 11)
    minute = rng.randint(0, 59)

    # 遠方ホテル泊メンバー: 新宿駅到着スポットを最初に経由
    if is_remote:
        arr = rng.choice(SHINJUKU_ARRIVAL_SPOTS)
        dur = rng.uniform(5, 20)   # 乗換・改札通過
        jl  = arr["lat"] + np_rng.uniform(-0.0001, 0.0001)
        jlo = arr["lon"] + np_rng.uniform(-0.0001, 0.0001)
        _add_rec(records, mid, round(jl, 6), round(jlo, 6),
                 date + timedelta(hours=hour, minutes=minute), dur)
        minute += int(dur) + rng.randint(3, 10)
        if minute >= 60:
            hour += minute // 60; minute = minute % 60

    # DOOH 経由（1〜3 箇所）
    n_wp  = rng.randint(1, 3)
    wayps = near_dooh[:6]
    wayps = rng.sample(wayps, min(n_wp, len(wayps)))
    wayps = sorted(wayps,
                   key=lambda d: haversine_m(d["lat"], d["lon"], STORE_LAT, STORE_LON),
                   reverse=True)

    for wp in wayps:
        dur = rng.uniform(10, 35)
        jl  = wp["lat"] + np_rng.uniform(-0.0001, 0.0001)
        jlo = wp["lon"] + np_rng.uniform(-0.0001, 0.0001)
        _add_rec(records, mid, round(jl, 6), round(jlo, 6),
                 date + timedelta(hours=hour, minutes=minute), dur)
        minute += int(dur) + rng.randint(5, 15)
        if minute >= 60:
            hour += minute // 60; minute = minute % 60

    # 直前カフェ等（50% 確率）
    if rng.random() < 0.5:
        sp  = rng.choice(DAYTIME_SPOTS)
        dur = rng.uniform(10, 30)
        jl  = sp["lat"] + np_rng.uniform(-0.0002, 0.0002)
        jlo = sp["lon"] + np_rng.uniform(-0.0002, 0.0002)
        _add_rec(records, mid, round(jl, 6), round(jlo, 6),
                 date + timedelta(hours=hour, minutes=minute), dur)
        minute += int(dur) + rng.randint(5, 15)
        if minute >= 60:
            hour += minute // 60; minute = minute % 60

    # 百貨店到着
    dur_store = rng.uniform(60, 180)
    jl  = STORE_LAT + np_rng.uniform(-0.0003, 0.0003)
    jlo = STORE_LON + np_rng.uniform(-0.0003, 0.0003)
    arrive = date + timedelta(hours=hour, minutes=minute + rng.randint(3, 15))
    _add_rec(records, mid, round(jl, 6), round(jlo, 6), arrive, dur_store)


FIELDS = ["member_id", "lat", "lon", "stay_datetime", "stay_duration_min"]


def generate():
    rng    = random.Random(RANDOM_SEED)
    np_rng = np.random.default_rng(RANDOM_SEED)
    base   = datetime.strptime(BASE_DATE_STR, "%Y-%m-%d")

    n_remote = int(N_MEMBERS * REMOTE_RATIO)   # 90 名: 遠方ホテル泊

    records = []

    for m_idx in range(N_MEMBERS):
        mid       = "M{:05d}".format(m_idx + 1)
        is_remote = m_idx < n_remote           # 先頭 30% を遠方グループに
        hotel     = (rng.choice(HOTELS_REMOTE) if is_remote
                     else rng.choice(HOTELS_LOCAL))

        # 滞在日数（3〜7 日）と百貨店訪問日
        stay_days   = rng.randint(3, N_DAYS)
        checkin_day = rng.randint(0, N_DAYS - stay_days)
        hotel_days  = list(range(checkin_day, checkin_day + stay_days))
        n_visits    = rng.randint(1, min(3, len(hotel_days)))
        store_days  = sorted(rng.sample(hotel_days, n_visits))

        # DOOH の距離優先順（新宿駅系を先に）
        near_dooh = sorted(
            DOOH_WAYPOINTS,
            key=lambda d: haversine_m(hotel["lat"], hotel["lon"], d["lat"], d["lon"])
        )

        for day_off in hotel_days:
            date = base + timedelta(days=day_off)

            # ── ホテル宿泊レコード ──
            ci_h   = rng.randint(20, 23)
            ci_min = rng.randint(0, 59)
            dur_h  = rng.uniform(360, 540)
            jlat   = hotel["lat"] + np_rng.uniform(-0.0001, 0.0001)
            jlon   = hotel["lon"] + np_rng.uniform(-0.0001, 0.0001)
            _add_rec(records, mid, round(jlat, 6), round(jlon, 6),
                     date + timedelta(hours=ci_h, minutes=ci_min), dur_h)

            if day_off not in store_days:
                # 百貨店訪問なし: 日中スポット 1〜2 件
                hour = rng.randint(10, 16)
                for sp in rng.sample(DAYTIME_SPOTS, rng.randint(1, 2)):
                    dur = rng.uniform(*sp["dur"])
                    jl  = sp["lat"] + np_rng.uniform(-0.0002, 0.0002)
                    jlo = sp["lon"] + np_rng.uniform(-0.0002, 0.0002)
                    _add_rec(records, mid, round(jl, 6), round(jlo, 6),
                             date + timedelta(hours=hour, minutes=rng.randint(0, 59)), dur)
                    hour += max(1, int(dur / 60) + 1)
            else:
                _store_visit_day(records, mid, hotel, near_dooh,
                                 rng, np_rng, date, is_remote)

    # shuffle
    rng2 = random.Random(RANDOM_SEED)
    rng2.shuffle(records)
    return records


if __name__ == "__main__":
    print("擬似 GPS データを生成中（対象: {}）...".format(STORE_NAME))
    records = generate()
    out = "sample_gps_data.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    n = len(records)
    members = len({r["member_id"] for r in records})
    lats = [r["lat"] for r in records]
    lons = [r["lon"] for r in records]
    dts  = sorted(r["stay_datetime"] for r in records)
    sz   = sum(len(",".join(str(r[f]) for f in FIELDS) + "\n").bit_length() // 8
               for r in records) // 1024  # rough estimate
    import os
    sz_kb = os.path.getsize(out) // 1024

    print("✅ {:,} レコード / {:,} メンバー".format(n, members))
    print("   期間: {} 〜 {}".format(dts[0], dts[-1]))
    print("   緯度範囲: {:.4f} 〜 {:.4f}".format(min(lats), max(lats)))
    print("   経度範囲: {:.4f} 〜 {:.4f}".format(min(lons), max(lons)))
    print("   ファイルサイズ: {} KB → {}".format(sz_kb, out))
    print("\nアプリ設定:")
    print("   百貨店名: {}".format(STORE_NAME))
    print("   緯度: {} / 経度: {}".format(STORE_LAT, STORE_LON))
    print("   訪問判定半径: 150m 推奨")
