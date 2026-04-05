# -*- coding: utf-8 -*-
"""
Liveboard 東京 DOOH 設置場所データ取得スクリプト
出典: https://liveboard.co.jp/screen/tokyo/area.html?local=all
実行: python fetch_liveboard.py
出力: liveboard_tokyo.csv
"""

import re
import json
import urllib.request


URL = "https://liveboard.co.jp/screen/tokyo/area.html?local=all"

SCREEN_TYPE_MAP = {
    "1": "屋外",
    "2": "交通",
    "3": "商業施設",
    "4": "その他",
}

OWNERSHIP_MAP = {
    "1": "LIVE BOARD 直営",
    "2": "パートナー",
}


def fetch_liveboard_tokyo():
    req = urllib.request.Request(
        URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="replace")

    # script タグから DETAIL_DATA を抽出（ブラケットカウント方式）
    m = re.search(r"DETAIL_DATA\s*=\s*\[", html)
    if not m:
        raise ValueError("DETAIL_DATA が見つかりませんでした")

    start = m.end() - 1  # '[' の位置
    depth = 0
    end = start
    for i, ch in enumerate(html[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    raw = html[start:end]
    data = json.loads(raw)
    return data


FIELDS = ["id", "name", "address", "station", "size", "lat", "lon",
          "screen_type", "ownership", "sound", "url"]


def build_rows(data):
    rows = []
    for d in data:
        lat = float(d.get("lat", 0) or 0)
        lng = float(d.get("lng", 0) or 0)
        if lat == 0 or lng == 0:
            continue
        rows.append({
            "id":          d.get("uid", ""),
            "name":        d.get("title", ""),
            "address":     d.get("address", ""),
            "station":     d.get("station", ""),
            "size":        d.get("size", ""),
            "lat":         lat,
            "lon":         lng,
            "screen_type": SCREEN_TYPE_MAP.get(d.get("screenType", ""), "不明"),
            "ownership":   OWNERSHIP_MAP.get(d.get("ownership", ""), "不明"),
            "sound":       "音あり" if d.get("sound") == "1" else "サイレント",
            "url":         "https://liveboard.co.jp" + d.get("permalink", ""),
        })
    return rows


def build_dataframe(data):
    import pandas as pd
    return pd.DataFrame(build_rows(data))


if __name__ == "__main__":
    import csv

    print("Liveboard 東京 DOOH データを取得中...")
    try:
        data = fetch_liveboard_tokyo()
        rows = build_rows(data)
        out  = "liveboard_tokyo.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        n = len(rows)
        print(f"✅ {n} 件の DOOH 設置場所を取得")
        st = {r["screen_type"] for r in rows}
        for s in ["屋外", "交通", "商業施設", "その他", "不明"]:
            c = sum(1 for r in rows if r["screen_type"] == s)
            if c: print(f"   {s}: {c} 件")
        print(f"   LIVE BOARD 直営: {sum(1 for r in rows if r['ownership']=='LIVE BOARD 直営')} 件")
        print(f"   パートナー: {sum(1 for r in rows if r['ownership']=='パートナー')} 件")
        print(f"\n出力: {out}")
        for r in rows[:10]:
            print(f"  {r['name']:<30} {r['station']:<10} {r['lat']:.6f} {r['lon']:.6f} {r['screen_type']}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"❌ エラー: {e}")
