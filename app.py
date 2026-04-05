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
import struct
import zlib
import xml.etree.ElementTree as _ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Liveboard 実データ取得
# ─────────────────────────────────────────────────────────────────────────────
import re

_LIVEBOARD_URL = "https://liveboard.co.jp/screen/tokyo/area.html?local=all"
_SCREEN_TYPE_MAP = {"1": "屋外", "2": "交通", "3": "商業施設", "4": "その他"}
_OWNERSHIP_MAP   = {"1": "LIVE BOARD 直営", "2": "パートナー"}


@st.cache_data(ttl=3600, show_spinner=False)
def load_dooh_df() -> pd.DataFrame:
    """Liveboard 東京 DOOH 設置場所を取得して DataFrame を返す（1時間キャッシュ）。
    取得失敗時はフォールバックの主要地点を返す。"""
    try:
        req = urllib.request.Request(
            _LIVEBOARD_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="replace")

        m = re.search(r"DETAIL_DATA\s*=\s*\[", html)
        if not m:
            raise ValueError("DETAIL_DATA not found")

        start = m.end() - 1
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

        data = json.loads(html[start:end])
        rows = []
        for d in data:
            lat = float(d.get("lat", 0) or 0)
            lng = float(d.get("lng", 0) or 0)
            if lat == 0 or lng == 0:
                continue
            rows.append({
                "id":          d.get("uid", ""),
                "name":        d.get("title", ""),
                "station":     d.get("station", ""),
                "lat":         lat,
                "lon":         lng,
                "screen_type": _SCREEN_TYPE_MAP.get(d.get("screenType", ""), "不明"),
                "ownership":   _OWNERSHIP_MAP.get(d.get("ownership", ""), "不明"),
                "url":         "https://liveboard.co.jp" + d.get("permalink", ""),
            })
        return pd.DataFrame(rows)
    except Exception:
        # フォールバック: 主要地点
        return pd.DataFrame([
            {"id": "F01", "name": "新宿三丁目駅前",    "station": "新宿三丁目", "lat": 35.691012, "lon": 139.704566, "screen_type": "交通", "ownership": "不明", "url": ""},
            {"id": "F02", "name": "渋谷フクラスビジョン", "station": "渋谷",      "lat": 35.657965, "lon": 139.700307, "screen_type": "交通", "ownership": "不明", "url": ""},
            {"id": "F03", "name": "東京駅メトロビジョン", "station": "東京",      "lat": 35.682020, "lon": 139.764845, "screen_type": "交通", "ownership": "不明", "url": ""},
        ])




# ─────────────────────────────────────────────────────────────────────────────
# Plateau 道路ネットワーク取得
# ─────────────────────────────────────────────────────────────────────────────
_GML_NS  = "http://www.opengis.net/gml"
_TRAN_NS = "http://www.opengis.net/citygml/transportation/2.0"
_TRAN_NS1= "http://www.opengis.net/citygml/transportation/1.0"


@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_plateau_catalog_road() -> dict:
    catalog = {}
    rows_per_page, start = 100, 0
    while True:
        url = (f"https://www.geospatial.jp/ckan/api/3/action/package_search"
               f"?fq=tags:PLATEAU&rows={rows_per_page}&start={start}")
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read())
        results = data["result"]["results"]
        total   = data["result"]["count"]
        for item in results:
            name = item.get("name", "")
            m = re.match(r"^plateau-(\d{5})-.*-(\d{4})$", name)
            if m:
                muni_cd = m.group(1); year = int(m.group(2))
                if not catalog.get(muni_cd) or int(catalog[muni_cd].split("-")[-1]) < year:
                    catalog[muni_cd] = name
        start += rows_per_page
        if start >= total:
            break
    return catalog


def _gsi_muni_code(lat: float, lon: float) -> Optional[str]:
    url = (f"https://mreversegeocoder.gsi.go.jp/reverse-geocoder/"
           f"LonLatToAddress?lat={lat}&lon={lon}")
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())["results"]["muniCd"]
    except Exception:
        return None


def _plateau_zip_url(dataset_id: str) -> Optional[str]:
    url = f"https://www.geospatial.jp/ckan/api/3/action/package_show?id={dataset_id}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        for res in data["result"]["resources"]:
            rname = res.get("name", ""); rurl = res.get("url", "")
            if "CityGML" in rname and rurl.endswith(".zip"):
                if "v3" in rname or "v3" in rurl:
                    return rurl
        for res in data["result"]["resources"]:
            rurl = res.get("url", "")
            if "CityGML" in res.get("name", "") and rurl.endswith(".zip"):
                return rurl
    except Exception:
        pass
    return None


def _zip_cd_for_layer(zip_url: str, layer: str) -> dict:
    req = urllib.request.Request(zip_url, headers={"Range": "bytes=-65536"})
    with urllib.request.urlopen(req, timeout=30) as r:
        tail = r.read()
    pos = tail.rfind(b"PK\x05\x06")
    if pos == -1:
        raise ValueError("ZIP EOCD not found")
    eocd = tail[pos:]
    cd_size   = struct.unpack_from("<I", eocd, 12)[0]
    cd_offset = struct.unpack_from("<I", eocd, 16)[0]
    with urllib.request.urlopen(
        urllib.request.Request(
            zip_url, headers={"Range": f"bytes={cd_offset}-{cd_offset+cd_size-1}"}),
        timeout=30) as r:
        cd_data = r.read()
    files = {}
    off = 0
    while off + 46 <= len(cd_data):
        if cd_data[off:off+4] != b"PK\x01\x02":
            break
        method      = struct.unpack_from("<H", cd_data, off+10)[0]
        comp_size   = struct.unpack_from("<I", cd_data, off+20)[0]
        fname_len   = struct.unpack_from("<H", cd_data, off+28)[0]
        extra_len   = struct.unpack_from("<H", cd_data, off+30)[0]
        comment_len = struct.unpack_from("<H", cd_data, off+32)[0]
        local_off   = struct.unpack_from("<I", cd_data, off+42)[0]
        fname = cd_data[off+46:off+46+fname_len].decode("utf-8", errors="replace")
        if layer in fname and fname.endswith(".gml"):
            files[fname.split("/")[-1]] = (local_off, comp_size, method)
        off += 46 + fname_len + extra_len + comment_len
    return files


def _extract_gml(zip_url: str, local_off: int, comp_size: int, method: int) -> bytes:
    with urllib.request.urlopen(
        urllib.request.Request(
            zip_url, headers={"Range": f"bytes={local_off}-{local_off+29}"}),
        timeout=30) as r:
        lh = r.read()
    data_start = local_off + 30 + struct.unpack_from("<H", lh, 26)[0] + struct.unpack_from("<H", lh, 28)[0]
    with urllib.request.urlopen(
        urllib.request.Request(
            zip_url, headers={"Range": f"bytes={data_start}-{data_start+comp_size-1}"}),
        timeout=90) as r:
        comp = r.read()
    return zlib.decompress(comp, -15) if method == 8 else comp


def _encode_mesh3(lat: float, lon: float) -> str:
    p  = int(lat * 1.5);          u  = int(lon) - 100
    q  = int((lat * 1.5 - p) * 8); v  = int((lon - int(lon)) * 8)
    r  = int(((lat * 1.5 - p) * 8 - q) * 10)
    w  = int((((lon - int(lon)) * 8 - v)) * 10)
    return f"{p:02d}{u:02d}{q}{v}{r}{w}"


def parse_citygml_roads(gml_bytes: bytes) -> list:
    try:
        root = _ET.fromstring(gml_bytes)
    except Exception:
        return []
    dim = 3
    for el in root.iter():
        sd = el.get(f"{{{_GML_NS}}}srsDimension") or el.get("srsDimension")
        if sd:
            dim = int(sd); break
    srs = next((el.get("srsName") for el in root.iter() if el.get("srsName")), "")
    swap = not ("4326" in srs or "WGS" in srs.upper())
    tran = _TRAN_NS if root.find(f".//{{{_TRAN_NS}}}Road") is not None else _TRAN_NS1
    result = []
    for road in root.findall(f".//{{{tran}}}Road"):
        for pos_el in road.iter(f"{{{_GML_NS}}}posList"):
            text = (pos_el.text or "").strip()
            if not text:
                continue
            vals = [float(v) for v in text.split()]
            pts = [(vals[i+1], vals[i]) if swap else (vals[i], vals[i+1])
                   for i in range(0, len(vals) - dim + 1, dim)]
            if len(pts) >= 2:
                result.append(pts)
    return result


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_plateau_roads(center_lat: float, center_lon: float,
                         radius_m: float) -> tuple:
    """Plateau CityGML から道路中心線を取得。GMLを並列ダウンロードで高速化。
    Returns (road_segments, status_msg)"""
    lat_sc = 111320.0
    lon_sc = 111320.0 * math.cos(math.radians(center_lat))
    dlat = radius_m / lat_sc * 1.2
    dlon = radius_m / lon_sc * 1.2

    mesh_sz_lat = (2.0/3.0) / 80
    mesh_sz_lon = 1.0 / 80
    prefixes: set = set()
    la = math.floor((center_lat - dlat) / mesh_sz_lat) * mesh_sz_lat
    while la <= center_lat + dlat + mesh_sz_lat:
        lo = math.floor((center_lon - dlon) / mesh_sz_lon) * mesh_sz_lon
        while lo <= center_lon + dlon + mesh_sz_lon:
            prefixes.add(_encode_mesh3(la + mesh_sz_lat/2, lo + mesh_sz_lon/2)[:8])
            lo += mesh_sz_lon
        la += mesh_sz_lat

    catalog = _fetch_plateau_catalog_road()
    muni_cd = _gsi_muni_code(center_lat, center_lon)
    if not muni_cd:
        return [], "❌ 市区町村コード取得失敗"
    dataset_id = catalog.get(muni_cd)
    if not dataset_id:
        return [], f"⚠️ {muni_cd} の Plateau データなし"
    zip_url = _plateau_zip_url(dataset_id)
    if not zip_url:
        return [], "⚠️ ZIP URL 取得失敗"

    cd = _zip_cd_for_layer(zip_url, "tran")
    needed = {f: info for f, info in cd.items()
              if any(f.startswith(p) for p in prefixes)}
    if not needed:
        return [], "⚠️ 対象エリアの道路 GML なし"

    # GML ファイルを並列ダウンロード（最大 6 並列）
    all_segs: list = []

    def _fetch_one(item):
        _, (off, comp_sz, method) = item
        try:
            return parse_citygml_roads(_extract_gml(zip_url, off, comp_sz, method))
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=min(6, len(needed))) as ex:
        for segs in ex.map(_fetch_one, needed.items()):
            all_segs.extend(segs)

    return all_segs, f"✅ Plateau 道路セグメント {len(all_segs):,} 本取得"


def build_road_network(road_segments: list):
    """
    道路セグメント [(lon,lat),...] リストから networkx グラフと
    scipy sparse 行列を構築。
    Returns (G, node_arr, node_key_to_id, sp_graph)
    """
    import networkx as nx
    from scipy.sparse import csr_matrix

    G = nx.Graph()
    node_key_to_id: dict = {}
    node_coords: list = []

    def get_nid(lon, lat):
        key = (round(lon, 5), round(lat, 5))
        if key not in node_key_to_id:
            nid = len(node_key_to_id)
            node_key_to_id[key] = nid
            G.add_node(nid, lon=lon, lat=lat)
            node_coords.append([lon, lat])
        return node_key_to_id[key]

    for seg in road_segments:
        prev = None
        for lon, lat in seg:
            nid = get_nid(lon, lat)
            if prev is not None and prev != nid:
                pdata = G.nodes[prev]; ndata = G.nodes[nid]
                dist = haversine_m(pdata["lat"], pdata["lon"],
                                   ndata["lat"], ndata["lon"])
                if dist > 0 and not G.has_edge(prev, nid):
                    G.add_edge(prev, nid, weight=dist)
            prev = nid

    node_arr = np.array(node_coords) if node_coords else np.empty((0, 2))

    # scipy sparse 行列（Dijkstra 高速化用）
    n = len(node_key_to_id)
    if n > 0 and G.number_of_edges() > 0:
        r, c, d = [], [], []
        for u, v, w in G.edges(data="weight"):
            r += [u, v]; c += [v, u]; d += [w, w]
        sp_graph = csr_matrix((d, (r, c)), shape=(n, n))
    else:
        sp_graph = None

    return G, node_arr, node_key_to_id, sp_graph


def compute_road_routes(gps_df: pd.DataFrame,
                         store_visits_df: pd.DataFrame,
                         G, node_arr, node_key_to_id: dict,
                         store_lat: float, store_lon: float,
                         pre_minutes: int = 30,
                         threshold_pct: float = 5.0,
                         sp_graph=None) -> pd.DataFrame:
    """
    GPS 滞留点を道路ネットワークにスナップして最短経路を計算。
    scipy Dijkstra（C実装・全ソース一括）で高速化。
    Returns: DataFrame [lon1,lat1,lon2,lat2,count,pct,approaching]
    """
    from scipy.spatial import KDTree

    if store_visits_df.empty or node_arr.shape[0] == 0:
        return pd.DataFrame()

    n_visitors = store_visits_df["member_id"].nunique()
    tree = KDTree(node_arr)
    id_to_key = {v: k for k, v in node_key_to_id.items()}

    # ── GPS データを member_id で事前グルーピング（ループ内 O(N) スキャン排除）
    gps_by_member = {mid: grp.sort_values("stay_datetime")
                     for mid, grp in gps_df.groupby("member_id")}

    # ── 来店前 pre_minutes の GPS 点列を収集
    member_pts: list = []
    for _, visit in store_visits_df.iterrows():
        t0 = visit.arrival_time - timedelta(minutes=pre_minutes)
        sub = gps_by_member.get(visit.member_id)
        if sub is None:
            continue
        sub = sub[(sub["stay_datetime"] >= t0) &
                  (sub["stay_datetime"] <= visit.arrival_time)]
        if len(sub) < 2:
            continue
        pts = list(zip(sub["lon"].values, sub["lat"].values))
        member_pts.append((visit.member_id, pts))

    if not member_pts:
        return pd.DataFrame()

    # ── 全ユニーク GPS 座標を一括スナップ（KDTree 呼び出し1回）
    unique_pts = list({pt for _, pts in member_pts for pt in pts})
    _, idxs = tree.query(unique_pts)
    snap_map = {pt: int(idx) for pt, idx in zip(unique_pts, idxs)}

    # ── メンバーごとのノードペアを収集
    all_pairs: set = set()
    member_pairs: dict = {}
    for mid, pts in member_pts:
        pairs = []
        for i in range(len(pts) - 1):
            n1, n2 = snap_map[pts[i]], snap_map[pts[i + 1]]
            if n1 != n2:
                pairs.append((n1, n2))
                all_pairs.add((n1, n2))
        if pairs:
            member_pairs[mid] = pairs

    if not all_pairs:
        return pd.DataFrame()

    # ── 最短経路計算（scipy C実装 または networkx フォールバック）
    path_cache: dict = {}

    if sp_graph is not None:
        from scipy.sparse.csgraph import dijkstra as _sp_dijkstra
        unique_sources = list({n1 for n1, _ in all_pairs})
        _, predecessors = _sp_dijkstra(
            sp_graph, directed=False,
            indices=unique_sources,
            return_predecessors=True,
        )
        src_to_row = {src: i for i, src in enumerate(unique_sources)}

        def _reconstruct(src, dst):
            pred = predecessors[src_to_row[src]]
            if pred[dst] < 0:
                return [src, dst]
            path, node, seen = [dst], dst, {dst}
            while node != src:
                p = pred[node]
                if p < 0 or p in seen:
                    return [src, dst]
                seen.add(p); path.append(p); node = p
            return list(reversed(path))

        for n1, n2 in all_pairs:
            path_cache[(n1, n2)] = _reconstruct(n1, n2)
    else:
        import networkx as nx
        for n1, n2 in all_pairs:
            try:
                path_cache[(n1, n2)] = nx.shortest_path(G, n1, n2, weight="weight")
            except Exception:
                path_cache[(n1, n2)] = [n1, n2]

    # ── エッジごとの通過メンバーを集計
    edge_members: dict = {}
    for mid, pairs in member_pairs.items():
        for n1, n2 in pairs:
            path = path_cache.get((n1, n2), [n1, n2])
            for j in range(len(path) - 1):
                key = tuple(sorted([path[j], path[j + 1]]))
                edge_members.setdefault(key, set()).add(mid)

    if not edge_members:
        return pd.DataFrame()

    threshold = max(1, n_visitors * threshold_pct / 100.0)
    rows = []
    for (n1, n2), members in edge_members.items():
        if len(members) < threshold:
            continue
        k1, k2 = id_to_key.get(n1), id_to_key.get(n2)
        if k1 is None or k2 is None:
            continue
        lon1, lat1 = k1
        lon2, lat2 = k2
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2
        brg = bearing_deg(lat1, lon1, lat2, lon2)
        brg_store = bearing_deg(mid_lat, mid_lon, store_lat, store_lon)
        diff = abs(brg - brg_store)
        if diff > 180:
            diff = 360 - diff
        rows.append({
            "lon1": lon1, "lat1": lat1,
            "lon2": lon2, "lat2": lat2,
            "count": len(members),
            "pct":   len(members) / n_visitors * 100,
            "approaching": diff < 90,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def extract_dooh_near_routes(segs_df: pd.DataFrame,
                              all_dooh_df: pd.DataFrame,
                              radius_m: float = 400) -> pd.DataFrame:
    """確定した移動経路セグメントの近傍にある DOOH 設置場所のみを返す。
    各セグメントの中点から radius_m 以内の設置場所に絞り込む。"""
    if segs_df.empty or all_dooh_df.empty:
        return all_dooh_df

    nearby_ids: set = set()
    for _, seg in segs_df.iterrows():
        mid_lat = (seg.lat1 + seg.lat2) / 2
        mid_lon = (seg.lon1 + seg.lon2) / 2
        for _, dooh in all_dooh_df.iterrows():
            if haversine_m(mid_lat, mid_lon, dooh.lat, dooh.lon) <= radius_m:
                nearby_ids.add(dooh["id"])

    return all_dooh_df[all_dooh_df["id"].isin(nearby_ids)].reset_index(drop=True)


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
# OSM 道路ネットワーク取得（Overpass API）
# ─────────────────────────────────────────────────────────────────────────────
_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_osm_roads(center_lat: float, center_lon: float,
                    radius_m: float = 800) -> tuple:
    """Overpass API から道路ジオメトリを取得し (road_segments, status_msg) を返す。
    複数エンドポイントにフォールバックして 504 を回避。
    road_segments: list of [(lon, lat), ...]"""
    dlat = radius_m / 111320 * 1.1
    dlon = radius_m / (111320 * math.cos(math.radians(center_lat))) * 1.1
    s = center_lat - dlat
    n = center_lat + dlat
    w = center_lon - dlon
    e = center_lon + dlon

    # 歩行者動線に必要な highway タイプのみに絞る（データ量削減）
    query = (
        f"[out:json][timeout:25][maxsize:8388608];\n"
        f"(\n"
        f'  way["highway"~"^(primary|secondary|tertiary|residential|'
        f'pedestrian|footway|path|living_street)$"]'
        f"({s:.6f},{w:.6f},{n:.6f},{e:.6f});\n"
        f");\n"
        f"out geom;"
    )
    data = urllib.parse.urlencode({"data": query}).encode()
    last_err = "不明なエラー"
    for endpoint in _OVERPASS_ENDPOINTS:
        req = urllib.request.Request(endpoint, data=data, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=35) as r:
                result = json.loads(r.read())
            road_segs = []
            for way in result.get("elements", []):
                if way.get("type") == "way":
                    pts = [(nd["lon"], nd["lat"])
                           for nd in way.get("geometry", []) if "lon" in nd]
                    if len(pts) >= 2:
                        road_segs.append(pts)
            return road_segs, f"✅ OSM 道路セグメント {len(road_segs):,} 本取得"
        except Exception as e:
            last_err = str(e)
            continue  # 次のエンドポイントを試す

    return [], f"❌ OSM 道路取得エラー（全エンドポイント失敗）: {last_err}"


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
SPEED_LOW_KMH  = 5.0   # ≤ 5 km/h : 低速（歩行・停滞）
SPEED_MID_KMH  = 15.0  # 5‒15 km/h: 中速（早歩き・自転車）
                        # > 15 km/h: 高速（乗り物）
SPEED_COLORS = {
    "低速": "rgba(39,174,96,0.85)",   # 緑: 歩行 ← DOOH 訴求に最適
    "中速": "rgba(230,126,34,0.80)",  # オレンジ: 早歩き
    "高速": "rgba(192,57,43,0.75)",   # 赤: 乗り物
}


def _speed_cat(kmh: float) -> str:
    if kmh <= SPEED_LOW_KMH:
        return "低速"
    elif kmh <= SPEED_MID_KMH:
        return "中速"
    return "高速"


def analyze_pre_arrival_routes(gps_df: pd.DataFrame,
                                store_visits: pd.DataFrame,
                                store_lat: float, store_lon: float,
                                pre_minutes: int = 30,
                                threshold_pct: float = 10.0) -> pd.DataFrame:
    """
    百貨店到着前 pre_minutes 分以内の GPS 点を連結してセグメント化。
    通行人数が threshold_pct % 以上のセグメントを返す。
    Returns: [lat1,lon1,lat2,lon2,count,pct,approaching,bearing,
              低速,中速,高速,dominant_speed]
    """
    if store_visits.empty:
        return pd.DataFrame()

    n_visitors = store_visits["member_id"].nunique()
    segs = []

    # GPS を member_id で事前グルーピング（ループ内 O(N) スキャン排除）
    gps_by_member = {mid: grp.sort_values("stay_datetime")
                     for mid, grp in gps_df.groupby("member_id")}

    for _, visit in store_visits.iterrows():
        t0 = visit.arrival_time - timedelta(minutes=pre_minutes)
        grp = gps_by_member.get(visit.member_id)
        if grp is None:
            continue
        sub = grp[(grp["stay_datetime"] >= t0) &
                  (grp["stay_datetime"] <= visit.arrival_time)]
        if len(sub) < 2:
            continue

        # .values で numpy 配列化（iloc 回避）
        cols = ["lat", "lon", "stay_datetime", "stay_duration_min"]
        vals = sub[cols].values
        for i in range(len(vals) - 1):
            r0_lat, r0_lon, r0_dt, r0_dur = vals[i]
            r1_lat, r1_lon, r1_dt, _      = vals[i + 1]
            la1, lo1 = round(float(r0_lat), 4), round(float(r0_lon), 4)
            la2, lo2 = round(float(r1_lat), 4), round(float(r1_lon), 4)
            if la1 == la2 and lo1 == lo2:
                continue

            # 速度計算: 出発時刻 = 滞在開始 + 滞在時間
            depart = r0_dt + timedelta(minutes=float(r0_dur))
            travel_s = max(30.0, (r1_dt - depart).total_seconds())
            dist_m = haversine_m(la1, lo1, la2, lo2)
            speed_kmh = dist_m / travel_s * 3.6

            brg = bearing_deg(la1, lo1, la2, lo2)
            mid_lat, mid_lon = (la1 + la2) / 2, (lo1 + lo2) / 2
            brg_store = bearing_deg(mid_lat, mid_lon, store_lat, store_lon)
            diff = abs(brg - brg_store)
            if diff > 180:
                diff = 360 - diff
            segs.append({
                "member_id":   visit.member_id,
                "lat1": la1, "lon1": lo1,
                "lat2": la2, "lon2": lo2,
                "bearing":     brg,
                "approaching": diff < 90,
                "speed_cat":   _speed_cat(speed_kmh),
            })

    if not segs:
        return pd.DataFrame()

    df = pd.DataFrame(segs)

    # セグメントごとの総通行人数
    grp = (df.groupby(["lat1","lon1","lat2","lon2","approaching"])["member_id"]
             .nunique().reset_index()
             .rename(columns={"member_id": "count"}))

    threshold = max(1, n_visitors * threshold_pct / 100.0)
    grp = grp[grp["count"] >= threshold].copy()
    if grp.empty:
        return pd.DataFrame()

    # 速度カテゴリ別人数をピボット
    spd = (df.groupby(["lat1","lon1","lat2","lon2","approaching","speed_cat"])["member_id"]
             .nunique().reset_index()
             .rename(columns={"member_id": "spd_n"}))
    pivot = (spd.pivot_table(index=["lat1","lon1","lat2","lon2","approaching"],
                              columns="speed_cat", values="spd_n", fill_value=0)
               .reset_index())
    for col in ["低速", "中速", "高速"]:
        if col not in pivot.columns:
            pivot[col] = 0

    grp = grp.merge(pivot, on=["lat1","lon1","lat2","lon2","approaching"], how="left")
    grp[["低速","中速","高速"]] = grp[["低速","中速","高速"]].fillna(0).astype(int)
    grp["dominant_speed"] = grp[["低速","中速","高速"]].idxmax(axis=1)

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
    ("prev_night_df", None),
    ("dooh_passages_df", None), ("dooh_n_sel", 0),
    ("route_segs_df", None), ("route_dooh_df", None),
    ("_gps_file_key", None),
    ("plateau_road_G", None),
    ("plateau_road_arr", None), ("plateau_road_kid", None),
    ("road_sp_graph", None), ("road_route_df", None),
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
    if uploaded is not None:
        # ファイルが新しい場合のみ処理（リランのたびに再実行しない）
        _fkey = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("_gps_file_key") != _fkey:
            df_tmp = load_gps_csv(uploaded)
            if df_tmp is not None:
                st.session_state.update({
                    "gps_df": df_tmp, "hotels_df": None,
                    "prev_night_df": None, "store_visitors": None,
                    "dooh_passages_df": None, "route_segs_df": None, "route_dooh_df": None,
                    "_gps_file_key": _fkey,
                    "plateau_road_G": None,
                    "plateau_road_arr": None, "plateau_road_kid": None,
                    "road_sp_graph": None, "road_route_df": None,
                })
                st.success(f"✅ {len(df_tmp):,} レコード / {df_tmp['member_id'].nunique():,} メンバー")

# 百貨店位置プレビュー
st.subheader("📍 百貨店位置確認")
fig_loc = go.Figure(go.Scattermapbox(
    lat=[store_lat], lon=[store_lon], mode="markers+text",
    marker=dict(size=18, color="#e74c3c", symbol="star"),
    text=[store_name], textposition="top right",
))
fig_loc.update_layout(
    mapbox=dict(style="open-street-map",
                center=dict(lat=store_lat, lon=store_lon), zoom=15),
    height=300, margin=dict(r=0, t=0, l=0, b=0),
)
st.plotly_chart(fig_loc, use_container_width=True)

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

# GPS 滞留点ヒートマップ
with st.expander("🔥 GPS 滞留点ヒートマップ", expanded=True):
    fig_heat = go.Figure(go.Densitymapbox(
        lat=gps_df["lat"], lon=gps_df["lon"],
        z=gps_df["stay_duration_min"],
        radius=15, opacity=0.7,
        colorscale="YlOrRd",
        hovertemplate="緯度: %{lat:.5f}<br>経度: %{lon:.5f}<extra></extra>",
    ))
    fig_heat.add_trace(go.Scattermapbox(
        lat=[store_lat], lon=[store_lon], mode="markers+text",
        marker=dict(size=16, color="#2980b9", symbol="star"),
        text=[store_name], textposition="top right", name=store_name,
    ))
    fig_heat.update_layout(
        mapbox=dict(style="open-street-map",
                    center=dict(lat=gps_df["lat"].mean(),
                                lon=gps_df["lon"].mean()), zoom=13),
        height=420, margin=dict(r=0, t=0, l=0, b=0),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

if not store_visitors:
    st.warning("百貨店訪問者が検出されませんでした。中心点・半径を調整してください。")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: 前夜ホテル分析
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("② 前夜宿泊ホテル分析")

hotel_search_r = 2000
col_h2, col_h3 = st.columns(2)
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
    st.caption("📺 DOOH 設置場所データ: **Liveboard 東京** より分析実行時に取得（実データ）")

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
        with st.spinner("Liveboard DOOH データ取得中..."):
            _all_dooh = load_dooh_df()
        with st.spinner("DOOH 通過分析中..."):
            passages = compute_dooh_passages(
                gps_df, prev_night_df, _all_dooh,
                selected_hotel_ids=sel_ids,
                passage_r_m=dooh_r,
            )
        st.session_state["dooh_passages_df"] = passages
        st.session_state["dooh_n_sel"] = n_sel
        st.session_state["_dooh_total"] = len(_all_dooh)

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

    _dooh_total = st.session_state.get("_dooh_total", "?")
    st.metric(f"閾値 {dooh_thr}% 超の DOOH 設置場所",
              f"{len(qualifying)} 箇所 / {_dooh_total} 箇所")

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
    _sankey_pn = (prev_night_df[prev_night_df["hotel_id"].isin(sel_ids)]
                  if sel_ids else prev_night_df)
    sankey_fig = build_sankey(
        passages_df, _sankey_pn,
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
        "百貨店到着前の GPS 滞留点を Plateau 道路ネットワーク上にスナップし、"
        "Dijkstra 最短経路で道路沿いの移動経路を推定します。"
        "チラシ・ポスター配布の最適場所を特定します。"
    )

    # ── Plateau 道路データ取得 ──────────────────────────────────────────────
    st.subheader("🏗️ Plateau 道路データ取得")
    plateau_r = st.slider("道路取得範囲 (m)", 300, 2000, 800, step=100,
                           key="plateau_road_r",
                           help="Plateau CityGML から取得する道路ネットワークの範囲")
    fetch_roads_btn = st.button("📥 Plateau 道路データ取得",
                                type="secondary", key="fetch_plateau_btn")
    if fetch_roads_btn:
        with st.spinner("Plateau から道路データを並列取得中...（初回は数十秒）"):
            p_segs, p_msg = fetch_plateau_roads(store_lat, store_lon, plateau_r)
        st.info(p_msg)
        if p_segs:
            with st.spinner("道路ネットワーク構築中..."):
                G, node_arr, node_kid, sp_graph = build_road_network(p_segs)
            st.session_state["plateau_road_G"]   = G
            st.session_state["plateau_road_arr"] = node_arr
            st.session_state["plateau_road_kid"] = node_kid
            st.session_state["road_sp_graph"]    = sp_graph
            st.session_state["road_route_df"]    = None
            c1, c2 = st.columns(2)
            c1.metric("道路ノード数", f"{len(node_kid):,}")
            c2.metric("道路エッジ数", f"{G.number_of_edges():,}")
        else:
            st.session_state["plateau_road_G"] = None

    _road_loaded = st.session_state.get("plateau_road_G") is not None
    if _road_loaded:
        st.success("✅ 道路ネットワーク読込済み")
    else:
        st.info("まず「Plateau 道路データ取得」を実行してください。")

    st.divider()

    # ── 経路分析パラメータ ─────────────────────────────────────────────────
    col_ra, col_rb = st.columns(2)
    with col_ra:
        pre_min     = st.slider("来店前の分析時間窓 (分)", 10, 60, 30, step=5)
    with col_rb:
        route_thr   = st.slider("表示閾値（訪問者の %）", 0.5, 30.0, 5.0, step=0.5)
    dooh_near_r = st.slider("経路沿い DOOH 抽出半径 (m)", 100, 800, 400, step=50,
                             help="確定経路セグメント中点からこの距離内の Liveboard DOOH を抽出")

    run_route = st.button("▶ 経路分析 + DOOH 抽出実行", type="primary",
                          disabled=not _road_loaded)
    if run_route:
        # GPS セグメント分析
        with st.spinner("来店前経路を分析中..."):
            segs = analyze_pre_arrival_routes(
                gps_df, store_visits_df, store_lat, store_lon,
                pre_minutes=pre_min, threshold_pct=route_thr,
            )
        st.session_state["route_segs_df"] = segs
        st.session_state["route_dooh_df"] = None

        # 道路ベース経路計算（scipy Dijkstra）
        _G   = st.session_state.get("plateau_road_G")
        _arr = st.session_state.get("plateau_road_arr")
        _kid = st.session_state.get("plateau_road_kid")
        _sp  = st.session_state.get("road_sp_graph")
        if _G is not None and _arr is not None and _arr.shape[0] > 0 and not segs.empty:
            with st.spinner("道路ネットワーク上で最短経路を計算中..."):
                rdf = compute_road_routes(
                    gps_df, store_visits_df, _G, _arr, _kid,
                    store_lat, store_lon,
                    pre_minutes=pre_min, threshold_pct=route_thr,
                    sp_graph=_sp,
                )
            st.session_state["road_route_df"] = rdf
            st.info(f"道路エッジ {_G.number_of_edges():,} 本 ／ "
                    f"通過エッジ {len(rdf):,} 本（閾値 {route_thr}%）")

        if not segs.empty:
            with st.spinner("Liveboard から経路沿い DOOH を抽出中..."):
                _all_dooh = load_dooh_df()
                _route_dooh = extract_dooh_near_routes(segs, _all_dooh, radius_m=dooh_near_r)
            st.session_state["route_dooh_df"] = _route_dooh

    if st.session_state["route_segs_df"] is None:
        st.info("「経路分析 + DOOH 抽出実行」ボタンを押してください。")
        st.stop()

    segs_df: pd.DataFrame = st.session_state["route_segs_df"]

    if segs_df.empty:
        st.warning("閾値を超える経路セグメントが見つかりませんでした。閾値を下げてください。")
        st.stop()

    n_total = store_visits_df["member_id"].nunique()
    app_segs = segs_df[segs_df["approaching"]]
    n_app    = app_segs["count"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("有効セグメント数",    f"{len(segs_df):,}")
    c2.metric("店舗方向 延べ通行数", f"{n_app:,}")
    c3.metric("分析対象来訪者",      f"{n_total:,}")

    # ── 速度別割合（来店方向セグメントのみ） ─────────────────────────────────
    if not app_segs.empty and all(c in app_segs.columns for c in ["低速","中速","高速"]):
        st.subheader("🚶 通行速度別 割合（店舗方向）")
        st.caption(
            f"低速 ≤ {SPEED_LOW_KMH:.0f} km/h（歩行）｜"
            f"中速 {SPEED_LOW_KMH:.0f}–{SPEED_MID_KMH:.0f} km/h（早歩き・自転車）｜"
            f"高速 > {SPEED_MID_KMH:.0f} km/h（乗り物）"
        )
        total_low  = int(app_segs["低速"].sum())
        total_mid  = int(app_segs["中速"].sum())
        total_high = int(app_segs["高速"].sum())
        total_spd  = total_low + total_mid + total_high or 1

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("🟢 低速（歩行）",
                   f"{total_low:,} 件",
                   f"{total_low/total_spd*100:.1f}%")
        mc2.metric("🟠 中速（早歩き）",
                   f"{total_mid:,} 件",
                   f"{total_mid/total_spd*100:.1f}%")
        mc3.metric("🔴 高速（乗り物）",
                   f"{total_high:,} 件",
                   f"{total_high/total_spd*100:.1f}%")

        # 速度別 円グラフ
        fig_spd = go.Figure(go.Pie(
            labels=["低速（歩行）", "中速（早歩き）", "高速（乗り物）"],
            values=[total_low, total_mid, total_high],
            marker_colors=[SPEED_COLORS["低速"], SPEED_COLORS["中速"], SPEED_COLORS["高速"]],
            hole=0.45,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} 件 (%{percent})<extra></extra>",
        ))
        fig_spd.update_layout(
            height=300, margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_spd, use_container_width=True)

    # ── 道路ベース経路マップ（OSM マップマッチング） ──────────────────────────
    rdf: pd.DataFrame = st.session_state.get("road_route_df")
    if rdf is not None and not rdf.empty:
        st.subheader("🛣️ 来店前経路マップ（OSM 道路ベース）")
        st.caption("🟢 緑: 店舗方向  ｜  🔵 ブルー: 逆方向・横断  ｜  線幅 = 通行者数に比例")
        fig_road = go.Figure()
        max_c = rdf["count"].max() or 1
        # 店舗方向・逆方向それぞれ None 区切りで1トレースにまとめる（高速化）
        for approaching, color, label in [
            (True,  "rgba(39,174,96,0.85)",  "→ 店舗方向"),
            (False, "rgba(52,152,219,0.50)", "← 逆方向"),
        ]:
            grp = rdf[rdf["approaching"] == approaching]
            if grp.empty:
                continue
            lats, lons, hover = [], [], []
            for _, row in grp.iterrows():
                tip = (f"{label}<br>{row['count']} 人 ({row['pct']:.1f}%)"
                       "<extra></extra>")
                lats += [row.lat1, row.lat2, None]
                lons += [row.lon1, row.lon2, None]
                hover += [tip, tip, None]
            # 線幅は通行者数の中央値で代表（バッチ化のトレードオフ）
            med_w = float(grp["count"].median() / max_c * 8)
            w = max(2.0, min(9.0, med_w))
            fig_road.add_trace(go.Scattermapbox(
                lat=lats, lon=lons,
                mode="lines",
                line=dict(color=color, width=w),
                hovertemplate=hover,
                name=label,
                showlegend=True,
            ))
        fig_road.add_trace(go.Scattermapbox(
            lat=[store_lat], lon=[store_lon], mode="markers+text",
            marker=dict(size=20, color="#e74c3c", symbol="star"),
            text=[store_name], textposition="top right", name=store_name,
        ))
        fig_road.update_layout(
            mapbox=dict(style="open-street-map",
                        center=dict(lat=store_lat, lon=store_lon), zoom=16),
            height=580, margin=dict(r=0, t=0, l=0, b=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                        bgcolor="rgba(255,255,255,0.9)"),
        )
        st.plotly_chart(fig_road, use_container_width=True)
        app_road = rdf[rdf["approaching"]]
        st.caption(
            f"店舗方向エッジ: {len(app_road)} 本 ／ 全エッジ: {len(rdf)} 本"
            f"（閾値 {route_thr}% 以上）"
        )
    else:
        st.info("道路データの取得に失敗しました。再度「経路分析 + DOOH 抽出実行」を押してください。")

    # ── 経路沿い DOOH 抽出結果 ────────────────────────────────────────────────
    route_dooh_df: pd.DataFrame = st.session_state.get("route_dooh_df")
    if route_dooh_df is not None and not route_dooh_df.empty:
        st.divider()
        st.subheader("📺 経路沿い DOOH 設置場所（Liveboard 実データ・動的抽出）")
        _all_dooh = load_dooh_df()
        st.caption(
            f"確定した移動経路セグメント近傍の DOOH を自動抽出: "
            f"**{len(route_dooh_df)} 箇所** / 東京全体 {len(_all_dooh)} 箇所"
        )

        # 経路 + 抽出 DOOH の複合マップ
        fig_dooh_rt = go.Figure()
        # 経路セグメント（店舗方向のみ）
        approach_segs = segs_df[segs_df["approaching"]]
        max_cnt = segs_df["count"].max() or 1
        for _, seg in approach_segs.iterrows():
            w = max(1.5, min(8, seg["count"] / max_cnt * 8))
            fig_dooh_rt.add_trace(go.Scattermapbox(
                lat=[seg.lat1, seg.lat2], lon=[seg.lon1, seg.lon2],
                mode="lines", line=dict(color="rgba(230,126,34,0.65)", width=w),
                hoverinfo="skip", showlegend=False,
            ))
        # 抽出した DOOH 設置場所
        fig_dooh_rt.add_trace(go.Scattermapbox(
            lat=route_dooh_df["lat"], lon=route_dooh_df["lon"],
            mode="markers+text",
            marker=dict(size=14, color="#8e44ad", opacity=0.9),
            text=route_dooh_df["name"], textposition="top right",
            customdata=route_dooh_df[["station", "screen_type", "ownership"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "最寄駅: %{customdata[0]}<br>"
                "種別: %{customdata[1]} ｜ %{customdata[2]}"
                "<extra></extra>"
            ),
            name="📺 経路沿い DOOH",
        ))
        # 百貨店
        fig_dooh_rt.add_trace(go.Scattermapbox(
            lat=[store_lat], lon=[store_lon], mode="markers+text",
            marker=dict(size=20, color="#e74c3c", symbol="star"),
            text=[store_name], textposition="top right", name=store_name,
        ))
        fig_dooh_rt.update_layout(
            mapbox=dict(style="open-street-map",
                        center=dict(lat=store_lat, lon=store_lon), zoom=15),
            height=520, margin=dict(r=0, t=0, l=0, b=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                        bgcolor="rgba(255,255,255,0.9)"),
        )
        st.plotly_chart(fig_dooh_rt, use_container_width=True)

        # 抽出 DOOH 一覧テーブル
        with st.expander(f"抽出 DOOH 一覧（{len(route_dooh_df)} 箇所）"):
            disp = route_dooh_df[["name", "station", "screen_type", "ownership", "url"]].copy()
            disp.columns = ["名称", "最寄駅", "種別", "運営", "詳細URL"]
            st.dataframe(disp, use_container_width=True, hide_index=True)
    elif route_dooh_df is not None and route_dooh_df.empty:
        st.info("確定した経路の近傍に Liveboard DOOH 設置場所が見つかりませんでした。抽出半径を広げてください。")

    # ── 配布推奨ポイント ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("💡 チラシ・ポスター配布 推奨ポイント（店舗方向通行 上位 5 箇所）")
    # 道路ベース経路がある場合はそちらの上位セグメントを使用
    _rdf_top = st.session_state.get("road_route_df")
    if _rdf_top is not None and not _rdf_top.empty:
        top5 = _rdf_top[_rdf_top["approaching"]].head(5)
    else:
        top5 = segs_df[segs_df["approaching"]].head(5)

    if top5.empty:
        st.info("店舗方向の有効セグメントが見つかりませんでした。")
    else:
        # テキスト一覧
        for rank, (_, seg) in enumerate(top5.iterrows(), 1):
            mid_lat = (seg.lat1 + seg.lat2) / 2
            mid_lon = (seg.lon1 + seg.lon2) / 2
            st.markdown(
                f"**{rank}位** 座標 ({mid_lat:.5f}, {mid_lon:.5f})"
                f" ― **{seg['count']} 人通行** ({seg['pct']:.1f}%) ｜ "
                f"方位 {seg['bearing']:.0f}° → 店舗方向"
            )

        # 配布ポイント地図
        fig_flyer = go.Figure()
        max_cnt_f = top5["count"].max() or 1
        for rank, (_, seg) in enumerate(top5.iterrows(), 1):
            w = max(2, min(10, seg["count"] / max_cnt_f * 10))
            fig_flyer.add_trace(go.Scattermapbox(
                lat=[seg.lat1, seg.lat2], lon=[seg.lon1, seg.lon2],
                mode="lines", line=dict(color="rgba(230,126,34,0.65)", width=w),
                hoverinfo="skip", showlegend=False,
            ))
        mid_lats = [(seg.lat1 + seg.lat2) / 2 for _, seg in top5.iterrows()]
        mid_lons = [(seg.lon1 + seg.lon2) / 2 for _, seg in top5.iterrows()]
        hover_texts = [
            f"{rank}位: {int(seg['count'])} 人通行 ({seg['pct']:.1f}%)"
            for rank, (_, seg) in enumerate(top5.iterrows(), 1)
        ]
        label_texts = [f"{rank}位" for rank in range(1, len(top5) + 1)]
        fig_flyer.add_trace(go.Scattermapbox(
            lat=mid_lats, lon=mid_lons,
            mode="markers+text",
            marker=dict(size=28, color="#e74c3c", opacity=0.95),
            text=label_texts, textposition="top right",
            hovertext=hover_texts, hoverinfo="text",
            name="📌 配布推奨ポイント",
        ))
        fig_flyer.add_trace(go.Scattermapbox(
            lat=[store_lat], lon=[store_lon], mode="markers+text",
            marker=dict(size=20, color="#e74c3c", symbol="star"),
            text=[store_name], textposition="top right", name=store_name,
        ))
        fig_flyer.update_layout(
            mapbox=dict(style="open-street-map",
                        center=dict(lat=store_lat, lon=store_lon), zoom=16),
            height=480, margin=dict(r=0, t=0, l=0, b=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                        bgcolor="rgba(255,255,255,0.9)"),
        )
        st.plotly_chart(fig_flyer, use_container_width=True)

    # セグメント一覧
    with st.expander("全セグメント一覧（上位 30）"):
        show = segs_df.head(30).copy()
        show["方向"] = show["approaching"].map({True: "→ 店舗方向", False: "← 逆/横断"})
        show["通行率 (%)"] = show["pct"].round(1)
        cols = ["lat1","lon1","lat2","lon2","count","通行率 (%)","方向"]
        if all(c in show.columns for c in ["低速","中速","高速","dominant_speed"]):
            show = show.rename(columns={"dominant_speed": "支配速度"})
            cols += ["低速","中速","高速","支配速度"]
        st.dataframe(show[cols], use_container_width=True, hide_index=True)

