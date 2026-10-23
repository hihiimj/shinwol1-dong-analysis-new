"""신월1동 생활권 구분 - 1차 데이터 정제

(1) 건축물대장(세대수)과 도로명주소 전자지도(건물 위치)를 주소로 짝지어
    신월1동의 "건물 노드 + 세대수" 표를 만든다.
(2) 전자지도 도로 중심선으로 도로망을 만들고, 건물 쌍마다 "도로를 따라 걷는 거리"를 계산해
    150m 이하인 쌍(= NetworkX 엣지 후보)을 표로 만든다.

실행:  python clean_data.py
입력:  data/raw/HangJeongDong_ver20260701.geojson   행정동 경계
       data/raw/building_register_sinwol.xlsx       건축물대장 표제부 (신월동)
       data/raw/road_address_map_11470.zip          도로명주소 전자지도 (양천구)
출력:  nodes_sinwol1.xlsx
         nodes      건물 노드 (세대수, 위치, 아파트 여부, 도로까지 거리)
         edges      도로 보행거리 150m 이하인 건물 쌍
         boundary   신월1동 경계 꼭짓점
         설명       칼럼 설명
"""
from __future__ import annotations

import re
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import unary_union
from shapely.strtree import STRtree

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUTPUT_XLSX = ROOT / "nodes_sinwol1.xlsx"

ADMDONG_FILE = RAW / "HangJeongDong_ver20260701.geojson"   # 행정동 경계
REGISTER_FILE = RAW / "building_register_sinwol.xlsx"       # 건축물대장 표제부 (신월동)
ADDRESS_MAP_ZIP = RAW / "road_address_map_11470.zip"        # 도로명주소 전자지도 (양천구)

TARGET_ADM_CD2 = "1147056000"   # 서울특별시 양천구 신월1동
SIG_CD = "11470"                # 양천구
LEGAL_EMD_CD = "103"            # 법정동 신월동
CRS_WGS84 = "EPSG:4326"
CRS_UTMK = "EPSG:5179"          # 미터 단위 좌표계 (거리 계산용)

WALK_LIMIT_M = 150              # 도로 보행거리가 이 값 이하인 건물 쌍만 엣지로 남긴다
ROAD_BUFFER_M = 500             # 경계 밖 이 거리까지의 도로도 쓴다 (경계 밖으로 돌아가는 길 포함)

# 건축물대장에서 쓰는 컬럼
REGISTER_COLS = {
    "관리건축물대장PK": "reg_pk",
    "건물명": "bld_name",
    "동명칭": "dong_name",
    "주부속구분코드명": "main_sub",
    "대지구분코드": "lot_type",
    "번": "lot_main",
    "지": "lot_sub",
    "새주소도로코드": "road_code",
    "새주소지상지하코드": "road_ug",
    "새주소본번": "bld_main",
    "새주소부번": "bld_sub",
    "주용도코드명": "main_use",
    "기타용도": "use_detail",
    "세대수(세대)": "n_sedae",
    "가구수(가구)": "n_gagu",
    "호수(호)": "n_ho",
    "연면적(㎡)": "gfa_m2",
}


# ---------------------------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------------------------
def int_str(s: pd.Series) -> pd.Series:
    """'0035', 35.0, '35' 를 모두 '35' 로 맞춘다."""
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).astype(str)


def normalize_dong(s: pd.Series) -> pd.Series:
    """동 이름 표기 통일: '101동'='101', '에이동'='A', '제1동'='1'."""
    alpha = {"에이": "A", "비": "B", "씨": "C", "디": "D"}

    def f(v):
        if not isinstance(v, str) or not v.strip():
            return ""
        v = re.sub(r"\s+", "", v).upper()
        v = re.sub(r"^제", "", v)
        v = re.sub(r"동$", "", v)
        v = alpha.get(v, v)
        return v.lstrip("0") or v

    return s.map(f)


def extract_zip_cp949(zip_path: Path, out_dir: Path) -> Path:
    """한글 파일명이 CP949로 저장된 zip을 푼다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if not info.flag_bits & 0x800:
                try:
                    name = name.encode("cp437").decode("cp949")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
            target = (out_dir / name).resolve()
            if out_dir.resolve() not in target.parents:
                raise ValueError(f"zip 안에 허용되지 않는 경로: {name}")
            if info.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
    return out_dir


# ---------------------------------------------------------------------------
# 1. 신월1동 경계
# ---------------------------------------------------------------------------
def load_boundary() -> gpd.GeoDataFrame:
    adm = gpd.read_file(ADMDONG_FILE)
    b = adm[adm["adm_cd2"].astype(str) == TARGET_ADM_CD2][["adm_nm", "adm_cd2", "geometry"]]
    assert len(b) == 1, "신월1동 경계를 찾지 못함"
    b = b.to_crs(CRS_UTMK)
    b["geometry"] = b.geometry.make_valid()
    return b.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. 건축물대장: 건물별 세대수
# ---------------------------------------------------------------------------
def load_register() -> pd.DataFrame:
    df = pd.read_excel(REGISTER_FILE, dtype=str)[list(REGISTER_COLS)].rename(columns=REGISTER_COLS)
    df = df.drop_duplicates("reg_pk")
    for c in ["n_sedae", "n_gagu", "n_ho", "gfa_m2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # 세대수 규칙
    #  (1) 공동주택은 세대수, 단독/다가구는 가구수에 적힌다. 둘 다 있으면 세대수 우선(중복 방지).
    hh = np.where(df.n_sedae > 0, df.n_sedae, df.n_gagu)
    is_main = df.main_sub.eq("주건축물")
    #  (2) 오피스텔이 세대수 0으로 적혀 있으면 호수를 쓴다.
    officetel = df.use_detail.str.contains("오피스텔", na=False) & is_main & (hh == 0) & (df.n_ho > 0)
    hh = np.where(officetel, df.n_ho, hh)
    #  (3) 단독주택 주건물이 0으로 적혀 있으면 호수, 그것도 없으면 1.
    house0 = df.main_use.eq("단독주택") & is_main & (hh == 0)
    hh = np.where(house0, np.where(df.n_ho > 0, df.n_ho, 1), hh)
    df["households"] = hh.astype(int)

    # 지도와 짝지을 주소 키
    df["rkey"] = (df.road_code.fillna("") + "_" + int_str(df.road_ug) + "_"
                  + int_str(df.bld_main) + "_" + int_str(df.bld_sub))
    df.loc[df.road_code.isna(), "rkey"] = np.nan                  # 도로명주소 없는 건물
    df["jkey"] = int_str(df.lot_type) + "_" + int_str(df.lot_main) + "_" + int_str(df.lot_sub)
    df["dkey"] = normalize_dong(df.dong_name)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. 도로명주소 전자지도: 건물 위치
# ---------------------------------------------------------------------------
def load_buildings(map_dir: Path) -> gpd.GeoDataFrame:
    shp = next(map_dir.rglob("TL_SPBD_BULD.shp"))
    b = gpd.read_file(shp, encoding="cp949").set_crs(CRS_UTMK, allow_override=True)
    b = b[(b.SIG_CD.astype(str) == SIG_CD) & (b.EMD_CD.astype(str) == LEGAL_EMD_CD)].copy()
    b["geometry"] = b.geometry.make_valid()
    b = b[b.geometry.notna() & ~b.geometry.is_empty].drop_duplicates("BD_MGT_SN")
    b["rkey"] = (b.SIG_CD.astype(str) + b.RN_CD.astype(str) + "_" + int_str(b.BULD_SE_CD) + "_"
                 + int_str(b.BULD_MNNM) + "_" + int_str(b.BULD_SLNO))
    b["jkey"] = int_str(b.MNTN_YN) + "_" + int_str(b.LNBR_MNNM) + "_" + int_str(b.LNBR_SLNO)
    b["dkey"] = normalize_dong(b.BULD_NM_DC)
    b["area"] = b.area
    return b.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. 건축물대장 한 줄마다 지도 건물 찾기 (주소로 짝짓기)
# ---------------------------------------------------------------------------
def representative(b: gpd.GeoDataFrame, key: str) -> pd.Series:
    """같은 주소에 건물이 여럿이면 주건물(M) 중 가장 큰 것 하나를 대표로 쓴다."""
    c = b.assign(_m=(b.BUL_DPN_SE == "M").astype(int))
    c = c.sort_values(["_m", "area"], ascending=False).drop_duplicates(key)
    return c.set_index(key)["BD_MGT_SN"]


def match(reg: pd.DataFrame, b: gpd.GeoDataFrame) -> pd.DataFrame:
    reg = reg.copy()
    reg["BD_MGT_SN"] = pd.Series(pd.NA, index=reg.index, dtype="object")
    reg["match"] = "실패"

    # 1순위: 도로명주소 + 동 이름 (아파트 단지의 101동, 102동을 구분)
    bd = b[b.dkey != ""].drop_duplicates(["rkey", "dkey"], keep=False)
    lut = bd.set_index(["rkey", "dkey"])["BD_MGT_SN"]
    hit = pd.Series(lut.reindex(pd.MultiIndex.from_arrays([reg.rkey, reg.dkey])).values, index=reg.index)
    m = hit.notna() & (reg.dkey != "")
    reg.loc[m, "BD_MGT_SN"], reg.loc[m, "match"] = hit[m], "1_도로명+동"

    # 2순위: 도로명주소, 3순위: 지번
    for key, label in [("rkey", "2_도로명"), ("jkey", "3_지번")]:
        todo = reg.BD_MGT_SN.isna()
        hit = reg.loc[todo, key].map(representative(b, key)).dropna()
        reg.loc[hit.index, "BD_MGT_SN"], reg.loc[hit.index, "match"] = hit, label
    return reg


# ---------------------------------------------------------------------------
# 5. 건물 단위로 합치고 신월1동 안만 남겨 노드 만들기
# ---------------------------------------------------------------------------
def make_nodes(reg: pd.DataFrame, b: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    r = reg[reg.BD_MGT_SN.notna()].copy()
    # 대표 용도/이름: 연면적이 가장 큰 주건물 기준
    r["_order"] = np.where(r.main_sub.eq("주건축물"), r.gfa_m2, -1)
    rep = (r.sort_values("_order", ascending=False).drop_duplicates("BD_MGT_SN")
             .set_index("BD_MGT_SN")[["main_use", "bld_name"]])
    # 아파트 여부: 이 건물에 붙은 대장 중 하나라도 기타용도에 '아파트'가 있으면 아파트로 표시
    #  (아파트 단지는 자체 분리수거장이 있어 분석에서 뺄 수도 있으므로 표시만 해 둔다)
    r["_apt"] = r.use_detail.str.contains("아파트", na=False)
    agg = r.groupby("BD_MGT_SN").agg(households=("households", "sum"), is_apartment=("_apt", "any"))
    hh = agg.join(rep)

    nodes = b.merge(hh, left_on="BD_MGT_SN", right_index=True, how="inner")
    nodes = nodes[nodes.households > 0]

    # 건물 폴리곤 안쪽 대표점을 노드 위치로 쓰고, 그 점이 신월1동 안에 있는 건물만 남긴다
    pt = nodes.representative_point()
    inside = pt.within(boundary.geometry.iloc[0])
    nodes, pt = nodes[inside].copy(), pt[inside]
    ll = pt.to_crs(CRS_WGS84)
    nodes["x"], nodes["y"] = pt.x.round(2), pt.y.round(2)
    nodes["lon"], nodes["lat"] = ll.x.round(7), ll.y.round(7)

    nodes = nodes.sort_values(["y", "x"], ascending=[False, True]).reset_index(drop=True)
    nodes.insert(0, "node_id", [f"N{i:05d}" for i in range(len(nodes))])
    return nodes[["node_id", "BD_MGT_SN", "bld_name", "main_use", "is_apartment",
                  "households", "x", "y", "lon", "lat"]]


# ---------------------------------------------------------------------------
# 6. 도로망 만들기 (NetworkX)
# ---------------------------------------------------------------------------
def load_roads(map_dir: Path, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """도로 중심선(TL_SPRD_MANAGE) 중 신월1동 경계 + 500m 안에 걸치는 도로."""
    shp = next(map_dir.rglob("TL_SPRD_MANAGE.shp"))
    rd = gpd.read_file(shp, encoding="cp949").set_crs(CRS_UTMK, allow_override=True)
    rd = rd[rd.intersects(boundary.geometry.iloc[0].buffer(ROAD_BUFFER_M))]
    return rd[rd.geometry.notna() & ~rd.geometry.is_empty].reset_index(drop=True)


def key(c) -> tuple:
    """좌표를 0.1m 단위로 반올림해 교차점 이름으로 쓴다 (같은 점이 미세하게 어긋나는 것 방지)."""
    return (round(c[0], 1), round(c[1], 1))


class RoadNet:
    """도로 중심선을 교차점마다 끊어 만든 보행 그래프.
    노드 = 교차점·꺾이는 점, 엣지 = 그 사이 도로 조각 (가중치 = 길이 m)."""

    def __init__(self, roads: gpd.GeoDataFrame):
        noded = unary_union(list(roads.geometry))            # 선이 만나는 곳마다 자른다
        self.segs = []
        for ln in getattr(noded, "geoms", [noded]):
            cs = list(ln.coords)
            self.segs += [LineString([cs[i], cs[i + 1]]) for i in range(len(cs) - 1) if cs[i] != cs[i + 1]]
        self.G = nx.Graph()
        for sg in self.segs:
            self.G.add_edge(key(sg.coords[0]), key(sg.coords[1]), length=sg.length)
        self.tree = STRtree(self.segs)

    def snap(self, x: float, y: float) -> dict:
        """건물 점을 가장 가까운 도로 조각에 수직으로 붙인다.
        off = 건물에서 도로까지 수직거리, t = 조각 시작점에서 붙은 지점까지 길이."""
        from shapely.geometry import Point
        p = Point(x, y)
        i = int(self.tree.nearest(p))
        sg = self.segs[i]
        t, off = sg.project(p), sg.distance(p)
        return {"seg": i, "t": t, "off": off,
                "ends": [(key(sg.coords[0]), off + t), (key(sg.coords[1]), off + sg.length - t)]}


def walk_pairs(net: RoadNet, nodes: pd.DataFrame, limit: float = WALK_LIMIT_M) -> pd.DataFrame:
    """건물 쌍마다 도로를 따라 걷는 최단거리를 구하고 limit 이하인 쌍만 남긴다.
    거리 = (건물 i → 도로 수직거리) + (도로를 따라 이동) + (도로 → 건물 j 수직거리)
    두 건물이 같은 도로 조각에 붙어 있으면 그 조각을 따라 바로 간다."""
    sn = [net.snap(x, y) for x, y in zip(nodes.x, nodes.y)]
    at: dict = {}                                            # 교차점 -> 그 교차점에 닿는 건물들
    by_seg: dict = {}
    for j, s in enumerate(sn):
        for n, o in s["ends"]:
            at.setdefault(n, []).append((j, o))
        by_seg.setdefault(s["seg"], []).append(j)

    best: dict = {}
    for i, s in enumerate(sn):
        reach: dict = {}
        for n, o in s["ends"]:                               # 조각 양 끝에서 다익스트라
            if o > limit:
                continue
            for m, x in nx.single_source_dijkstra_path_length(net.G, n, cutoff=limit - o,
                                                              weight="length").items():
                if x + o < reach.get(m, np.inf):
                    reach[m] = x + o
        for m, x in reach.items():
            for j, o in at.get(m, []):
                if j > i and x + o <= limit and x + o < best.get((i, j), np.inf):
                    best[(i, j)] = x + o
        for j in by_seg[s["seg"]]:                           # 같은 도로 조각 위의 건물
            if j > i:
                v = abs(sn[j]["t"] - s["t"]) + sn[j]["off"] + s["off"]
                if v <= limit and v < best.get((i, j), np.inf):
                    best[(i, j)] = v

    ids = nodes.node_id.to_numpy()
    xy = nodes[["x", "y"]].to_numpy()
    rows = [(ids[i], ids[j], round(d, 1), round(float(np.hypot(*(xy[i] - xy[j]))), 1))
            for (i, j), d in sorted(best.items())]
    edges = pd.DataFrame(rows, columns=["source", "target", "walk_m", "straight_m"])
    snap_off = pd.Series([s["off"] for s in sn], index=nodes.index)
    return edges, snap_off


# ---------------------------------------------------------------------------
# 7. 저장
# ---------------------------------------------------------------------------
DESCRIPTION = [
    ("nodes", "node_id", "노드 번호 (N00000부터, 북쪽에서 남쪽 순서)"),
    ("nodes", "BD_MGT_SN", "도로명주소 전자지도 건물관리번호"),
    ("nodes", "bld_name", "건축물대장 건물명 (단독주택은 대부분 비어 있음)"),
    ("nodes", "main_use", "건축물대장 주용도"),
    ("nodes", "is_apartment", "기타용도에 '아파트'가 있는 건물이면 True (분석에서 뺄지 정할 때 사용)"),
    ("nodes", "households", "세대수 (노드 가중치)"),
    ("nodes", "x, y", "건물 대표점 좌표, EPSG:5179 (UTM-K, 단위 m)"),
    ("nodes", "lon, lat", "건물 대표점 위경도, EPSG:4326 (지도 표시용)"),
    ("nodes", "snap_to_road_m", "건물 대표점에서 가장 가까운 도로 중심선까지 수직거리 (m)"),
    ("nodes", "walk_degree", "도로 보행거리 150m 안에 있는 다른 건물 수 (엣지 수)"),
    ("edges", "source, target", "연결된 두 건물의 node_id"),
    ("edges", "walk_m", "도로를 따라 걷는 최단거리 (m). 150m 이하만 수록"),
    ("edges", "straight_m", "같은 두 건물의 직선거리 (m, 비교용)"),
    ("boundary", "part, order", "신월1동 경계 폴리곤 조각 번호와 꼭짓점 순서"),
    ("boundary", "x, y / lon, lat", "경계 꼭짓점 좌표 (EPSG:5179 / EPSG:4326)"),
]


def boundary_table(boundary: gpd.GeoDataFrame) -> pd.DataFrame:
    g = boundary.geometry.iloc[0]
    rows = [(pi, vi, round(x, 2), round(y, 2))
            for pi, poly in enumerate(getattr(g, "geoms", [g]))
            for vi, (x, y) in enumerate(poly.exterior.coords)]
    t = pd.DataFrame(rows, columns=["part", "order", "x", "y"])
    ll = gpd.GeoSeries(gpd.points_from_xy(t.x, t.y), crs=CRS_UTMK).to_crs(CRS_WGS84)
    t["lon"], t["lat"] = ll.x.round(7), ll.y.round(7)
    return t


def save_xlsx(nodes: pd.DataFrame, edges: pd.DataFrame, boundary: gpd.GeoDataFrame) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        nodes.to_excel(w, sheet_name="nodes", index=False)
        edges.to_excel(w, sheet_name="edges", index=False)
        boundary_table(boundary).to_excel(w, sheet_name="boundary", index=False)
        pd.DataFrame(DESCRIPTION, columns=["시트", "칼럼", "설명"]).to_excel(w, sheet_name="설명", index=False)
    wb = load_workbook(OUTPUT_XLSX)
    for ws in wb.worksheets:
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="2F6FB0")
            c.alignment = Alignment(horizontal="center")
        ws.freeze_panes = "A2"
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width * 1.3), 70)
    wb.save(OUTPUT_XLSX)


# ---------------------------------------------------------------------------
def main() -> None:
    for p in [ADMDONG_FILE, REGISTER_FILE, ADDRESS_MAP_ZIP]:
        if not p.exists():
            raise FileNotFoundError(f"원본 파일 없음: {p} (README.txt 의 원본 목록 참고)")

    boundary = load_boundary()
    with tempfile.TemporaryDirectory() as tmp:          # zip은 임시 폴더에 풀고 끝나면 지운다
        map_dir = extract_zip_cp949(ADDRESS_MAP_ZIP, Path(tmp))
        bld = load_buildings(map_dir)
        roads = load_roads(map_dir, boundary)
    reg = load_register()
    reg = match(reg, bld)
    nodes = make_nodes(reg, bld, boundary)

    net = RoadNet(roads)
    edges, snap_off = walk_pairs(net, nodes)
    nodes["snap_to_road_m"] = snap_off.round(1)
    deg = pd.concat([edges.source, edges.target]).value_counts()
    nodes["walk_degree"] = nodes.node_id.map(deg).fillna(0).astype(int)

    save_xlsx(nodes, edges, boundary)

    s = reg.groupby("match").agg(대장행수=("reg_pk", "size"), 세대수=("households", "sum"))
    print("[건축물대장 -> 지도 건물 짝짓기 결과 (신월동 전체)]")
    print(s.to_string())
    print(f"짝지은 비율: {(reg.match != '실패').mean():.1%}")
    print(f"\n[신월1동 노드] 건물 {len(nodes):,}개, 총 세대수 {nodes.households.sum():,}")
    apt = nodes.is_apartment
    print(f"  아파트 {apt.sum()}동 {nodes.households[apt].sum():,}세대 / "
          f"아파트 외 {(~apt).sum():,}동 {nodes.households[~apt].sum():,}세대")
    print(f"\n[도로망] 도로 {len(roads):,}개 -> 도로 조각 {net.G.number_of_edges():,}개, "
          f"교차점 {net.G.number_of_nodes():,}개")
    q = nodes.snap_to_road_m.quantile([.5, .9, .99]).round(1).tolist()
    print(f"  건물 -> 도로 수직거리: 중앙값 {q[0]}m, 90% {q[1]}m, 99% {q[2]}m, 최대 {nodes.snap_to_road_m.max()}m")
    print(f"\n[엣지] 도로 보행거리 {WALK_LIMIT_M}m 이하 건물 쌍 {len(edges):,}개")
    print(f"  건물당 연결 수: 중앙값 {nodes.walk_degree.median():.0f}, 최소 {nodes.walk_degree.min()}, "
          f"연결 없는 건물 {(nodes.walk_degree == 0).sum()}개")
    print(f"  보행거리/직선거리 중앙값 {(edges.walk_m / edges.straight_m.clip(lower=1)).median():.2f}배")
    print(f"\n저장: {OUTPUT_XLSX.name}")


if __name__ == "__main__":
    main()
