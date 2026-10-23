"""2단계 분할: K-means 로 큰 생활권 3개 -> 각각 NetworkX 로 3개씩 -> 생활권 9개

1단계  K-means (직선거리, 세대수 가중), k = 3
         신월1동을 큰 생활권 3개로 나눈다. 큰 틀의 세대수를 고르게 잡는 역할.
2단계  큰 생활권마다 따로 NetworkX (표준 모듈러시티, NetworkX 내장 Louvain)
         노드  = 그 큰 생활권 안의 건물 (세대수)
         엣지  = 도로 보행거리 150m 이하 건물 쌍 (두 건물이 같은 큰 생활권에 있을 때만)
         선 굵기 = (1 - 보행거리/150) x √(세대_i x 세대_j)
         정확히 3개로 나눈다 -> 3 x 3 = 9개
         결과 고르기: γ 후보마다 seed 20개를 돌려
           "정확히 3개 + 자투리 없음" 결과가 가장 많이 나온 γ 에서 점수(모듈러시티)가 가장 높은 결과
평가   생활권마다 세대 보행거리 합이 가장 작은 건물을 거점으로 두고,
         각 세대가 자기 생활권 거점까지 도로를 따라 걷는 거리를 잰다.

실행:  python hybrid_kmeans_networkx.py      (약 5~10분)
입력:  nodes_sinwol1.xlsx (같은 폴더, 1차 정제 결과)
출력:  hybrid_kmeans_networkx.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import MultiPoint
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "nodes_sinwol1.xlsx"
OUTPUT_PNG = "hybrid_kmeans_networkx.png"

K_BIG = 3                      # 1단계 큰 생활권 수
K_SUB = 3                      # 2단계 큰 생활권 하나당 하위 생활권 수
N_ZONES = K_BIG * K_SUB
SEED = 42
COARSE_GAMMAS = np.geomspace(0.03, 5, 40)   # 2단계 1차 탐색 (넓게, seed 3개)
COARSE_SEEDS = range(3)
FINE_STEPS = 21                              # 2단계 2차 탐색 (3개 근처를 촘촘히, seed 20개)
FINE_SEEDS = range(20)

# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
nodes = pd.read_excel(INPUT, sheet_name="nodes")
edges = pd.read_excel(INPUT, sheet_name="edges")
boundary = pd.read_excel(INPUT, sheet_name="boundary")
XY = nodes[["x", "y"]].to_numpy()
HH = nodes.households.to_numpy().astype(float)
H_TOTAL = int(HH.sum())
N = len(nodes)
FRAGMENT_HH = H_TOTAL / N_ZONES * 0.50        # 평균 생활권 세대수의 50% 미만 = 자투리
idx = {nid: i for i, nid in enumerate(nodes.node_id)}
src = edges.source.map(idx).to_numpy()
dst = edges.target.map(idx).to_numpy()
WEIGHT = (1 - edges.walk_m.to_numpy() / 150) * np.sqrt(HH[src] * HH[dst])

# ===========================================================================
# 1단계: 세대수 가중 K-means, k = 3
# ===========================================================================
km = KMeans(n_clusters=K_BIG, n_init=100, random_state=SEED).fit(XY, sample_weight=HH)
order = np.lexsort((km.cluster_centers_[:, 0], -km.cluster_centers_[:, 1]))     # 북쪽 -> 남쪽
big = np.array([{o: n + 1 for n, o in enumerate(order)}[l] for l in km.labels_])  # 1, 2, 3
for b in range(1, K_BIG + 1):
    print(f"[1단계] 큰 생활권 {b}: 건물 {(big == b).sum()}개, {int(HH[big == b].sum()):,}세대")


# ===========================================================================
# 2단계: 큰 생활권마다 NetworkX 표준 모듈러시티로 3개
# ===========================================================================
def split_region(b: int):
    members = np.where(big == b)[0]
    inside = (big[src] == b) & (big[dst] == b)           # 큰 생활권 안쪽 엣지만
    G = nx.Graph()
    G.add_nodes_from(members.tolist())
    G.add_weighted_edges_from(zip(src[inside].tolist(), dst[inside].tolist(), WEIGHT[inside].tolist()))

    def run(gamma, seed):
        comms = nx.community.louvain_communities(G, weight="weight", resolution=gamma, seed=seed)
        q = nx.community.modularity(G, comms, weight="weight", resolution=gamma)
        return comms, q

    def clean(comms):
        return len(comms) == K_SUB and all(HH[list(c)].sum() >= FRAGMENT_HH for c in comms)

    # 1차 탐색: 생활권 수가 3개를 지나는 γ 구간
    coarse = [(gm, float(np.median([len(run(gm, s)[0]) for s in COARSE_SEEDS]))) for gm in COARSE_GAMMAS]
    first_above = next((gm for gm, k in coarse if k > K_SUB), COARSE_GAMMAS[-1])
    lo = max((gm for gm, k in coarse if k < K_SUB and gm < first_above), default=COARSE_GAMMAS[0])
    hi = min((gm for gm, k in coarse if k > K_SUB and gm > lo), default=COARSE_GAMMAS[-1])

    # 2차 탐색: 그 구간을 21칸, seed 20개씩
    log, runs = [], {}
    for gm in np.linspace(lo, hi, FINE_STEPS):
        res = [(*run(gm, s), s) for s in FINE_SEEDS]
        runs[gm] = res
        log.append({"gamma": gm, "k3": sum(len(r[0]) == K_SUB for r in res),
                    "k3_clean": sum(clean(r[0]) for r in res)})
    log = pd.DataFrame(log)
    # 자투리 없는 결과가 하나도 없으면 NetworkX 1안·2안과 같은 규칙으로:
    # 정확히 3개가 가장 많이 나온 γ 에서 점수가 가장 높은 결과를 쓰고 '자투리 있음'으로 표시
    if log.k3_clean.max() > 0:
        gm = log.sort_values(["k3_clean", "gamma"], ascending=[False, True]).iloc[0].gamma
        comms, q, seed = max((r for r in runs[gm] if clean(r[0])), key=lambda r: r[1])
        status = "자투리 없음"
    elif log.k3.max() > 0:
        gm = log.sort_values(["k3", "gamma"], ascending=[False, True]).iloc[0].gamma
        comms, q, seed = max((r for r in runs[gm] if len(r[0]) == K_SUB), key=lambda r: r[1])
        status = "자투리 있음 (자투리 없는 3개 결과 없음)"
    else:
        raise RuntimeError(f"큰 생활권 {b}: 정확히 3개가 나온 결과가 없음")
    inner = sum(d["weight"] for u, v, d in G.edges(data=True)
                if any(u in c and v in c for c in comms)) / G.size(weight="weight")
    print(f"[2단계] 큰 생활권 {b}: {status}, γ 구간 {lo:.3f}~{hi:.3f}, 선택 γ={gm:.3f}, seed={seed}, "
          f"3개+자투리 없음 {int(log.k3_clean.sum())}/{len(log) * len(FINE_SEEDS)}회, "
          f"선 굵기의 {inner:.0%}가 하위 생활권 안쪽")
    return comms, {"gamma": gm, "seed": seed, "Q": q, "log": log, "inner": inner, "status": status}


zone = np.zeros(N, int)           # 최종 생활권 1~9  (큰 생활권 b 의 하위 s -> (b-1)*3 + s)
sub_info = {}
for b in range(1, K_BIG + 1):
    comms, info = split_region(b)
    cen = np.array([np.average(XY[list(c)], axis=0, weights=HH[list(c)]) for c in comms])
    for rank, ci in enumerate(np.lexsort((cen[:, 0], -cen[:, 1]))):          # 하위도 북쪽 -> 남쪽
        zone[list(comms[ci])] = (b - 1) * K_SUB + rank + 1
    sub_info[b] = info
zone_big = {z: (z - 1) // K_SUB + 1 for z in range(1, N_ZONES + 1)}
zone_name = {z: f"{zone_big[z]}-{(z - 1) % K_SUB + 1}" for z in range(1, N_ZONES + 1)}


# ===========================================================================
# 평가: 모든 건물 쌍 도로 보행거리 -> 생활권 거점(보행거리 합 최소 건물)까지 거리
#   edges 의 150m 이하 보행거리에서 양쪽 '도로까지 수직거리'를 빼 도로 위 이동만 남기고,
#   이어 붙여 최단거리를 구한 뒤 출발·도착 수직거리를 더한다. (kmeans 브랜치와 같은 계산)
# ===========================================================================
snap = nodes.snap_to_road_m.to_numpy()
on_road = np.maximum(edges.walk_m.to_numpy() - snap[src] - snap[dst], 0.01)
D = dijkstra(csr_matrix((on_road, (src, dst)), shape=(N, N)), directed=False) + snap[:, None] + snap[None, :]
np.fill_diagonal(D, 0)
hub = np.array([np.where(zone == z)[0][np.argmin(HH[zone == z] @ D[np.ix_(zone == z, zone == z)])]
                for z in range(1, N_ZONES + 1)])
walk = D[np.arange(N), hub[zone - 1]]

G_all = nx.Graph()
G_all.add_nodes_from(range(N))
G_all.add_edges_from(zip(src.tolist(), dst.tolist()))
zt = pd.DataFrame([{"zone": z, "name": zone_name[z], "big": zone_big[z],
                    "buildings": int((zone == z).sum()), "households": int(HH[zone == z].sum()),
                    "walk_mean": float(np.average(walk[zone == z], weights=HH[zone == z])),
                    "connected": nx.is_connected(G_all.subgraph(np.where(zone == z)[0].tolist()))}
                   for z in range(1, N_ZONES + 1)])
h = zt.households
cov = {d: float(HH[walk <= d].sum() / H_TOTAL) for d in (200, 300, 400)}
wmean = float(np.average(walk, weights=HH))
print(zt.round(1).to_string(index=False))
print(f"세대수 {h.min():,}~{h.max():,} (최대/최소 {h.max() / h.min():.2f}배, 변동계수 {h.std(ddof=0) / h.mean():.1%})")
print(f"거점까지 세대 평균 보행 {wmean:.0f}m, 200m 안 {cov[200]:.1%}, 300m 안 {cov[300]:.1%}, "
      f"400m 안 {cov[400]:.1%}, 가장 먼 세대 {walk.max():.0f}m")

# ===========================================================================
# 그림
# ===========================================================================
for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID = "#1f2328", "#6e7781", "#e6e8eb"
# 큰 생활권 3개 = 색 계열 3개 (파랑 / 주황 / 초록), 하위 3개 = 같은 계열의 진하기
# 하위 생활권 9개 = 서로 뚜렷이 다른 9색 (tab10 에서 회색 제외)
SUB_COLORS = [c for i, c in enumerate(plt.get_cmap("tab10").colors) if i != 7]
ZC = {z: SUB_COLORS[z - 1] for z in range(1, N_ZONES + 1)}
# 큰 생활권 3개 = 1단계 지도에서는 회색 3단계, 2단계 지도에서는 굵은 검은 점선 경계
BIG_COLORS = ["#4d4d4d", "#8c8c8c", "#c4c4c4"]
apt = nodes.is_apartment.to_numpy()


def draw_boundary(ax):
    for _, part in boundary.groupby("part"):
        ax.plot(part.x, part.y, color=INK, linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_axis_off()


def big_outline(ax, lw=2.6):
    for b in range(1, K_BIG + 1):
        hull = MultiPoint([tuple(p) for p in XY[big == b]]).convex_hull
        ax.add_patch(MplPolygon(np.asarray(hull.exterior.coords), closed=True, fill=False,
                                edgecolor=INK, linestyle=(0, (6, 3)), linewidth=lw, zorder=4))
        # 큰 생활권 번호를 신월1동 경계 바깥 왼쪽, 그 큰 생활권 높이에 표시
        cy = np.average(XY[big == b, 1], weights=HH[big == b])
        ax.text(boundary.x.min() - 25, cy, f"큰 생활권 {b}", ha="right", va="center", fontsize=11,
                fontweight="bold", color=INK, zorder=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=INK, linestyle="--"))


def sub_fill(ax):
    """하위 생활권마다 건물을 감싸는 영역을 옅게 칠하고 테두리를 그 색으로 그린다."""
    for z in range(1, N_ZONES + 1):
        pts = XY[zone == z]
        if len(pts) >= 3:
            hull = MultiPoint([tuple(p) for p in pts]).convex_hull
            if hull.geom_type == "Polygon":
                ax.add_patch(MplPolygon(np.asarray(hull.exterior.coords), closed=True, facecolor=ZC[z],
                                        alpha=0.16, edgecolor=ZC[z], linewidth=1.4, zorder=1))


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(21, 10.5), facecolor="white")
gsp = fig.add_gridspec(2, 3, width_ratios=[0.8, 1.1, 1], height_ratios=[1.15, 1], wspace=0.22, hspace=0.32)

# (1) 1단계 지도
ax = fig.add_subplot(gsp[:, 0])
for b in range(1, K_BIG + 1):
    m = big == b
    ax.scatter(XY[m, 0], XY[m, 1], s=4 + HH[m], color=BIG_COLORS[b - 1], edgecolors="white", linewidths=0.2)
    cx, cy = np.average(XY[m], axis=0, weights=HH[m])
    ax.text(cx, cy, f"{b}\n{int(HH[m].sum()):,}세대", ha="center", va="center", fontsize=11, fontweight="bold",
            color=INK, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=INK))
draw_boundary(ax)
ax.set_title("1단계 · K-means (직선거리, 세대수 가중)\n큰 생활권 3개", loc="left", color=INK, fontsize=12)

# (2) 2단계 최종 지도
ax = fig.add_subplot(gsp[:, 1])
sub_fill(ax)
for z in range(1, N_ZONES + 1):
    m = zone == z
    ax.scatter(XY[m, 0], XY[m, 1], s=6 + HH[m] * 1.2, color=ZC[z], edgecolors="white", linewidths=0.3, zorder=3)
ax.scatter(XY[apt, 0], XY[apt, 1], s=6 + HH[apt] * 1.2, facecolors="none", edgecolors=INK, linewidths=0.8, zorder=3)
big_outline(ax)
for z in range(1, N_ZONES + 1):
    hx, hy = XY[hub[z - 1]]
    ax.scatter(hx, hy, s=380, marker="*", color=ZC[z], edgecolors=INK, linewidths=1.1, zorder=6)
    ax.annotate(zone_name[z], (hx, hy), xytext=(9, 7), textcoords="offset points", fontsize=11,
                fontweight="bold", color="white", zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=ZC[z], edgecolor=INK, linewidth=0.8))
draw_boundary(ax)
ax.set_title("2단계 · 큰 생활권마다 NetworkX 3개씩\n(도로 보행거리, 표준 모듈러시티)\n"
             "굵은 검은 점선 = 큰 생활권, 색 = 하위 생활권 9개\n★ = 거점, 검은 테두리 점 = 아파트",
             loc="left", color=INK, fontsize=12)

# (3) 생활권별 세대수
ax = fig.add_subplot(gsp[0, 2])
labels = [f"생활권 {n}" for n in zt.name]
bars = ax.barh(labels, h, color=[ZC[z] for z in zt.zone], height=0.65)
for bb, r in zip(bars, zt.itertuples()):
    ax.text(bb.get_width() + 30, bb.get_y() + bb.get_height() / 2,
            f"{r.households:,}세대 · 건물 {r.buildings}개 · 평균 보행 {r.walk_mean:.0f}m",
            va="center", fontsize=9.5, color=MUTED)
ax.axvline(H_TOTAL / N_ZONES, color=MUTED, linestyle="--", linewidth=1, zorder=0)
ax.invert_yaxis()
ax.set_xlim(0, 2000)
ax.set_xlabel(f"세대수 (점선 = 균등 배분 {H_TOTAL / N_ZONES:,.0f}세대)", color=MUTED)
ax.grid(axis="x", color=GRID)
ax.set_title(f"생활권별 세대수 · {h.min():,}~{h.max():,}세대 (최대/최소 {h.max() / h.min():.2f}배, "
             f"변동계수 {h.std(ddof=0) / h.mean():.0%})", loc="left", color=INK, fontsize=12)
style(ax)

# (4) 거점까지 보행거리
ax = fig.add_subplot(gsp[1, 2])
o = np.argsort(walk)
ax.plot(walk[o], np.cumsum(HH[o]) / H_TOTAL * 100, color="#2f6fb0", linewidth=2)
for d in (200, 300, 400):
    ax.axvline(d, color=GRID, linewidth=1, zorder=0)
    ax.scatter(d, cov[d] * 100, s=25, color="#2f6fb0", zorder=5)
    ax.annotate(f"{d}m 안\n{cov[d] * 100:.1f}%", (d, cov[d] * 100), xytext=(6, -26),
                textcoords="offset points", fontsize=9, color=INK)
ax.set_xlim(0, 700)
ax.set_ylim(0, 102)
ax.set_xlabel("자기 생활권 거점까지 도로 보행거리 (m)", color=MUTED)
ax.set_ylabel("세대 누적 비율 (%)", color=MUTED)
ax.grid(axis="y", color=GRID)
ax.set_title(f"거점까지 걷는 거리 · 세대 평균 {wmean:.0f}m, 가장 먼 세대 {walk.max():.0f}m",
             loc="left", color=INK, fontsize=12)
style(ax)

n_frag = int((h < FRAGMENT_HH).sum())
fig.suptitle(f"2단계 분할 · K-means 큰 생활권 {K_BIG}개 → NetworkX {K_SUB}개씩 = 생활권 {N_ZONES}개 "
             f"(건물 {N:,}개, {H_TOTAL:,}세대, 아파트 포함)\n자투리 기준: 평균의 50%({FRAGMENT_HH:,.0f}세대) 미만 · "
             f"결과: " + ("자투리 없음" if n_frag == 0 else f"자투리 {n_frag}개"), x=0.02, y=1.0, ha="left", va="bottom",
             fontsize=15, color=INK)
fig.savefig(ROOT / OUTPUT_PNG, dpi=135, bbox_inches="tight")
print("saved", OUTPUT_PNG)
