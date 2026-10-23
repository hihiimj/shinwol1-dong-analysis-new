"""1안: 노드(세대수)에 벌점을 직접 줌 (세대수 벌점 모듈러시티, leidenalg CPM)

노드  = 건물 (세대수)
엣지  = 도로 보행거리 150m 이하 건물 쌍 (1차 정제 결과)
선 굵기 = 거리 가중치(선형) x 세대_i x 세대_j      (= 세대 쌍 단위 연결 강도)
          거리 가중치(선형) = 1 - 보행거리/150
점수  = 세대수 벌점 모듈러시티
          Q = Σ 생활권 [ 생활권 안 선 굵기 합 − γ × (생활권 세대수 합)(세대수 합 − 1)/2 ]
          -> 세대수가 벌점에 직접 들어간다 (생활권에 세대가 많이 모일수록 벌점)
계산  = 그래프는 NetworkX 로 만들고, 점수 최대화는 leidenalg (CPMVertexPartition, node_sizes = 세대수)
          NetworkX 내장 Louvain 은 벌점에 세대수를 넣을 수 없어서 leidenalg 를 쓴다
최종 결과 = 정확히 9개 + 자투리 없는 결과 중 점수가 가장 높은 것

실행:  python plan1_node_penalty.py
입력:  nodes_sinwol1.xlsx (같은 폴더, 1st_data_cleaning 브랜치의 정제 결과)
출력:  plan1_node_penalty.png
"""
from __future__ import annotations

from pathlib import Path

import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
PLAN = "1안"
PLAN_LABEL = "노드(세대수)에 벌점을 직접 줌 (세대수 벌점 모듈러시티)"
EDGE_LABEL = "선 굵기 = (1 − 보행거리/150) × 세대×세대"
OUTPUT_PNG = "plan1_node_penalty.png"

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "nodes_sinwol1.xlsx"
N_ZONES = 9
COARSE_GAMMAS = np.geomspace(1e-5, 1e-1, 40)   # 1차 탐색 (넓게, seed 3개)
COARSE_SEEDS = range(3)
FINE_STEPS = 21                            # 2차 탐색 (9개 근처, seed 20개)
FINE_SEEDS = range(20)
FRAGMENT_RATIO = 0.50                      # 세대수가 평균의 50% 미만인 생활권 = 자투리


def distance_weight(walk_m):
    """선형: 0m에서 1, 150m에서 0."""
    return 1 - walk_m / 150


# ---------------------------------------------------------------------------
# 1. 데이터
# ---------------------------------------------------------------------------
nodes = pd.read_excel(INPUT, sheet_name="nodes")
edges = pd.read_excel(INPUT, sheet_name="edges")
boundary = pd.read_excel(INPUT, sheet_name="boundary")
idx = {nid: i for i, nid in enumerate(nodes.node_id)}
XY = nodes[["x", "y"]].to_numpy()
HH = nodes.households.to_numpy().astype(int)
H_TOTAL = int(HH.sum())
FRAGMENT_HH = H_TOTAL / N_ZONES * FRAGMENT_RATIO
src = edges.source.map(idx).to_numpy()
dst = edges.target.map(idx).to_numpy()

# ---------------------------------------------------------------------------
# 2. 그래프 (NetworkX)
# ---------------------------------------------------------------------------
weight = distance_weight(edges.walk_m.to_numpy()) * HH[src] * HH[dst]
G = nx.Graph()
G.add_nodes_from((i, {"households": int(h)}) for i, h in enumerate(HH))
G.add_weighted_edges_from(zip(src.tolist(), dst.tolist(), weight.tolist()), weight="weight")

# leidenalg 로 넘기기 (노드 번호 그대로)
g = ig.Graph(n=G.number_of_nodes(), edges=list(G.edges()))
g.es["weight"] = [G.edges[e]["weight"] for e in G.edges()]
g.vs["households"] = [G.nodes[v]["households"] for v in G.nodes()]


# ---------------------------------------------------------------------------
# 3. 생활권 나누기
# ---------------------------------------------------------------------------
def partition(gamma: float, seed: int):
    p = la.find_partition(g, la.CPMVertexPartition, weights="weight", node_sizes="households",
                          resolution_parameter=float(gamma), n_iterations=-1, seed=seed)
    return np.array(p.membership), p.quality()


# ===========================================================================
# 아래는 1안과 같은 코드 (γ 탐색, 결과 고르기, 요약, 그림)
# ===========================================================================
def n_fragments(mem):
    return int((np.bincount(mem, weights=HH) < FRAGMENT_HH).sum())


coarse = [(gm, float(np.median([len(set(partition(gm, s)[0])) for s in COARSE_SEEDS])))
          for gm in COARSE_GAMMAS]
first_above = next((gm for gm, k in coarse if k > N_ZONES), COARSE_GAMMAS[-1])
lo = max((gm for gm, k in coarse if k < N_ZONES and gm < first_above), default=COARSE_GAMMAS[0])
hi = min((gm for gm, k in coarse if k > N_ZONES and gm > lo), default=COARSE_GAMMAS[-1])

fine_gammas = np.linspace(lo, hi, FINE_STEPS)
log, runs = [], {}
for gm in fine_gammas:
    res = [(*partition(gm, s), s) for s in FINE_SEEDS]
    runs[gm] = res
    k9 = [r for r in res if len(set(r[0])) == N_ZONES]
    log.append({"gamma": gm, "k9": len(k9), "k9_clean": sum(n_fragments(r[0]) == 0 for r in k9)})
log = pd.DataFrame(log)

if log.k9_clean.max() > 0:
    best_gamma = log.sort_values(["k9_clean", "gamma"], ascending=[False, True]).iloc[0].gamma
    cands = [r for r in runs[best_gamma] if len(set(r[0])) == N_ZONES and n_fragments(r[0]) == 0]
    status = "자투리 없는 9개"
elif log.k9.max() > 0:
    best_gamma = log.sort_values(["k9", "gamma"], ascending=[False, True]).iloc[0].gamma
    cands = [r for r in runs[best_gamma] if len(set(r[0])) == N_ZONES]
    status = "자투리 있음 (자투리 없는 9개 결과 없음)"
else:
    raise RuntimeError("정확히 9개가 나온 결과가 없음")
mem, quality, seed = max(cands, key=lambda r: r[1])

ids = np.unique(mem)                                   # 생활권 번호: 세대 가중 중심 북쪽 -> 남쪽
cen = np.array([np.average(XY[mem == c], axis=0, weights=HH[mem == c]) for c in ids])
order = ids[np.lexsort((cen[:, 0], -cen[:, 1]))]
zone = np.array([{o: n + 1 for n, o in enumerate(order)}[m] for m in mem])

rows = []
for z in range(1, N_ZONES + 1):
    m = zone == z
    cx, cy = np.average(XY[m], axis=0, weights=HH[m])
    rows.append({"zone": z, "buildings": int(m.sum()), "households": int(HH[m].sum()),
                 "fragment": bool(HH[m].sum() < FRAGMENT_HH),
                 "connected": nx.is_connected(G.subgraph(np.where(m)[0])), "cx": cx, "cy": cy,
                 "mean_dist_m": float(np.average(np.hypot(XY[m, 0] - cx, XY[m, 1] - cy), weights=HH[m]))})
zt = pd.DataFrame(rows)
h = zt.households
same = zone[src] == zone[dst]
inside_share = weight[same].sum() / weight.sum()
apt = nodes.is_apartment.to_numpy()
apt_zones = sorted(set(zone[apt]))

print(f"[{PLAN}] γ 탐색 구간 {lo:.4g}~{hi:.4g}, 선택 γ={best_gamma:.4g}, seed={seed}, {status}")
print(zt[["zone", "buildings", "households", "fragment", "connected"]].to_string(index=False))
print(f"세대수 {h.min():,}~{h.max():,} (최대/최소 {h.max() / h.min():.2f}배, 변동계수 {h.std(ddof=0) / h.mean():.1%}), "
      f"선 굵기의 {inside_share:.0%}가 생활권 안쪽, 한 덩어리로 이어진 생활권 {zt.connected.sum()}/9")

# ---------------------------------------------------------------------------
# 그림 (1안과 같은 틀, 같은 눈금)
# ---------------------------------------------------------------------------
for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID, ALERT = "#1f2328", "#6e7781", "#e6e8eb", "#cf222e"
COLORS = [c for i, c in enumerate(plt.get_cmap("tab10").colors) if i != 7]
BAR_MAX = 3000                                       # 1안·2안 막대 눈금을 같게 고정


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(17, 10), facecolor="white")
gsp = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1.25, 1], wspace=0.1, hspace=0.32)

ax = fig.add_subplot(gsp[:, 0])
for z in range(1, N_ZONES + 1):
    sel = same & (zone[src] == z)
    ax.add_collection(LineCollection(np.stack([XY[src[sel]], XY[dst[sel]]], axis=1),
                                     colors=[COLORS[z - 1]], linewidths=0.25, alpha=0.06))
for z in range(1, N_ZONES + 1):
    m = zone == z
    ax.scatter(XY[m, 0], XY[m, 1], s=5 + HH[m] * 1.2, color=COLORS[z - 1], edgecolors="white", linewidths=0.3)
ax.scatter(XY[apt, 0], XY[apt, 1], s=5 + HH[apt] * 1.2, facecolors="none", edgecolors=INK, linewidths=0.9)
for r in zt.itertuples():
    col = ALERT if r.fragment else INK
    ax.scatter(r.cx, r.cy, s=260, color="white", edgecolors=col, linewidths=2 if r.fragment else 1.2, zorder=5)
    ax.text(r.cx, r.cy, str(r.zone), ha="center", va="center", fontsize=11, fontweight="bold", color=col, zorder=6)
for _, part in boundary.groupby("part"):
    ax.plot(part.x, part.y, color=INK, linewidth=1.2)
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title("생활권 지도 · 점 크기 = 세대수, 검은 테두리 = 아파트, 숫자 = 생활권 중심\n"
             "옅은 선 = 같은 생활권 안 도로 보행 150m 연결", loc="left", color=INK, fontsize=12)

ax = fig.add_subplot(gsp[0, 1])
cols = [ALERT if f else COLORS[z - 1] for z, f in zip(zt.zone, zt.fragment)]
bars = ax.barh([f"생활권 {z}" for z in zt.zone], h, color=cols, height=0.65)
for b, r in zip(bars, zt.itertuples()):
    n_apt = int((apt & (zone == r.zone)).sum())
    ax.text(b.get_width() + BAR_MAX * 0.015, b.get_y() + b.get_height() / 2,
            f"{r.households:,}세대 · 건물 {r.buildings}개" + (f" (아파트 {n_apt}동)" if n_apt else "")
            + ("  ← 자투리" if r.fragment else ""), va="center", fontsize=10, color=ALERT if r.fragment else MUTED)
ax.axvline(H_TOTAL / N_ZONES, color=MUTED, linestyle="--", linewidth=1, zorder=0)
ax.invert_yaxis()
ax.set_xlim(0, BAR_MAX)
ax.set_xlabel(f"세대수 (점선 = 균등 배분 {H_TOTAL / N_ZONES:,.0f}세대)", color=MUTED)
ax.grid(axis="x", color=GRID)
ax.set_title(f"생활권별 세대수 · {h.min():,}~{h.max():,}세대 (최대/최소 {h.max() / h.min():.2f}배, "
             f"변동계수 {h.std(ddof=0) / h.mean():.0%})", loc="left", color=INK, fontsize=12)
style(ax)

ax = fig.add_subplot(gsp[1, 1])
w = (fine_gammas[1] - fine_gammas[0]) * 0.4
ax.bar(log.gamma - w / 2, log.k9, width=w, color="#afb8c1", label="정확히 9개")
ax.bar(log.gamma + w / 2, log.k9_clean, width=w, color="#2f6fb0", label="정확히 9개 + 자투리 없음")
ax.axvline(best_gamma, color=ALERT, linestyle=":", linewidth=1.2)
ax.annotate(f"선택 γ = {best_gamma:.4g}", (best_gamma, len(FINE_SEEDS) * 0.92), xytext=(6, 0),
            textcoords="offset points", color=ALERT, fontsize=9)
ax.set_ylim(0, len(FINE_SEEDS))
ax.set_xlabel("벌점 세기 γ (클수록 생활권이 잘게 쪼개짐)", color=MUTED)
ax.set_ylabel(f"seed {len(FINE_SEEDS)}개 중 횟수", color=MUTED)
ax.grid(axis="y", color=GRID)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title(f"γ 탐색 · {len(fine_gammas) * len(FINE_SEEDS)}회 중 9개 {int(log.k9.sum())}회, "
             f"자투리 없는 9개 {int(log.k9_clean.sum())}회", loc="left", color=INK, fontsize=12)
style(ax)

fig.suptitle(f"{PLAN} · {PLAN_LABEL}\n{EDGE_LABEL} · 자투리 기준: 평균의 50%({FRAGMENT_HH:,.0f}세대) 미만\n결과: {status}, "
             f"선 굵기의 {inside_share:.0%}가 생활권 안쪽", x=0.02, y=1.0, ha="left", va="bottom", fontsize=14, color=INK)
fig.savefig(ROOT / OUTPUT_PNG, dpi=140, bbox_inches="tight")
print("saved", OUTPUT_PNG)
