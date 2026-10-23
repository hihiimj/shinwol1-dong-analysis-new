"""2안: 세대수 가중 K-medoids (도로 보행거리) 로 생활권 9개 나누기

중심점 = 실제 건물 하나 (medoid). 그 생활권 세대들이 걷는 거리 합이 가장 작은 건물
거리   = 도로를 따라 걷는 거리 (1차 정제의 150m 이하 보행거리를 이어 붙여 모든 건물 쌍 계산)
목표   = Σ 세대수 × (건물 -> 자기 생활권 중심 건물 보행거리)  를 최소로
         K-means 처럼 거리를 제곱하지 않는다. "세대들이 걷는 거리의 합"을 그대로 줄인다.
계산   = 직접 구현 (scikit-learn 에는 세대수 가중 K-medoids 가 없음)
         (1) 시작 중심 고르기 100번 (k-means++ 방식, seed 42)
         (2) 배정 <-> 중심 다시 고르기를 반복 (Voronoi 반복)
         (3) 가장 좋은 결과에서 중심을 다른 건물로 바꿔 보며 더 줄어들면 바꾸기 (PAM 교환)

실행:  python kmedoids_walk.py
입력:  nodes_sinwol1.xlsx (같은 폴더, 1차 정제 결과)
출력:  kmedoids_walk.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

PLAN = "2안"
PLAN_LABEL = "세대수 가중 K-medoids · 도로 보행거리"
OUTPUT_PNG = "kmedoids_walk.png"

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "nodes_sinwol1.xlsx"
N_ZONES = 9
SEED = 42
N_INIT = 100

# ---------------------------------------------------------------------------
# 1. 데이터
# ---------------------------------------------------------------------------
nodes = pd.read_excel(INPUT, sheet_name="nodes")
edges = pd.read_excel(INPUT, sheet_name="edges")
boundary = pd.read_excel(INPUT, sheet_name="boundary")
XY = nodes[["x", "y"]].to_numpy()
HH = nodes.households.to_numpy().astype(float)
H_TOTAL = int(HH.sum())
N = len(nodes)


# ---------------------------------------------------------------------------
# 2. 모든 건물 쌍의 도로 보행거리 (1안과 같은 계산)
#    edges 시트의 150m 이하 보행거리에서 양쪽 건물의 '도로까지 수직거리'를 빼면
#    도로 위에서 이동한 거리만 남는다. 이것을 이어 붙여 최단거리(다익스트라)를 구하고
#    출발·도착 건물의 수직거리를 다시 더하면 모든 건물 쌍의 도로 보행거리가 된다.
# ---------------------------------------------------------------------------
def walk_matrix() -> np.ndarray:
    idx = {nid: i for i, nid in enumerate(nodes.node_id)}
    s = edges.source.map(idx).to_numpy()
    t = edges.target.map(idx).to_numpy()
    snap = nodes.snap_to_road_m.to_numpy()
    on_road = np.maximum(edges.walk_m.to_numpy() - snap[s] - snap[t], 0.01)
    R = dijkstra(csr_matrix((on_road, (s, t)), shape=(N, N)), directed=False)
    D = R + snap[:, None] + snap[None, :]
    np.fill_diagonal(D, 0)
    return D


D = walk_matrix()

# ---------------------------------------------------------------------------
# 3. 세대수 가중 K-medoids
# ---------------------------------------------------------------------------
def cost(med: np.ndarray) -> float:
    """Σ 세대수 × 가장 가까운 중심 건물까지 보행거리"""
    return float(HH @ D[:, med].min(axis=1))


def seed_medoids(rng) -> np.ndarray:
    """k-means++ 방식: 이미 고른 중심에서 먼(세대가 많은) 건물일수록 다음 중심으로 뽑힐 확률이 큼."""
    med = [rng.choice(N, p=HH / HH.sum())]
    for _ in range(N_ZONES - 1):
        d = D[:, med].min(axis=1)
        p = HH * d
        med.append(rng.choice(N, p=p / p.sum()))
    return np.array(med)


def voronoi(med: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """배정 -> 각 생활권 안에서 세대 보행거리 합이 최소인 건물로 중심 교체, 바뀌지 않을 때까지."""
    for _ in range(max_iter):
        lab = D[:, med].argmin(axis=1)
        new = med.copy()
        for k in range(N_ZONES):
            m = np.where(lab == k)[0]
            new[k] = m[np.argmin(HH[m] @ D[np.ix_(m, m)])]
        if np.array_equal(np.sort(new), np.sort(med)):
            break
        med = new
    return med


def pam_swap(med: np.ndarray) -> np.ndarray:
    """중심 하나를 다른 건물로 바꿨을 때 목표가 줄어들면 가장 많이 줄어드는 쪽으로 바꾼다. 더 못 줄일 때까지."""
    best = cost(med)
    while True:
        improved = False
        for k in range(N_ZONES):
            others = np.delete(med, k)
            d_other = D[:, others].min(axis=1)                      # 나머지 중심까지 거리
            totals = HH @ np.minimum(d_other[:, None], D)           # 후보 건물 j 를 새 중심으로 했을 때 목표
            j = int(np.argmin(totals))
            if totals[j] < best - 1e-6:
                med = med.copy()
                med[k] = j
                best = float(totals[j])
                improved = True
        if not improved:
            return med


rng = np.random.default_rng(SEED)
runs = [voronoi(seed_medoids(rng)) for _ in range(N_INIT)]
med = pam_swap(min(runs, key=cost))
print(f"[{PLAN}] 시작 {N_INIT}번 중 최선 목표 {min(cost(r) for r in runs):,.0f} -> 교환 후 {cost(med):,.0f}")

lab = D[:, med].argmin(axis=1)
order = np.lexsort((XY[med, 0], -XY[med, 1]))               # 번호: 북쪽 -> 남쪽
zone = np.array([{o: n + 1 for n, o in enumerate(order)}[l] for l in lab])
hub = med[order]                                             # 거점 = 중심 건물 (medoid)

# ===========================================================================
# 아래는 1안과 같은 코드 (요약, 그림)
# ===========================================================================
walk_to_hub = D[np.arange(N), hub[zone - 1]]                # 각 건물 -> 자기 생활권 거점 보행거리

rows = []
for z in range(1, N_ZONES + 1):
    m = zone == z
    w = HH[m]
    rows.append({"zone": z, "buildings": int(m.sum()), "households": int(w.sum()),
                 "walk_mean": float(np.average(walk_to_hub[m], weights=w)),
                 "walk_max": float(walk_to_hub[m].max()),
                 "in300": float(w[walk_to_hub[m] <= 300].sum() / w.sum())})
zt = pd.DataFrame(rows)
h = zt.households
cov = {d: float(HH[walk_to_hub <= d].sum() / H_TOTAL) for d in (200, 300, 400)}
wmean = float(np.average(walk_to_hub, weights=HH))
print(f"[{PLAN}] {PLAN_LABEL}")
print(zt.round(1).to_string(index=False))
print(f"세대수 {h.min():,}~{h.max():,} (최대/최소 {h.max() / h.min():.2f}배, 변동계수 {h.std(ddof=0) / h.mean():.1%})")
print(f"거점까지 세대 평균 보행 {wmean:.0f}m, 200m 안 {cov[200]:.1%}, 300m 안 {cov[300]:.1%}, 400m 안 {cov[400]:.1%}")

for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID = "#1f2328", "#6e7781", "#e6e8eb"
COLORS = [c for i, c in enumerate(plt.get_cmap("tab10").colors) if i != 7]
BAR_MAX = 2000                                    # 1안·2안 막대 눈금을 같게 고정
apt = nodes.is_apartment.to_numpy()


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(17, 10), facecolor="white")
gsp = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1.2, 1], wspace=0.1, hspace=0.32)

ax = fig.add_subplot(gsp[:, 0])
for z in range(1, N_ZONES + 1):
    m = zone == z
    ax.scatter(XY[m, 0], XY[m, 1], s=5 + HH[m] * 1.2, color=COLORS[z - 1], edgecolors="white", linewidths=0.3)
ax.scatter(XY[apt, 0], XY[apt, 1], s=5 + HH[apt] * 1.2, facecolors="none", edgecolors=INK, linewidths=0.9)
for z in range(1, N_ZONES + 1):
    hx, hy = XY[hub[z - 1]]
    ax.scatter(hx, hy, s=420, marker="*", color=COLORS[z - 1], edgecolors=INK, linewidths=1.2, zorder=6)
    ax.annotate(str(z), (hx, hy), xytext=(9, 7), textcoords="offset points", fontsize=12,
                fontweight="bold", color=INK, zorder=7,
                bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor=INK, linewidth=0.8))
for _, part in boundary.groupby("part"):
    ax.plot(part.x, part.y, color=INK, linewidth=1.2)
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title("생활권 지도 · 점 크기 = 세대수, 검은 테두리 = 아파트\n★ = 생활권 거점 (세대 보행거리 합이 최소인 중심 건물)",
             loc="left", color=INK, fontsize=12)

ax = fig.add_subplot(gsp[0, 1])
bars = ax.barh([f"생활권 {z}" for z in zt.zone], h, color=COLORS[:N_ZONES], height=0.65)
for b, r in zip(bars, zt.itertuples()):
    n_apt = int((apt & (zone == r.zone)).sum())
    ax.text(b.get_width() + BAR_MAX * 0.015, b.get_y() + b.get_height() / 2,
            f"{r.households:,}세대 · 건물 {r.buildings}개" + (f" (아파트 {n_apt}동)" if n_apt else "")
            + f" · 평균 보행 {r.walk_mean:.0f}m", va="center", fontsize=9.5, color=MUTED)
ax.axvline(H_TOTAL / N_ZONES, color=MUTED, linestyle="--", linewidth=1, zorder=0)
ax.invert_yaxis()
ax.set_xlim(0, BAR_MAX)
ax.set_xlabel(f"세대수 (점선 = 균등 배분 {H_TOTAL / N_ZONES:,.0f}세대)", color=MUTED)
ax.grid(axis="x", color=GRID)
ax.set_title(f"생활권별 세대수 · {h.min():,}~{h.max():,}세대 (최대/최소 {h.max() / h.min():.2f}배, "
             f"변동계수 {h.std(ddof=0) / h.mean():.0%})", loc="left", color=INK, fontsize=12)
style(ax)

ax = fig.add_subplot(gsp[1, 1])
o = np.argsort(walk_to_hub)
ax.plot(walk_to_hub[o], np.cumsum(HH[o]) / H_TOTAL * 100, color="#2f6fb0", linewidth=2)
for d in (200, 300, 400):
    ax.axvline(d, color=GRID, linewidth=1, zorder=0)
    ax.annotate(f"{d}m 안\n{cov[d] * 100:.1f}%", (d, cov[d] * 100), xytext=(6, -26), textcoords="offset points",
                fontsize=9, color=INK)
    ax.scatter(d, cov[d] * 100, s=25, color="#2f6fb0", zorder=5)
ax.set_xlim(0, 700)
ax.set_ylim(0, 102)
ax.set_xlabel("자기 생활권 거점까지 도로 보행거리 (m)", color=MUTED)
ax.set_ylabel("세대 누적 비율 (%)", color=MUTED)
ax.grid(axis="y", color=GRID)
ax.set_title(f"거점까지 걷는 거리 · 세대 평균 {wmean:.0f}m", loc="left", color=INK, fontsize=12)
style(ax)

fig.suptitle(f"{PLAN} · {PLAN_LABEL} (k={N_ZONES}, 아파트 포함)\n"
             f"건물 {N:,}개, {H_TOTAL:,}세대 · 목표: Σ 세대수 × 중심 건물까지 도로 보행거리  최소",
             x=0.02, y=1.0, ha="left", va="bottom", fontsize=14, color=INK)
fig.savefig(ROOT / OUTPUT_PNG, dpi=140, bbox_inches="tight")
print("saved", OUTPUT_PNG)
