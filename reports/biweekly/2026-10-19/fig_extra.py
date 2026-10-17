# 補足図2枚。数値は facts.md（Phase 1 の基準、MobileNetV4-S の活性化関数、実機の推論時間）
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
f = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(f); plt.rcParams["font.family"] = "Noto Sans CJK JP"
B, R, K, G = "#2E6DB4", "#C0392B", "#222222", "#7F7F7F"

def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=11); ax.grid(color="#E6E6E6", lw=0.8); ax.set_axisbelow(True)

# 1) Phase 1: 計算量（GMACs）と PGD-10。DeiT 以外は計算量の順に並ぶ
pts = [("MobileNetV4-S", 0.19, 30.71, B), ("EfficientNet-B0", 0.39, 35.93, B), ("ConvNeXt-Atto", 0.55, 37.02, B),
       ("MobileNetV4-M", 0.83, 39.39, B), ("DeiT-Tiny", 1.07, 35.00, G)]
fig, ax = plt.subplots(figsize=(7.6, 1.95), dpi=220)
xs = [p[1] for p in pts[:4]]; ys = [p[2] for p in pts[:4]]
ax.plot(xs, ys, color=B, lw=1.2, ls="--", alpha=0.6)
for name, x, y, c in pts:
    ax.scatter([x], [y], s=46, color=c, zorder=3)
    dx, dy, ha = {"MobileNetV4-M": (-0.03, 0.3, "right"), "DeiT-Tiny": (0.0, -1.9, "center"),
                   "ConvNeXt-Atto": (-0.02, 0.8, "right")}.get(name, (0.025, -1.6, "left"))
    ax.text(x + dx, y + dy, name, fontsize=11, color=c, ha=ha)
ax.set_xlim(0.1, 1.2); ax.set_ylim(29, 41.5)
ax.set_xlabel("推論1枚の計算量（GMACs）", fontsize=12); ax.set_ylabel("PGD-10（%）", fontsize=12)
style(ax); fig.tight_layout(); fig.savefig("fig_compute.png")

# 2) 活性化関数：実機の推論時間と PGD-10（MobileNetV4-S）
pts = [("ReLU（基準）", 0.254, 30.71, K), ("hard-swish", 0.255, 32.62, B), ("SiLU", 0.296, 33.27, R)]
fig, ax = plt.subplots(figsize=(8.3, 2.3), dpi=220)
for name, x, y, c in pts:
    ax.scatter([x], [y], s=52, color=c, zorder=3)
    ax.text(x + 0.0015, y + (0.25 if name != "ReLU（基準）" else -0.9), name, fontsize=11, color=c)
ax.annotate("", xy=(0.2945, 33.2), xytext=(0.2565, 32.68), arrowprops=dict(arrowstyle="->", color=G, lw=1.0))
ax.text(0.275, 31.75, "+0.65pt、17% 遅い", fontsize=10.5, color=G, ha="center")
ax.set_xlim(0.245, 0.31); ax.set_ylim(29.5, 34.5)
ax.set_xlabel("実機の推論時間（ms、Galaxy S24 の NPU）", fontsize=12); ax.set_ylabel("PGD-10（%）", fontsize=12)
style(ax); fig.tight_layout(); fig.savefig("fig_tradeoff.png")
