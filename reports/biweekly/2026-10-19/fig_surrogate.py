# 勾配だけ滑らか（SAT の softplus β=10）と ReLU の本当の勾配。数値は decision 0062 の決定欄（80% が |x|<0.3、ResNet-18 は 49%）
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
f = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(f); plt.rcParams["font.family"] = "Noto Sans CJK JP"
x = np.linspace(-1, 1, 801)
sig = lambda z: 1 / (1 + np.exp(-z))
fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=200)
ax.axvspan(-0.3, 0.3, color="#DDEBF7", zorder=0)
ax.text(0, 1.13, "MobileNetV4 の ReLU 入力の 80% がこの範囲\n（敵対的訓練した ResNet-18 では 49%）", ha="center", va="bottom", fontsize=14, color="#1F4E79")
ax.plot(x, (x > 0).astype(float), color="#222222", lw=2.6, label="ReLU の本当の勾配（0 か 1）")
ax.plot(x, sig(10 * x), color="#C0392B", lw=2.6, label="SAT の滑らかな勾配（β=10）")
ax.plot(x, sig(50 * x), color="#7F7F7F", lw=1.8, ls="--", label="β=50（ほぼ ReLU と同じ）")
ax.set_xlim(-1, 1); ax.set_ylim(-0.05, 1.42)
ax.set_yticks([0, 0.5, 1]); ax.set_xticks([-1, -0.5, -0.3, 0, 0.3, 0.5, 1])
ax.set_xlabel("ReLU への入力 x", fontsize=15); ax.set_ylabel("勾配（逆方向に流す傾き）", fontsize=15)
ax.tick_params(labelsize=13)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(loc="center right", bbox_to_anchor=(1.02, 0.40), fontsize=13, frameon=False)
fig.tight_layout(); fig.savefig("fig_surrogate.png")
