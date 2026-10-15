import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
f="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(f); plt.rcParams["font.family"]="Noto Sans CJK JP"
fig,ax=plt.subplots(figsize=(8.2,4.4),dpi=200); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off")
B="#2E6DB4"; T="#DDEBF7"; G="#F2F2F2"; K="#222222"
def box(x,y,w,h,txt,fc,fs=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",fc=fc,ec=B,lw=1.2))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=K)
def arrow(x1,x2,y): ax.annotate("",xy=(x2,y),xytext=(x1,y),arrowprops=dict(arrowstyle="->",color="#595959",lw=1.4))
ax.text(0.1,5.55,"標準の stem（ConvNeXt）",fontsize=13,fontweight="bold",color=B)
box(0.2,4.2,1.5,0.8,"画像\n224×224",G); arrow(1.75,2.35,4.6)
box(2.4,4.2,3.4,0.8,"4×4 の畳み込み 1回\n（4×4 ずつ切って並べる）",T); arrow(5.85,6.45,4.6)
box(6.5,4.2,1.6,0.8,"本体の層へ",G)
ax.text(0.1,3.15,"ConvStem [Singh+ 2023]",fontsize=13,fontweight="bold",color=B)
box(0.2,1.2,1.5,0.8,"画像\n224×224",G); arrow(1.75,2.05,1.6)
for i,x in enumerate([2.3,4.6]):
    box(x,0.9,2.0,1.4,"3×3 畳み込み\n（1/2 に縮小）\n＋ 正規化 ＋ 活性化",T,fs=11)
    if i<1: arrow(x+2.05,x+2.25,1.6)
arrow(6.65,7.15,1.6); box(7.2,1.2,1.6,0.8,"本体の層へ",G)
ax.text(5.0,0.35,"小さな畳み込みを2段重ねて、少しずつ縮める",ha="center",fontsize=11,color="#595959")
plt.savefig("fig_stem.png",bbox_inches="tight",facecolor="white"); print("ok")
