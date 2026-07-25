"""Two-panel benchmark visual: the stability/niche trade-off, and the per-seed bimodality."""
import json, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = [json.loads(l) for l in open("/tmp/allv.jsonl")]
by = collections.defaultdict(list)
for r in rows: by[r["variant"]].append(r)

# headline variants, in a sensible order
order = ["A_vanilla","B_background","E_salient","C_both","D_both_noinv"]
label = {"A_vanilla":"vanilla\n(no spatial)","B_background":"attach=\nbackground",
         "E_salient":"attach=\nsalient","C_both":"attach=both\n(2x2)","D_both_noinv":"2x2\n(no breaker)"}
TAU = 0.7

def mode_frac(rs):
    return np.mean([r["bg_ari_layer"]>0.8 and r["bg_ari_lesion"]<0.15 for r in rs])
def niche_frac(rs):
    v=[r for r in rs if r.get("sp_probe_niche") is not None]
    return np.mean([r["sp_probe_niche"]>TAU for r in v]) if v else np.nan

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))

# ---- panel 1: trade-off ----
mf = [mode_frac(by[v]) for v in order]
nf = [niche_frac(by[v]) for v in order]
x = np.arange(len(order)); w=0.38
b1=ax[0].bar(x-w/2, mf, w, label="background stability\n(clean bg domains)", color="#2c7fb8")
b2=ax[0].bar(x+w/2, [0 if np.isnan(v) else v for v in nf], w,
             label=f"niche recovery\n(spatial probe > {TAU})", color="#d95f0e")
ax[0].set_xticks(x); ax[0].set_xticklabels([label[v] for v in order], fontsize=9)
ax[0].set_ylabel("fraction of 12+ seeds"); ax[0].set_ylim(0,1)
ax[0].set_title("The stability / niche trade-off\n(no variant achieves both)", fontsize=11)
ax[0].legend(fontsize=8, loc="upper right"); ax[0].grid(axis="y", alpha=.3)
for xi,v in zip(x, nf):
    if np.isnan(v): ax[0].text(xi+w/2, 0.02, "n/a", ha="center", fontsize=7, color="gray")

# ---- panel 2: per-seed bimodality ----
show = ["A_vanilla","B_background","C_both"]
col = {"A_vanilla":"#999999","B_background":"#2c7fb8","C_both":"#d95f0e"}
for i,v in enumerate(show):
    y=[r["bg_ari_layer"] for r in by[v]]
    xj=np.random.default_rng(0).normal(i, 0.06, len(y))
    ax[1].scatter(xj, y, s=28, color=col[v], alpha=.75, edgecolors="none")
ax[1].axhspan(0.8, 1.0, color="green", alpha=.07)
ax[1].axhline(0.8, ls="--", color="green", lw=1)
ax[1].text(2.35, 0.9, "clean", color="green", fontsize=8, va="center")
ax[1].axhspan(0.0, 0.55, color="red", alpha=.05)
ax[1].text(2.35, 0.30, "lesion-\ncontaminated", color="#b03030", fontsize=8, va="center")
ax[1].set_xticks(range(len(show)))
ax[1].set_xticklabels(["vanilla","attach=bg","2x2"], fontsize=9)
ax[1].set_ylabel("background ARI vs layer (per seed)"); ax[1].set_ylim(-0.05,1.02)
ax[1].set_title("Outcome is BIMODAL, not noisy\n(each dot = one seed)", fontsize=11)
ax[1].grid(axis="y", alpha=.3)

fig.suptitle("Haruka spatial benchmark — LPC simulation, KMeans+GMM clustering, 12+ seeds/variant",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig("outputs/benchmark_summary.png", dpi=150, bbox_inches="tight")
print("wrote outputs/benchmark_summary.png")

# numeric caption
print("\nvariant           mode_frac  niche_frac(t=0.7)")
for v in order:
    print(f"  {v:16s} {mode_frac(by[v]):.2f}       "
          f"{'n/a' if np.isnan(niche_frac(by[v])) else f'{niche_frac(by[v]):.2f}'}")
