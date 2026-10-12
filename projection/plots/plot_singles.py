"""Single-block FFN replacement: KL to the original model per layer, Muon vs AdamW (paper figure).

Inputs are the configs.csv files that run_joint_blocks.py --plan sweep writes (rows with kind == "single" and
config == "L<layer>_ffn").

Usage:
  python plots/plot_singles.py --muon out/joint/moonlight_muon/sweep/configs.csv \\
      --adamw out/joint/moonlight_adamw/sweep/configs.csv --out moonlight_single_block_kl
"""
import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--muon", required=True, help="configs.csv of Moonlight (Muon).")
ap.add_argument("--adamw", required=True, help="configs.csv of Moonlight-A (AdamW).")
ap.add_argument("--out", default="moonlight_single_block_kl", help="Output path without extension.")
a = ap.parse_args()
rows = []
for opt, path in (("muon", a.muon), ("adamw", a.adamw)):
    for r in csv.DictReader(open(path)):
        if r["kind"] == "single" and r["config"].endswith("_ffn"):
            rows.append(dict(optimizer=opt, layer=int(r["config"][1:].split("_")[0]), block="ffn", kl=r["kl"]))
C = {"adamw": "#2a78d6", "muon": "#eb6834"}
LBL = {"adamw": "AdamW", "muon": "Muon"}
MK = {"adamw": "o", "muon": "s"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888888", "xtick.color": "#444444", "ytick.color": "#444444"})
fig, ax = plt.subplots(figsize=(3.25, 1.9))
for opt in ["adamw", "muon"]:
    pts = sorted((int(r["layer"]), float(r["kl"])) for r in rows if r["optimizer"] == opt and r["block"] == "ffn")
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C[opt], lw=1.4, marker=MK[opt], ms=3.2,
            markeredgecolor="white", markeredgewidth=0.5, label=LBL[opt])
ax.set_yscale("log")
ax.set_ylabel("KL (nats/token)")
ax.set_xlabel("Layer of the replaced FFN")
ax.set_xticks(range(0, max(r["layer"] for r in rows) + 1, 2))
ax.grid(axis="y", color="#e6e6e6", lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right", ncol=2)
fig.tight_layout()
fig.savefig(f"{a.out}.pdf", bbox_inches="tight")
fig.savefig(f"{a.out}.png", dpi=200, bbox_inches="tight")
