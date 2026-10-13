"""Figures and summary numbers for the reduced-class (slot) tasks, the
cross-validated utterance-level classifiers and the DCT ablation.
Usage (from the repository root): python -m src.figures.slot_task_figures
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

from src.config import RESULTS_ROOT, FIGURES_ROOT as OUT
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"), dpi=220)
    plt.close(fig); print("wrote", name)

def ms(v): return 100 * np.mean(v), 100 * np.std(v, ddof=1)

# ---------------- collect results ----------------
ut = pd.read_csv(os.path.join(RESULTS_ROOT, "cv_utterance_all_targets.csv"))
def tier1(feat, clf, target):
    return ut[(ut.features == feat) & (ut.target == target)].sort_values("fold")[f"{clf}_acc"].values
def seq(inp, target):
    prefix = {"logmel": "cnn_logmel", "logmel_dct40": "cnn_logmel_dct40", "logmel_dct13": "cnn_logmel_dct13",
              "wav2vec2": "wav2vec2_sequence"}[inp]
    return pd.read_csv(os.path.join(RESULTS_ROOT, f"cv_{prefix}_{target}.csv")).sort_values("fold")["held_out_acc"].values
mh = pd.read_csv(os.path.join(RESULTS_ROOT, "cv_cnn_multihead_mfcc.csv")).sort_values("fold")

TASKS = [("location", 4), ("action", 6), ("object", 14), ("intent", 31)]
MODELS = [
    ("Majority class", lambda t: ut[(ut.features == "prosodic") & (ut.target == t)].sort_values("fold")["majority_acc"].values),
    ("Prosodic + SVM", lambda t: tier1("prosodic", "svm", t)),
    ("Prosodic + MFCC + SVM", lambda t: tier1("prosodic_mfcc", "svm", t)),
    ("Prosodic + MFCC + MLP", lambda t: tier1("prosodic_mfcc", "mlp", t)),
    ("1D-CNN, log-mel", lambda t: seq("logmel", t)),
    ("wav2vec 2.0 + 1D-CNN", lambda t: seq("wav2vec2", t)),
]
table = {}
print("\n=== Slot-task table (mean +- sd, %) ===")
for name, fn in MODELS:
    row = {}
    for t, k in TASKS:
        v = fn(t); row[t] = ms(v)
    table[name] = row
    print(f"{name:28s}" + "  ".join(f"{t}:{row[t][0]:.1f}+-{row[t][1]:.1f}" for t, _ in TASKS))
print("multi-head CNN (MFCC) slots:", {c: ms(mh[c].values) for c in ["action_acc", "object_acc", "location_acc", "full_intent_acc"]})

# chance / majority per task for reference
for t, k in TASKS:
    print(f"uniform chance {t}: {100/k:.1f}%")

# ---------------- Fig: slot tasks ----------------
fig, ax = plt.subplots(figsize=(7.0, 2.6))
names = [m[0] for m in MODELS]
cols = [BLUE, ORANGE, AQUA, YELLOW]
w = 0.19
for j, (t, k) in enumerate(TASKS):
    means = [table[n][t][0] for n in names]; sds = [table[n][t][1] for n in names]
    x = np.arange(len(names)) + (j - 1.5) * (w + 0.015)
    ax.bar(x, means, width=w, color=cols[j], yerr=sds, error_kw=dict(ecolor=INK2, lw=0.6, capsize=1.5),
           label=f"{t.capitalize()} ({k} classes)")
ax.set_xticks(np.arange(len(names)))
ax.set_xticklabels([n.replace(" + MFCC", "\n+ MFCC").replace(", ", ",\n").replace("wav2vec 2.0 + ", "wav2vec 2.0\n+ ") for n in names], fontsize=7.5)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105); ax.yaxis.grid(True); ax.set_axisbelow(True)
ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
fig.tight_layout(); save(fig, "fig_slot_tasks")

# ---------------- DCT ablation ----------------
abl = [("MFCC, 13 (128 mel bands)", pd.read_csv(os.path.join(RESULTS_ROOT, "cv_cnn_mfcc_intent.csv")).sort_values("fold")["held_out_acc"].values),
       ("DCT of log-mel, first 13", seq("logmel_dct13", "intent")),
       ("DCT of log-mel, all 40", seq("logmel_dct40", "intent")),
       ("Log-mel, 40 bands", seq("logmel", "intent"))]
print("\n=== DCT ablation (intent) ===")
for n, v in abl: print(f"{n:28s} {ms(v)[0]:.1f} +- {ms(v)[1]:.1f}  folds {np.round(100*v,1)}")
def paired(a, b, la, lb):
    d = 100 * (a - b); p = stats.ttest_rel(a, b).pvalue
    print(f"  {la} - {lb}: mean {d.mean():+.1f}, wins {(d>0).sum()}/5, p={p:.4f}")
A = dict(abl)
paired(A["Log-mel, 40 bands"], A["DCT of log-mel, all 40"], "logmel40", "dct40")
paired(A["DCT of log-mel, all 40"], A["DCT of log-mel, first 13"], "dct40", "dct13")
paired(A["DCT of log-mel, first 13"], A["MFCC, 13 (128 mel bands)"], "dct13", "mfcc13")
paired(A["Log-mel, 40 bands"], A["DCT of log-mel, first 13"], "logmel40", "dct13")

fig, ax = plt.subplots(figsize=(3.5, 1.9))
yy = np.arange(len(abl))[::-1]
for yi, (n, v) in zip(yy, abl):
    m, s = ms(v)
    ax.barh(yi, m, xerr=s, color=AQUA, height=0.6, error_kw=dict(ecolor=INK2, lw=0.7, capsize=2))
    ax.text(m + s + 1, yi, f"{m:.1f} $\\pm$ {s:.1f}", va="center", fontsize=7.5)
ax.set_yticks(yy); ax.set_yticklabels([a[0] for a in abl], fontsize=7.5)
ax.set_xlim(0, 100); ax.set_xlabel("31-class intent accuracy (%)"); ax.xaxis.grid(True); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, "fig_dct_ablation")

# ---------------- Tier 1 under CV (intent) and paired tests vs CNN ----------------
print("\n=== Tier 1 under CV, intent ===")
for f in ["prosodic", "prosodic_mfcc"]:
    for c in ["svm", "mlp"]:
        print(f, c, "%.1f +- %.1f" % ms(tier1(f, c, "intent")))
print("majority", "%.1f +- %.1f" % ms(ut[(ut.features=='prosodic')&(ut.target=='intent')]['majority_acc'].values))
cnn = pd.read_csv(os.path.join(RESULTS_ROOT, "cv_cnn_mfcc_intent.csv")).sort_values("fold")["held_out_acc"].values
paired(cnn, tier1("prosodic_mfcc", "svm", "intent"), "cnn_mfcc", "svm_mfcc_stats")
paired(tier1("prosodic_mfcc", "svm", "intent"), tier1("prosodic", "svm", "intent"), "svm_mfcc", "svm_pros")
