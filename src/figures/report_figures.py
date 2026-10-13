"""Regenerates every figure for the report and slides from the project data.

Usage (from the repository root): python -m src.figures.report_figures
All signal-analysis settings mirror the project notebooks (25 ms / 10 ms
Hamming framing, energy floor + 10 dB, ZCR 70th percentile, ACF 70-400 Hz,
LPC order 16 via Levinson-Durbin, frame at t = 0.752 s of "Bring socks").
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import librosa
from scipy.signal import lfilter, freqz

from src.config import DATA_ROOT as DATA, RESULTS_ROOT, FIGURES_ROOT as OUT
W2V_PREDS = os.path.join(RESULTS_ROOT, "wav2vec2_sequence_fold1_predictions.npz")
WAV = os.path.join(DATA, "wavs/speakers/2BqVo8kVB2Skwgyb/ac4fca80-447a-11e9-a9a5-5dbec3b8816a.wav")
os.makedirs(OUT, exist_ok=True)

# ---------------- style ----------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
FAMILY = {"Utterance-level (classical)": BLUE, "Sequence (classical)": ORANGE,
          "Sequence CNN (from scratch)": AQUA, "Self-supervised (frozen)": YELLOW}
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5",
                                                    "#1c5cab", "#104281", "#0d366b"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=220)
    plt.close(fig)
    print("wrote", name)


def ygrid(ax):
    ax.yaxis.grid(True); ax.set_axisbelow(True)


# ---------------- shared front end (as in the notebooks) ----------------
y, sr = librosa.load(WAV, sr=None)
FL, HL = int(sr * 0.025), int(sr * 0.010)
frames = librosa.util.frame(y, frame_length=FL, hop_length=HL)
nF = frames.shape[1]
ft = (np.arange(nF) * HL + FL / 2) / sr
t = np.arange(len(y)) / sr

ste_db = 10 * np.log10(np.sum(frames.astype(np.float64) ** 2, axis=0) + 1e-10)
sg = np.sign(frames); sg[sg == 0] = 1
zcr = np.mean(np.abs(np.diff(sg, axis=0)) / 2, axis=0)
e_th = np.percentile(ste_db, 15) + 10
speech = ste_db > e_th
z_th = np.percentile(zcr[speech], 70)
lab = np.full(nF, "S"); lab[speech & (zcr <= z_th)] = "V"; lab[speech & (zcr > z_th)] = "U"
print("V/U/S counts:", {k: int((lab == k).sum()) for k in "VUS"})

# ---------------- Fig: framing & windowing ----------------
i752 = int(np.argmin(np.abs(ft - 0.752)))
raw = frames[:, i752].astype(np.float64); win = np.hamming(FL)
fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.7))
tm = np.arange(FL) / sr * 1000
axs[0].plot(tm, raw, color=BLUE, lw=1.0); axs[0].set_title("(a) Raw 25 ms frame")
axs[1].plot(tm, win, color=INK2, lw=1.2); axs[1].set_title("(b) Hamming window")
axs[2].plot(tm, raw * win, color=BLUE, lw=1.0); axs[2].set_title("(c) Windowed frame")
for a in axs: a.set_xlabel("Time within frame (ms)")
axs[0].set_ylabel("Amplitude")
fig.tight_layout(); save(fig, "fig_framing")

# ---------------- Fig: waveform + STE + ZCR + V/U/S ----------------
fig, axs = plt.subplots(4, 1, figsize=(7.0, 4.0), sharex=True,
                        gridspec_kw={"height_ratios": [1.2, 1, 1, 0.32]})
axs[0].plot(t, y, color=BLUE, lw=0.5); axs[0].set_ylabel("Amplitude")
axs[0].set_title('Utterance "Bring socks" (16 kHz)')
axs[1].plot(ft, ste_db, color=BLUE, lw=1.1); axs[1].axhline(e_th, color=INK2, ls="--", lw=0.8)
axs[1].set_ylabel("STE (dB)"); axs[1].text(0.02, e_th + 2, "energy threshold", ha="left", fontsize=7, color=INK2)
axs[2].plot(ft, zcr, color=ORANGE, lw=1.1); axs[2].axhline(z_th, color=INK2, ls="--", lw=0.8)
axs[2].set_ylabel("ZCR"); axs[2].text(0.66, z_th + 0.01, "ZCR threshold", ha="left", fontsize=7, color=INK2)
cmap = {"V": BLUE, "U": ORANGE, "S": "#d9d8d3"}
for k in range(nF):
    axs[3].axvspan(ft[k] - HL / sr / 2, ft[k] + HL / sr / 2, color=cmap[lab[k]], lw=0)
axs[3].set_yticks([]); axs[3].set_ylabel("V/U/S", rotation=0, ha="right", va="center")
axs[3].set_xlabel("Time (s)")
for sp in ["left"]: axs[3].spines[sp].set_visible(False)
axs[3].legend(handles=[Patch(color=BLUE, label=f"Voiced ({(lab=='V').sum()})"),
                       Patch(color=ORANGE, label=f"Unvoiced ({(lab=='U').sum()})"),
                       Patch(color="#d9d8d3", label=f"Silence ({(lab=='S').sum()})")],
              loc="upper center", bbox_to_anchor=(0.5, -1.6), ncol=3, frameon=False)
fig.align_ylabels(axs); fig.tight_layout(h_pad=0.4); save(fig, "fig_vus")

# ---------------- Fig: pitch ----------------
fmin, fmax = 70, 400
lmin, lmax = int(sr / fmax), int(sr / fmin)
def acf(fr, L):
    fr = fr - fr.mean(); return np.array([np.dot(fr[:len(fr) - k], fr[k:]) for k in range(L + 1)])
f0 = np.full(nF, np.nan); conf = np.full(nF, np.nan)
for k in range(nF):
    if lab[k] != "V": continue
    r = acf(frames[:, k].astype(np.float64), lmax)
    if r[0] <= 0: continue
    seg = r[lmin:lmax + 1]; pk = lmin + int(np.argmax(seg))
    conf[k] = seg.max() / r[0]; f0[k] = sr / pk
rel = conf >= 0.3
print("pitch: estimates", int(np.sum(~np.isnan(f0))), "reliable", int(rel.sum()))
r = acf(raw, lmax); rn = r / r[0]; lags = np.arange(len(r)) / sr * 1000
pk = lmin + int(np.argmax(r[lmin:lmax + 1]))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"width_ratios": [1, 1.25]})
axs[0].axvspan(lmin / sr * 1000, lmax / sr * 1000, color="#f4f3ef", lw=0)
axs[0].plot(lags, rn, color=BLUE, lw=1.1)
axs[0].axvline(pk / sr * 1000, color=ORANGE, ls="--", lw=1)
axs[0].text(7.6, -0.62, f"peak lag: {pk} samples\n$F_0$ = {sr/pk:.1f} Hz", fontsize=7.5, va="top", ha="left", backgroundcolor="white")
axs[0].set_xlabel("Lag (ms)"); axs[0].set_ylabel("Normalised $r[k]$")
axs[0].set_title("(a) Short-time ACF, frame at 0.752 s")
axs[1].scatter(ft[rel], f0[rel], s=9, color=BLUE, label="confidence $\\geq$ 0.3", zorder=3)
lo = (~rel) & ~np.isnan(f0)
axs[1].scatter(ft[lo], f0[lo], s=9, facecolors="none", edgecolors=INK2, lw=0.6, label="confidence < 0.3")
axs[1].set_xlim(0, t[-1]); axs[1].set_ylim(60, 420)
axs[1].set_xlabel("Time (s)"); axs[1].set_ylabel("$F_0$ (Hz)")
axs[1].set_title("(b) Pitch contour (voiced frames)"); axs[1].legend(frameon=False, loc="upper right")
ygrid(axs[1]); fig.tight_layout(); save(fig, "fig_pitch")

# ---------------- Fig: spectrograms ----------------
def spec_db(x, n_win, hop):
    S = np.abs(librosa.stft(x, n_fft=max(512, int(2 ** np.ceil(np.log2(n_win)))), win_length=n_win,
                            hop_length=hop, window="hamming")) ** 2
    return librosa.power_to_db(S, ref=np.max, top_db=80)
fig = plt.figure(figsize=(7.0, 2.5))
gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1, 1], wspace=0.28)
ax0 = fig.add_subplot(gs[0])
S = spec_db(y, FL, HL)
fr = np.linspace(0, sr / 2, S.shape[0]); tt = np.arange(S.shape[1]) * HL / sr
m = ax0.pcolormesh(tt, fr, S, cmap=SEQ, shading="auto", rasterized=True, vmin=-80, vmax=0)
ax0.set_ylim(0, 5000); ax0.set_xlabel("Time (s)"); ax0.set_ylabel("Frequency (Hz)")
ax0.set_title("(a) Spectrogram, 25 ms Hamming")
seg = y[int(0.40 * sr):int(1.05 * sr)]
for j, (wl, ttl) in enumerate([(int(0.040 * sr), "(b) Narrowband, 40 ms"), (int(0.005 * sr), "(c) Wideband, 5 ms")]):
    ax = fig.add_subplot(gs[j + 1])
    Sx = spec_db(seg, wl, int(0.002 * sr))
    fx = np.linspace(0, sr / 2, Sx.shape[0]); tx = 0.40 + np.arange(Sx.shape[1]) * 0.002
    ax.pcolormesh(tx, fx, Sx, cmap=SEQ, shading="auto", rasterized=True, vmin=-80, vmax=0)
    ax.set_ylim(0, 5000); ax.set_xlabel("Time (s)"); ax.set_title(ttl)
    ax.set_yticklabels([])
cb = fig.colorbar(m, ax=fig.axes, fraction=0.025, pad=0.015); cb.set_label("dB"); cb.outline.set_visible(False)
save(fig, "fig_spectrogram")

# ---------------- Fig: LPC ----------------
def lev(rr, p):
    a = np.zeros(p + 1); a[0] = 1.0; e = rr[0]
    for i in range(1, p + 1):
        acc = rr[i] + np.sum(a[1:i] * rr[i - 1:0:-1]); kk = -acc / e
        an = a.copy(); an[1:i] = a[1:i] + kk * a[i - 1:0:-1]; an[i] = kk; a = an; e *= (1 - kk ** 2)
    return a, e
fw = raw * win
rr = np.array([np.dot(fw[:len(fw) - k], fw[k:]) for k in range(17)])
a16, g = lev(rr, 16)
w, h = freqz(np.sqrt(g), a16, worN=1024, fs=sr)
nfft = 1024
X = np.abs(np.fft.rfft(fw, nfft)); fx = np.fft.rfftfreq(nfft, 1 / sr)
env = 20 * np.log10(np.abs(h) + 1e-12); spc = 20 * np.log10(X + 1e-12)
ref = max(env.max(), spc.max())
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"width_ratios": [1.2, 1]})
axs[0].plot(fx, spc - ref, color="#9fb7d6", lw=0.8, label="frame spectrum")
axs[0].plot(w, env - ref, color=ORANGE, lw=1.6, label="LP envelope ($p$ = 16)")
for fq in (305, 656):
    axs[0].axvline(fq, color=INK2, ls=":", lw=0.8)
axs[0].text(320, -8, "305 Hz", fontsize=7); axs[0].text(670, -20, "656 Hz", fontsize=7)
axs[0].set_xlim(0, 5000); axs[0].set_ylim(-100, 5)
axs[0].set_xlabel("Frequency (Hz)"); axs[0].set_ylabel("Magnitude (dB re peak)")
axs[0].set_title("(a) LP spectral envelope, frame at 0.752 s"); axs[0].legend(frameon=False, loc="upper right")
res = lfilter(a16, [1.0], raw)[16:]  # first p samples are filter start-up transient
axs[1].plot(tm[16:], res, color=BLUE, lw=0.8); axs[1].set_xlim(0, 25)
for kx in np.arange(1, 9) * (sr / 307.7):
    if kx < FL: axs[1].axvline(kx / sr * 1000, color=INK2, ls=":", lw=0.6)
axs[1].set_xlabel("Time within frame (ms)"); axs[1].set_ylabel("Residual $e[n]$")
axs[1].set_title("(b) Prediction residual (first $p$ samples excluded)"); fig.tight_layout(); save(fig, "fig_lpc")

# ---------------- Fig: log-mel vs MFCC ----------------
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FL, hop_length=HL, win_length=FL,
                                     window="hamming", center=False, n_mels=40)
logmel = librosa.power_to_db(mel, ref=np.max, top_db=80)
mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=FL, hop_length=HL, win_length=FL,
                          window="hamming", center=False)
tt = np.arange(logmel.shape[1]) * HL / sr
fig, axs = plt.subplots(2, 1, figsize=(7.0, 3.0), sharex=True)
m0 = axs[0].pcolormesh(tt, np.arange(1, 41), logmel, cmap=SEQ, shading="auto", rasterized=True)
axs[0].set_ylabel("Mel band"); axs[0].set_title("(a) 40-band log-mel spectrogram")
c0 = fig.colorbar(m0, ax=axs[0], pad=0.01); c0.set_label("dB"); c0.outline.set_visible(False)
mfz = (mf - mf.mean(axis=1, keepdims=True)) / (mf.std(axis=1, keepdims=True) + 1e-8)
m1 = axs[1].pcolormesh(tt, np.arange(1, 14), mfz, cmap=LinearSegmentedColormap.from_list(
    "div", ["#1c5cab", "#86b6ef", "#f4f3ef", "#f2a07f", "#c2410c"]), shading="auto", rasterized=True,
    vmin=-3, vmax=3)
axs[1].set_ylabel("MFCC index"); axs[1].set_xlabel("Time (s)")
axs[1].set_title("(b) 13 MFCCs (each coefficient standardised for display)")
c1 = fig.colorbar(m1, ax=axs[1], pad=0.01); c1.set_label("z-score"); c1.outline.set_visible(False)
fig.tight_layout(); save(fig, "fig_logmel_mfcc")

# ---------------- Dataset figures ----------------
splits = {s: pd.read_csv(os.path.join(DATA, "data", f"{s}_data.csv")) for s in ["train", "valid", "test"]}
allc = pd.concat([d.assign(split=s) for s, d in splits.items()])
allc["intent"] = allc["action"] + " | " + allc["object"] + " | " + allc["location"]
cnt = allc[allc.split == "train"]["intent"].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(3.5, 4.6))
ax.barh(np.arange(len(cnt)), cnt.values, color=BLUE, height=0.72)
ax.set_yticks(np.arange(len(cnt))); ax.set_yticklabels([s.replace(" | none", "") for s in cnt.index], fontsize=6.3)
ax.set_xlabel("Training utterances"); ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.set_ylim(-0.6, len(cnt) - 0.4)
for i, v in enumerate(cnt.values):
    if i in (0, len(cnt) - 1, len(cnt) - 2): ax.text(v + 20, i, str(v), va="center", fontsize=6.5, color=INK2)
ax.set_title("Intent distribution (training split, 31 classes)")
fig.tight_layout(); save(fig, "fig_intent_dist")

feat = pd.read_csv(os.path.join(DATA, "features_cache.csv"))
fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.9), gridspec_kw={"width_ratios": [1.3, 1, 1]})
axs[0].hist(feat["duration_s"].clip(upper=6), bins=50, color=BLUE)
axs[0].axvline(feat["duration_s"].quantile(0.95), color=ORANGE, ls="--", lw=1)
axs[0].text(feat["duration_s"].quantile(0.95) + 0.1, axs[0].get_ylim()[1] * 0.85, "95th pct.\n3.41 s", fontsize=7)
axs[0].set_xlabel("Duration (s)"); axs[0].set_ylabel("Utterances"); axs[0].set_title("(a) Utterance duration")
for ax, slot, ttl in [(axs[1], "action", "(b) Action slot"), (axs[2], "location", "(c) Location slot")]:
    vc = allc[slot].value_counts()
    ax.barh(np.arange(len(vc)), vc.values, color=BLUE, height=0.7)
    ax.set_yticks(np.arange(len(vc))); ax.set_yticklabels(vc.index, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("Utterances"); ax.set_title(ttl); ax.xaxis.grid(True); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, "fig_dataset_stats")
print("duration 95th pct:", feat["duration_s"].quantile(0.95))
print("slot cardinalities:", {s: allc[s].nunique() for s in ["action", "object", "location"]})
print("speakers per split:", {s: d.speakerId.nunique() for s, d in splits.items()})

# ---------------- Results: cross-validation numbers ----------------
def cv(f, col="held_out_acc"):
    d = pd.read_csv(os.path.join(RESULTS_ROOT, f)); return d[col].values
R = {
    "cnn_mfcc": cv("cv_cnn_mfcc_intent.csv"), "cnn_logmel": cv("cv_cnn_logmel_intent.csv"),
    "mh_cnn": cv("cv_cnn_multihead_mfcc.csv", "full_intent_acc"),
    "w2v_svm": cv("cv_wav2vec2_pooled_intent.csv", "svm_acc"), "w2v_mlp": cv("cv_wav2vec2_pooled_intent.csv", "mlp_acc"),
    "w2v_seq": cv("cv_wav2vec2_sequence_intent.csv"), "hmm": cv("cv_hmm_mfcc_intent.csv", "accuracy"),
}
for k, v in R.items(): print(f"{k}: mean {100*v.mean():.2f} std(ddof=1) {100*v.std(ddof=1):.2f}  folds {np.round(100*v,1)}")
mh = pd.read_csv(os.path.join(RESULTS_ROOT, "cv_cnn_multihead_mfcc.csv"))
for c in ["action_acc", "object_acc", "location_acc", "full_intent_acc"]:
    print(c, f"{100*mh[c].mean():.2f} +- {100*mh[c].std(ddof=1):.2f}")

# Overall comparison, ordered classical -> modern
ut = pd.read_csv(os.path.join(RESULTS_ROOT, "cv_utterance_all_targets.csv"))
def utv(feat, col):
    v = ut[(ut.features == feat) & (ut.target == "intent")].sort_values("fold")[col].values
    return 100 * v.mean(), 100 * v.std(ddof=1)
rows = [
    ("Majority-class baseline", *utv("prosodic", "majority_acc"), None),
    ("Prosodic stats + MLP", *utv("prosodic", "mlp_acc"), "Utterance-level (classical)"),
    ("Prosodic stats + SVM", *utv("prosodic", "svm_acc"), "Utterance-level (classical)"),
    ("+ MFCC stats + MLP", *utv("prosodic_mfcc", "mlp_acc"), "Utterance-level (classical)"),
    ("+ MFCC stats + SVM", *utv("prosodic_mfcc", "svm_acc"), "Utterance-level (classical)"),
    ("Gaussian HMM (MFCC)", 100 * R["hmm"].mean(), 100 * R["hmm"].std(ddof=1), "Sequence (classical)"),
    ("DTW 3-NN (MFCC)$^{\\dagger}$", 11.6, None, "Sequence (classical)"),
    ("1D-CNN (MFCC)", 100 * R["cnn_mfcc"].mean(), 100 * R["cnn_mfcc"].std(ddof=1), "Sequence CNN (from scratch)"),
    ("Multi-head 1D-CNN (MFCC)", 100 * R["mh_cnn"].mean(), 100 * R["mh_cnn"].std(ddof=1), "Sequence CNN (from scratch)"),
    ("1D-CNN (log-mel)", 100 * R["cnn_logmel"].mean(), 100 * R["cnn_logmel"].std(ddof=1), "Sequence CNN (from scratch)"),
    ("wav2vec 2.0 mean-pool + SVM", 100 * R["w2v_svm"].mean(), 100 * R["w2v_svm"].std(ddof=1), "Self-supervised (frozen)"),
    ("wav2vec 2.0 mean-pool + MLP", 100 * R["w2v_mlp"].mean(), 100 * R["w2v_mlp"].std(ddof=1), "Self-supervised (frozen)"),
    ("wav2vec 2.0 sequence + 1D-CNN", 100 * R["w2v_seq"].mean(), 100 * R["w2v_seq"].std(ddof=1), "Self-supervised (frozen)"),
]
fig, ax = plt.subplots(figsize=(7.0, 3.3))
yy = np.arange(len(rows))[::-1]
for yi, (name, v, sd, fam) in zip(yy, rows):
    col = FAMILY.get(fam, "#bdbcb6")
    ax.barh(yi, v, color=col, height=0.66, xerr=sd, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    lbl = f"{v:.1f}" + (f" $\\pm$ {sd:.1f}" if sd is not None else "")
    ax.text(v + (sd or 0) + 1.0, yi, lbl, va="center", fontsize=7.5, color=INK)
ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 105); ax.set_xlabel("31-class intent accuracy (%)")
ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.legend(handles=[Patch(color=c, label=k) for k, c in FAMILY.items()], frameon=False,
          loc="upper right", fontsize=7.5)
fig.tight_layout(); save(fig, "fig_overall")

# Per-fold accuracies (speaker-grouped CV models)
cvrows = [("Gaussian HMM\n(MFCC)", "hmm", ORANGE), ("1D-CNN\n(MFCC)", "cnn_mfcc", AQUA),
          ("Multi-head CNN\n(MFCC)", "mh_cnn", AQUA), ("1D-CNN\n(log-mel)", "cnn_logmel", AQUA),
          ("wav2vec 2.0\npool + SVM", "w2v_svm", YELLOW), ("wav2vec 2.0\npool + MLP", "w2v_mlp", YELLOW),
          ("wav2vec 2.0\nseq. + CNN", "w2v_seq", YELLOW)]
fig, ax = plt.subplots(figsize=(7.0, 2.4))
markers = ["o", "s", "^", "D", "v"]
for i, (name, key, col) in enumerate(cvrows):
    v = 100 * R[key]
    for f in range(5):
        ax.scatter(i + (f - 2) * 0.09, v[f], marker=markers[f], s=22, color=col, edgecolor="white", lw=0.5, zorder=3)
    ax.hlines(v.mean(), i - 0.3, i + 0.3, color=INK, lw=1.2, zorder=4)
ax.set_xticks(range(len(cvrows))); ax.set_xticklabels([r[0] for r in cvrows], fontsize=7.5)
ax.set_ylabel("Held-out accuracy (%)"); ax.set_ylim(0, 100); ygrid(ax)
ax.legend(handles=[plt.Line2D([], [], marker=markers[f], ls="", color=INK2, label=f"Fold {f+1}") for f in range(5)]
          + [plt.Line2D([], [], color=INK, lw=1.2, label="Mean")], ncol=6, frameon=False,
          loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=7.5)
fig.tight_layout(); save(fig, "fig_folds")

# Multi-head per-slot accuracy
fig, ax = plt.subplots(figsize=(3.5, 2.1))
cols = ["action_acc", "object_acc", "location_acc", "full_intent_acc"]
names = ["Action\n(6)", "Object\n(14)", "Location\n(4)", "Full intent\n(31)"]
mv = [100 * mh[c].mean() for c in cols]; sv = [100 * mh[c].std(ddof=1) for c in cols]
ax.bar(range(4), mv, yerr=sv, color=[AQUA] * 3 + ["#0f7a54"], width=0.6,
       error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
for i, (m_, s_) in enumerate(zip(mv, sv)): ax.text(i, m_ + s_ + 1.5, f"{m_:.1f}", ha="center", fontsize=7.5)
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=7.5); ax.set_ylim(0, 105)
ax.set_ylabel("Accuracy (%)"); ygrid(ax); fig.tight_layout(); save(fig, "fig_multihead_slots")

# ---------------- Confusion matrices ----------------
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

def short(s):
    a, o, l = s.split("_")
    parts = [a.replace("change language", "change lang."), o, l]
    return " ".join(p for p in parts if p != "none")

def plot_cm(yt, yp, classes, name, title, size=(3.5, 3.5)):
    cm = confusion_matrix(yt, yp, labels=range(len(classes))).astype(float)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(cmn, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([short(c) for c in classes], rotation=90, fontsize=4.6)
    ax.set_yticklabels([short(c) for c in classes], fontsize=4.6)
    ax.set_xlabel("Predicted intent"); ax.set_ylabel("True intent"); ax.set_title(title, fontsize=8.5)
    for s in ax.spines.values(): s.set_visible(True); s.set_color(GRID)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.set_label("Row-normalised recall", fontsize=7)
    cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
    fig.tight_layout(); save(fig, name)
    return cm

fm = pd.read_csv(os.path.join(DATA, "features_cache_mfcc.csv"))
pros = ["mean_energy_db", "std_energy_db", "mean_zcr", "std_zcr", "voiced_fraction", "mean_f0",
        "std_f0", "mean_f1", "mean_f2", "duration_s"]
mfc = pros + [f"mfcc{i}_{s}" for i in range(1, 14) for s in ("mean", "std")]
le = LabelEncoder().fit(fm["intent"])
for cols_, (C, gm), name, ttl in [(pros, (10, 0.01), "fig_cm_svm_prosodic", "SVM, prosodic statistics (test split)"),
                                  (mfc, (1, 0.01), "fig_cm_svm_mfcc", "SVM, prosodic + MFCC statistics (test split)")]:
    tr = fm[fm.split == "train"].dropna(subset=cols_); te = fm[fm.split == "test"].dropna(subset=cols_)
    sc = StandardScaler().fit(tr[cols_].values)
    clf = SVC(kernel="rbf", C=C, gamma=gm).fit(sc.transform(tr[cols_].values), le.transform(tr["intent"]))
    yp = clf.predict(sc.transform(te[cols_].values)); yt = le.transform(te["intent"])
    print(name, "test acc", accuracy_score(yt, yp))
    cm = plot_cm(yt, yp, le.classes_, name, ttl)
    rec = np.diag(cm) / cm.sum(1)
    order = np.argsort(rec)
    print("  worst:", [(short(le.classes_[i]), round(rec[i], 3)) for i in order[:8]])
    print("  best:", [(short(le.classes_[i]), round(rec[i], 3)) for i in order[-5:]])
    pc = np.bincount(yp, minlength=len(le.classes_)) / len(yp)
    tc = np.bincount(yt, minlength=len(le.classes_)) / len(yt)
    top = np.argsort(pc)[-3:]
    print("  most predicted:", [(short(le.classes_[i]), round(pc[i], 3), round(tc[i], 3)) for i in top])

if W2V_PREDS and os.path.exists(W2V_PREDS):
    d = np.load(W2V_PREDS, allow_pickle=True)
    yt, yp, classes = d["y_true"], d["y_pred"], d["classes"]
    print("w2v fold1 acc", (yt == yp).mean())
    cm = plot_cm(yt, yp, list(classes), "fig_cm_w2v", "wav2vec 2.0 sequence + 1D-CNN (fold 1)")
    off = cm.copy(); np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:10], off.shape))[0]
    print("  top confusions:", [(short(classes[i]), short(classes[j]), int(off[i, j])) for i, j in idx])
    rec = np.diag(cm) / cm.sum(1); order = np.argsort(rec)
    print("  worst recall:", [(short(classes[i]), round(rec[i], 3)) for i in order[:6]])
    spk = d["speaker"]
    print("  per-speaker acc range:", np.round(sorted([(yt[spk == s] == yp[spk == s]).mean() for s in np.unique(spk)]), 3))
    # slot-level errors
    sl = np.array([c.split("_") for c in classes])
    for j, nm in enumerate(["action", "object", "location"]):
        print(f"  {nm} slot acc:", (sl[yt, j] == sl[yp, j]).mean())
