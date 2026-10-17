"""Grafici per la presentazione finale (slide 5.4, 5.5, 5.7 e riserve), disegnati per la proiezione.

Le figure di docs/resources/ vengono dai notebook di confronto e hanno piu' pannelli e
etichette piccole: rimpicciolite in una slide diventano illeggibili. Qui ogni grafico porta un
solo messaggio, a caratteri grandi, calcolato dagli aggregati committati in results/metrics/
(nessuna rigenerazione, nessun costo):

  slide_rouge1_f1[_budgetref].png        ROUGE-1 F1 per metodo, test o a lunghezza contenuta
  slide_geval_dimensioni[_budgetref].png G-Eval per dimensione e media (heatmap), idem
  slide_prima_dopo.png                   ROUGE-1 F1 e G-Eval, test contro test_budgetref

I risultati principali della presentazione sono quelli a lunghezza contenuta (test_budgetref);
le figure su test vanno nelle slide di riserva («a lunghezza libera»).

Uso:  python scripts/figure_presentazione.py   (scrive in docs/resources/slides/)
Colori: tre famiglie dai primi tre slot della palette categoriale di riferimento,
validati anche a coppie arbitrarie (adatti a barre e punti mescolati).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "resources" / "slides"
OUT.mkdir(parents=True, exist_ok=True)
M = REPO / "results" / "metrics"

NOMI = {
    "firstk_psr": "First-k (PSR)", "firstk_nltk": "First-k (NLTK)", "textrank": "TextRank", "lexrank": "LexRank",
    "centroid_mmr": "Centroid+MMR (TF-IDF)", "centroid_mmr_bert": "Centroid+MMR (BERT)",
    "lsa": "LSA (Gong & Liu)", "lsa_steinberger": "LSA (Steinberger)", "sbert_kmeans": "SBERT K-Means",
    "sbert_agglom": "SBERT agglomerativo", "lda": "LDA",
    "bart": "BART", "pegasus": "PEGASUS", "primera": "PRIMERA",
    "qwen": "Qwen2.5-7B", "gemma": "Gemma 4", "mistral": "Mistral-7B", "gpt5mini": "GPT-5-mini",
    "qwen_fewshot": "Qwen few-shot",
}
FAM = {**{k: "estr" for k in ["firstk_psr", "firstk_nltk", "textrank", "lexrank", "centroid_mmr", "centroid_mmr_bert",
                              "lsa", "lsa_steinberger", "sbert_kmeans", "sbert_agglom", "lda"]},
       **{k: "astr" for k in ["bart", "pegasus", "primera"]},
       **{k: "llm" for k in ["qwen", "gemma", "mistral", "gpt5mini", "qwen_fewshot"]}}
COL = {"estr": "#2a78d6", "astr": "#eb6834", "llm": "#1baf7a"}
ETI = {"estr": "Estrattivi", "astr": "Astrattivi specializzati", "llm": "LLM"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"

plt.rcParams.update({"font.family": "Arial", "font.size": 12, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK, "axes.edgecolor": GRID, "svg.fonttype": "none"})

def leggi(m, scope, chiave):
    suf = "_geval" if chiave.startswith("geval") else ""
    return json.load(open(M / f"{m}_{scope}{suf}_aggregate.json", encoding="utf-8"))["overall"][chiave]

def fmt(v, nd): return f"{v:.{nd}f}"
W, H = 8.25, 3.75  # pollici: area utile della slide sotto il titolo

def assi_puliti(ax):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)

def legenda_famiglie(fig, y=0.995):
    h = [Patch(facecolor=COL[f], label=ETI[f]) for f in ("estr", "astr", "llm")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.6, y),
               fontsize=12, handlelength=1.2, columnspacing=1.6)

SOLO_TETTO = ("bart", "pegasus", "primera")  # nell'ambito a budget: non rigenerati, solo tetto

def etichetta(m, budget):
    return NOMI[m] + ("*" if budget and m in SOLO_TETTO else "")

# ---------- 5.2: ROUGE-1 F1 per metodo ----------
# Alta quanto lo spazio fra titolo e piè di pagina (la slide non ha riga dei link): le 19
# righe hanno ~14 pt di passo. Niente asse x: i valori sono scritti sulle barre, e metrica e
# ambito stanno nel titolo della slide. Si disegna per `test` (lunghezza libera, riserva) e per
# `test_budgetref` (lunghezza contenuta, risultati principali).
H52 = 4.25

def grafico_rouge(scope, nome):
    budget = scope.endswith("budgetref")
    v = {m: leggi(m, scope, "rouge1_f1") for m in NOMI}
    ordine = sorted(v, key=v.get)  # dal basso verso l'alto: il migliore in cima
    fig, ax = plt.subplots(figsize=(W, H52), dpi=220)
    fig.subplots_adjust(left=0.23, right=0.97, top=0.93, bottom=0.02)
    y = range(len(ordine))
    ax.barh(y, [v[m] for m in ordine], height=0.62, color=[COL[FAM[m]] for m in ordine], edgecolor="white", linewidth=1)
    for i, m in enumerate(ordine):
        ax.text(v[m] + 0.004, i, fmt(v[m], 3), va="center", fontsize=11, color=INK2)
    ax.set_yticks(list(y), [etichetta(m, budget) for m in ordine], fontsize=11)
    ax.set_xlim(0, 0.5); ax.set_ylim(-0.6, len(ordine) - 0.4)
    ax.set_xticks([])
    assi_puliti(ax); ax.spines["bottom"].set_visible(False)
    h = [Patch(facecolor=COL[f], label=ETI[f]) for f in ("estr", "astr", "llm")]
    if budget:
        h.append(Line2D([], [], ls="", label="* solo tetto"))
    fig.legend(handles=h, loc="upper right", ncol=len(h), frameon=False, bbox_to_anchor=(0.97, 1.0),
               fontsize=11, handlelength=1.2, columnspacing=1.4)
    fig.savefig(OUT / nome, facecolor="white"); plt.close(fig)

# ---------- BERTScore F1 per metodo (punti) ----------
# I 19 valori stanno fra ~0.83 e ~0.88: con barre ancorate a zero sembrerebbero uguali, e un
# asse tagliato su un grafico a barre esagera le differenze. Si usano punti su un asse ristretto,
# dichiarato con tacche ed etichette, che è la forma corretta quando lo zero non è informativo.
def grafico_bertscore(scope, nome):
    budget = scope.endswith("budgetref")
    v = {m: leggi(m, scope, "bertscore_f1") for m in NOMI}
    ordine = sorted(v, key=v.get)
    fig, ax = plt.subplots(figsize=(W, H52), dpi=220)
    fig.subplots_adjust(left=0.23, right=0.97, top=0.93, bottom=0.075)
    for i, m in enumerate(ordine):
        ax.axhline(i, color=GRID, lw=0.6, zorder=1)
        ax.scatter(v[m], i, s=70, color=COL[FAM[m]], edgecolor="white", linewidth=1, zorder=3)
        ax.text(v[m] + 0.0012, i, fmt(v[m], 3), va="center", fontsize=11, color=INK2)
    ax.set_yticks(range(len(ordine)), [etichetta(m, budget) for m in ordine], fontsize=11)
    ax.set_xlim(0.83, 0.88); ax.set_ylim(-0.6, len(ordine) - 0.4)
    ax.set_xticks([0.83, 0.84, 0.85, 0.86, 0.87, 0.88])
    assi_puliti(ax); ax.xaxis.grid(False); ax.tick_params(axis="x", labelsize=10, pad=2)
    h = [Patch(facecolor=COL[f], label=ETI[f]) for f in ("estr", "astr", "llm")]
    if budget:
        h.append(Line2D([], [], ls="", label="* solo tetto"))
    fig.legend(handles=h, loc="upper right", ncol=len(h), frameon=False, bbox_to_anchor=(0.97, 1.0),
               fontsize=11, handlelength=1.2, columnspacing=1.4)
    fig.savefig(OUT / nome, facecolor="white"); plt.close(fig)

# ---------- 5.3: G-Eval per dimensione (heatmap) ----------
DIM = [("geval_coherence", "Coherence"), ("geval_consistency", "Consistency"), ("geval_fluency", "Fluency"),
       ("geval_relevance", "Relevance"), ("geval_media", "Media")]
cmap = LinearSegmentedColormap.from_list("blu", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

def grafico_geval(scope, nome):
    from matplotlib.patches import Rectangle
    budget = scope.endswith("budgetref")
    g = {m: [leggi(m, scope, k) for k, _ in DIM] for m in NOMI}
    ordine = sorted(g, key=lambda m: g[m][-1], reverse=True)
    fig, ax = plt.subplots(figsize=(W, H), dpi=220)
    dati = [g[m] for m in ordine]
    ax.imshow(dati, cmap=cmap, vmin=1.5, vmax=5, aspect="auto")
    for i, riga in enumerate(dati):
        for j, val in enumerate(riga):
            ax.text(j, i, fmt(val, 2), ha="center", va="center", fontsize=10.5,
                    color="white" if val >= 3.6 else INK, fontweight="bold" if j == 4 else "normal")
    ax.set_xticks(range(len(DIM)), [n for _, n in DIM], fontsize=11.5); ax.xaxis.tick_top()
    ax.set_yticks(range(len(ordine)), [etichetta(m, budget) for m in ordine], fontsize=10.5)
    for i, m in enumerate(ordine):
        ax.add_patch(Rectangle((-0.62, i - 0.42), 0.09, 0.84, color=COL[FAM[m]], clip_on=False))
    ax.set_xlim(-0.65, len(DIM) - 0.5)
    ax.tick_params(axis="y", pad=4)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.axvline(3.5, color="white", lw=3)
    ax.set_xticks([x - 0.5 for x in range(1, 5)], minor=True); ax.set_yticks([y - 0.5 for y in range(1, 19)], minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
    h = [Patch(facecolor=COL[f], label=ETI[f]) for f in ("estr", "astr", "llm")]
    if budget:
        h.append(Line2D([], [], ls="", label="* solo tetto"))
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, bbox_to_anchor=(0.6, 0.0),
               fontsize=11.5, handlelength=1.2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.075, 1, 1)); fig.savefig(OUT / nome, facecolor="white"); plt.close(fig)

grafico_rouge("test", "slide_rouge1_f1.png")
grafico_rouge("test_budgetref", "slide_rouge1_f1_budgetref.png")
grafico_bertscore("test_budgetref", "slide_bertscore_f1_budgetref.png")
grafico_geval("test", "slide_geval_dimensioni.png")
grafico_geval("test_budgetref", "slide_geval_dimensioni_budgetref.png")

# ---------- 5.6: prima e dopo il budget (dumbbell, due pannelli) ----------
# Stessa logica di 5.2: tutta l'altezza disponibile, legenda e nota in un'unica riga in alto.
H56 = 4.15
r0 = {m: leggi(m, "test", "rouge1_f1") for m in NOMI}
r1 = {m: leggi(m, "test_budgetref", "rouge1_f1") for m in NOMI}
g0 = {m: leggi(m, "test", "geval_media") for m in NOMI}
g1 = {m: leggi(m, "test_budgetref", "geval_media") for m in NOMI}
ordine = sorted(NOMI, key=lambda m: r1[m])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, H56), dpi=220, sharey=True)
fig.subplots_adjust(left=0.215, right=0.985, top=0.875, bottom=0.065, wspace=0.07)
for ax, prima, dopo, lim, tit in ((a1, r0, r1, (0.25, 0.46), "ROUGE-1 F1"), (a2, g0, g1, (2.0, 5.0), "G-Eval medio (1-5)")):
    for i in range(len(ordine)):
        ax.axhline(i, color=GRID, lw=0.6, zorder=1)  # guida di riga: lega il nome ai suoi punti
    for i, m in enumerate(ordine):
        c = COL[FAM[m]]
        ax.plot([prima[m], dopo[m]], [i, i], color=c, lw=2.2, alpha=0.5, solid_capstyle="round", zorder=2)
        ax.scatter(prima[m], i, s=38, facecolor="white", edgecolor=c, linewidth=1.8, zorder=3)
        ax.scatter(dopo[m], i, s=42, color=c, edgecolor="white", linewidth=1, zorder=4)
    ax.set_xlim(*lim); ax.set_ylim(-0.6, len(ordine) - 0.4)
    ax.set_title(tit, fontsize=12, color=INK, pad=3, loc="left", fontweight="bold")
    assi_puliti(ax); ax.xaxis.grid(False); ax.tick_params(axis="x", labelsize=10, pad=2)
a1.set_yticks(range(len(ordine)), [etichetta(m, True) for m in ordine], fontsize=10.5)
h = [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2, markeredgewidth=1.6, markersize=7, label="a lunghezza libera"),
     Line2D([], [], marker="o", ls="", color=INK2, markersize=7, label="a lunghezza contenuta")]
h += [Patch(facecolor=COL[f], label=ETI[f]) for f in ("estr", "astr", "llm")]
h += [Line2D([], [], ls="", label="* solo tetto")]
fig.legend(handles=h, loc="upper right", ncol=6, frameon=False, bbox_to_anchor=(0.995, 1.0), fontsize=10,
           handlelength=0.9, handletextpad=0.35, columnspacing=0.9)
fig.savefig(OUT / "slide_prima_dopo.png", facecolor="white"); plt.close(fig)

print("figure scritte in", OUT)
