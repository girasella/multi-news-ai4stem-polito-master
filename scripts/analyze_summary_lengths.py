#!/usr/bin/env python3
# coding=utf-8
"""Per-cluster analysis of generated-summary lengths, across every benchmark method.

Answers the question the aggregate per-method means could not: *inside a single
Multi-News cluster*, how far apart are the summaries produced by the different
methods, and how do they relate to that cluster's reference summary?

Motivation (issue #15). The per-method means (55 words for ``bart`` up to 477 for
``lda``) hide the dispersion at the level where the comparison actually happens —
the cluster. They also hide the fact that the target is not a constant: the
reference summary grows with the number of source articles (median 161 words for
a 1-article cluster, 384 for 8+), which is why a single global word cap is the
wrong instrument.

Three things are computed, all read-only from files already committed:

1. **Dispersion inside each cluster** — spread (max - min), max/min ratio,
   coefficient of variation, and spread relative to the reference length.
2. **Per method** — the distribution of ``len / reference``, the share of clusters
   falling inside the target band, and how well the method's length tracks the
   reference length / the article count (it mostly does not).
3. **T(n_articoli)** — the median reference length per article-count bucket, the
   per-cluster word budget used by the containment work. Fitted on the **train**
   split so no test-set label information leaks into the budget.

Lengths are whitespace word counts (``str.split()``), the repo-wide convention:
they come straight from the ``parole_generate`` column that ``summ_utils`` already
writes into every per-example CSV, so nothing is regenerated or recomputed here.

Run from anywhere (paths resolve relative to this file):

    python scripts/analyze_summary_lengths.py                      # scope test (default)
    python scripts/analyze_summary_lengths.py --tabella-budget     # (re)fit T(n) on train
    python scripts/analyze_summary_lengths.py --scope test_budget  # the "after" run
    python scripts/analyze_summary_lengths.py --figure results/figures/analisi_lunghezze

Writes: results/metrics/analisi_lunghezze_{scope}.json
        scripts/budget_lunghezza.json           (only with --tabella-budget, or if missing)
        <dir>/*.png                             (only with --figure)
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / 'notebooks'
sys.path.insert(0, str(NOTEBOOKS_DIR))
import summ_utils as su  # noqa: E402  (needs NOTEBOOKS_DIR on sys.path first)

TABELLA_BUDGET_PATH = REPO_ROOT / 'scripts' / 'budget_lunghezza.json'

# Clusters with this many articles or more share the last bucket: beyond it the
# per-bucket sample gets too small for a stable median (22 clusters at 8+ in test).
BUCKET_MAX = 8

# Target band around the per-cluster budget T(n_articoli). The lower edge is what
# truncation alone can never reach for the short methods; the upper edge is the
# ceiling applied by scripts/applica_budget.py.
BANDA = (0.80, 1.25)

PERCENTILI = (10, 25, 50, 75, 90)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def carica_lunghezze(metrics_dir, scope):
    """{metodo: {row_id: parole}} from the committed per-example CSVs of `scope`.

    The G-Eval files ({m}_{scope}_geval_per_example.csv) do not match the glob and
    carry no length column anyway, so they are naturally excluded.
    """
    suffisso = f'_{scope}_per_example.csv'
    dati = {}
    for path in sorted(Path(metrics_dir).glob(f'*{suffisso}')):
        metodo = path.name[:-len(suffisso)]
        per_riga = {}
        with open(path, encoding='utf-8', newline='') as f:
            for riga in csv.DictReader(f):
                try:
                    per_riga[int(riga['row_id'])] = int(float(riga['parole_generate']))
                except (KeyError, TypeError, ValueError):
                    continue
        if per_riga:
            dati[metodo] = per_riga
    return dati


def scansiona_complete_tab(path, splits):
    """One streaming pass over complete.tab -> {split: [(row_id, n_articoli, parole_rif)]}.

    The article count is taken from the RAW document (counting the ``|||||``
    separator): after ``prepara_documento`` the separator has become a newline and
    is no longer distinguishable from newlines inside an article.
    """
    voluti = set(splits)
    out = {s: [] for s in splits}
    for es in su.itera_complete_tab(path):
        if es['split'] not in voluti:
            continue
        n_articoli = es['document'].count(su.SEPARATORE_ARTICOLI) + 1
        parole = len(su.pulisci_riferimento(es['summary']).split())
        out[es['split']].append((es['row_id'], n_articoli, parole))
    return out


# ---------------------------------------------------------------------------
# Budget table T(n_articoli)
# ---------------------------------------------------------------------------

def calcola_tabella_budget(righe):
    """Median reference length per article-count bucket -> {'1': 161, ..., '8': 384}."""
    per_bucket = defaultdict(list)
    for _, n_articoli, parole in righe:
        per_bucket[min(n_articoli, BUCKET_MAX)].append(parole)
    tabella = {str(n): int(round(float(np.median(v)))) for n, v in sorted(per_bucket.items())}
    conteggi = {str(n): len(v) for n, v in sorted(per_bucket.items())}
    return tabella, conteggi


def budget_riga(n_articoli, tabella):
    """Word budget for one cluster, from its article count."""
    return tabella[str(min(n_articoli, BUCKET_MAX))]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def distribuzione(valori, cifre=2):
    """Percentile summary of a 1-D sample, plus mean/min/max."""
    a = np.asarray(valori, dtype=np.float64)
    d = {f'p{p}': round(float(np.percentile(a, p)), cifre) for p in PERCENTILI}
    d.update(media=round(float(a.mean()), cifre),
             min=round(float(a.min()), cifre),
             max=round(float(a.max()), cifre),
             n=int(a.size))
    return d


def pearson(x, y):
    """Pearson r, or None when one of the two samples is constant."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 3)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analizza(lunghezze, righe_split, tabella):
    """Per-cluster dispersion + per-method behaviour, over the common row_ids."""
    rif = {row_id: (n_articoli, parole) for row_id, n_articoli, parole in righe_split}
    metodi = sorted(lunghezze)
    comuni = sorted(set.intersection(*(set(d) for d in lunghezze.values())) & set(rif))
    if not comuni:
        raise SystemExit('Nessun cluster in comune fra i metodi trovati: niente da analizzare.')

    # --- 1. dispersion inside each cluster ---
    spread, rapporto, cv, spread_rel = [], [], [], []
    for row_id in comuni:
        L = np.array([lunghezze[m][row_id] for m in metodi], dtype=np.float64)
        parole_rif = rif[row_id][1]
        spread.append(L.max() - L.min())
        rapporto.append(L.max() / max(L.min(), 1.0))
        cv.append(L.std() / L.mean() if L.mean() else 0.0)
        spread_rel.append((L.max() - L.min()) / max(parole_rif, 1))

    # --- 2. per method ---
    R = np.array([rif[r][1] for r in comuni], dtype=np.float64)        # reference words
    A = np.array([rif[r][0] for r in comuni], dtype=np.float64)        # article count
    B = np.array([budget_riga(rif[r][0], tabella) for r in comuni], dtype=np.float64)

    per_metodo = {}
    for m in metodi:
        L = np.array([lunghezze[m][r] for r in comuni], dtype=np.float64)
        in_banda = float(((L >= BANDA[0] * B) & (L <= BANDA[1] * B)).mean())
        per_metodo[m] = {
            'parole': distribuzione(L, cifre=1),
            'rapporto_riferimento': distribuzione(L / np.maximum(R, 1), cifre=2),
            'rapporto_budget': distribuzione(L / B, cifre=2),
            'pct_in_banda': round(100 * in_banda, 1),
            'corr_len_riferimento': pearson(L, R),
            'corr_len_n_articoli': pearson(L, A),
        }

    return {
        'n_cluster': len(comuni),
        'metodi': metodi,
        'riferimento': distribuzione(R, cifre=1),
        'n_articoli': distribuzione(A, cifre=2),
        'dispersione_per_cluster': {
            'spread_parole': distribuzione(spread, cifre=1),
            'rapporto_max_min': distribuzione(rapporto, cifre=2),
            'cv': distribuzione(cv, cifre=3),
            'spread_su_riferimento': distribuzione(spread_rel, cifre=2),
        },
        'per_metodo': per_metodo,
        '_serie': {'comuni': comuni, 'rapporto': rapporto, 'R': R, 'B': B},
    }


# ---------------------------------------------------------------------------
# Printing (Italian: this output is the report)
# ---------------------------------------------------------------------------

def stampa(ris, tabella, conteggi_budget, scope):
    etichette = '  '.join(f'{f"p{p}":>8}' for p in PERCENTILI)
    print(f'\nAmbito "{scope}" - {ris["n_cluster"]} cluster con tutti e '
          f'{len(ris["metodi"])} i metodi e il riassunto di riferimento\n')

    print('1. DISPERSIONE DENTRO CIASCUN CLUSTER')
    print(f'   {"":<32}{etichette}')
    nomi = {'spread_parole': 'spread max-min (parole)',
            'rapporto_max_min': 'rapporto max/min',
            'cv': 'CV (std/media)',
            'spread_su_riferimento': 'spread / lunghezza riferimento'}
    for chiave, nome in nomi.items():
        d = ris['dispersione_per_cluster'][chiave]
        print(f'   {nome:<32}' + '  '.join(f'{d[f"p{p}"]:>8}' for p in PERCENTILI))

    print('\n2. LUNGHEZZA DEL RIFERIMENTO')
    d = ris['riferimento']
    print(f'   {"parole riferimento":<32}' + '  '.join(f'{d[f"p{p}"]:>8}' for p in PERCENTILI))
    print(f'   media {d["media"]}  min {d["min"]}  max {d["max"]}')

    print(f'\n3. BUDGET T(n_articoli) - mediana del riferimento per numero di articoli '
          f'(tarata su train)')
    print(f'   {"n_articoli":<14}' + ''.join(f'{n:>8}' for n in tabella))
    print(f'   {"T (parole)":<14}' + ''.join(f'{v:>8}' for v in tabella.values()))
    print(f'   {"n_cluster":<14}' + ''.join(f'{conteggi_budget[n]:>8}' for n in tabella))

    print(f'\n4. PER METODO - rapporto lunghezza/riferimento, banda '
          f'[{BANDA[0]:.2f}, {BANDA[1]:.2f}]xT')
    print(f'   {"metodo":<20}{"parole":>8}{"p10":>7}{"mediana":>9}{"p90":>7}'
          f'{"in banda":>10}{"corr(rif)":>11}{"corr(n_art)":>13}')
    ordinati = sorted(ris['per_metodo'].items(),
                      key=lambda kv: kv[1]['rapporto_riferimento']['p50'])
    for metodo, d in ordinati:
        r = d['rapporto_riferimento']
        c1 = '  n/d' if d['corr_len_riferimento'] is None else f'{d["corr_len_riferimento"]:>11.2f}'
        c2 = '  n/d' if d['corr_len_n_articoli'] is None else f'{d["corr_len_n_articoli"]:>13.2f}'
        print(f'   {metodo:<20}{d["parole"]["media"]:>8.1f}{r["p10"]:>7.2f}'
              f'{r["p50"]:>9.2f}{r["p90"]:>7.2f}{d["pct_in_banda"]:>9.1f}%{c1}{c2}')


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def scrivi_figure(ris, lunghezze, cartella, scope):
    """Three PNGs: ratio boxplot, max/min histogram, reference-vs-length scatter."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cartella = Path(cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    comuni = ris['_serie']['comuni']
    R = ris['_serie']['R']
    scritte = []

    # (a) rapporto lunghezza/riferimento per metodo, con la banda evidenziata
    ordinati = sorted(ris['per_metodo'],
                      key=lambda m: ris['per_metodo'][m]['rapporto_riferimento']['p50'])
    serie = [np.array([lunghezze[m][r] for r in comuni], dtype=np.float64) / np.maximum(R, 1)
             for m in ordinati]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.axhspan(BANDA[0], BANDA[1], color='#2a9d8f', alpha=0.15,
               label=f'banda {BANDA[0]:.2f}-{BANDA[1]:.2f}x il riferimento')
    ax.axhline(1.0, color='#264653', lw=1, ls='--', label='pari al riferimento')
    # `tick_labels` (matplotlib >= 3.9) replaced the old `labels` kwarg
    ax.boxplot(serie, showfliers=False, tick_labels=ordinati)
    ax.set_yscale('log')
    ax.set_ylabel('lunghezza generata / lunghezza riferimento')
    ax.set_title(f'Rapporto col riassunto di riferimento, per cluster — ambito "{scope}"')
    ax.tick_params(axis='x', rotation=90)
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    p = cartella / f'rapporto_riferimento_{scope}.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    scritte.append(p)

    # (b) distribuzione del rapporto max/min dentro il cluster
    fig, ax = plt.subplots(figsize=(8, 4))
    rapporto = np.asarray(ris['_serie']['rapporto'], dtype=np.float64)
    ax.hist(np.clip(rapporto, 0, 30), bins=60, color='#e76f51', edgecolor='white', lw=0.4)
    mediana = float(np.median(rapporto))
    ax.axvline(mediana, color='#264653', lw=1.5, ls='--', label=f'mediana {mediana:.1f}x')
    ax.set_xlabel('rapporto lunghezza massima / minima dentro il cluster '
                  '(asse troncato a 30x)')
    ax.set_ylabel('numero di cluster')
    ax.set_title(f'Quanto sono diversi fra loro i riassunti di uno stesso cluster — ambito "{scope}"')
    ax.legend()
    fig.tight_layout()
    p = cartella / f'rapporto_max_min_{scope}.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    scritte.append(p)

    # (c) il metodo si adatta al cluster? riferimento vs lunghezza generata
    campione = [m for m in ('lda', 'lexrank', 'lsa', 'primera', 'bart', 'gpt5mini')
                if m in lunghezze][:6]
    if campione:
        righe = (len(campione) + 2) // 3
        fig, axes = plt.subplots(righe, 3, figsize=(11, 3.4 * righe), squeeze=False)
        for ax, metodo in zip(axes.ravel(), campione):
            L = np.array([lunghezze[metodo][r] for r in comuni], dtype=np.float64)
            ax.scatter(R, L, s=3, alpha=0.15, color='#457b9d', edgecolors='none')
            lim = float(max(R.max(), L.max()))
            ax.plot([0, lim], [0, lim], color='#264653', lw=1, ls='--')
            r = ris['per_metodo'][metodo]['corr_len_riferimento']
            ax.set_title(f'{metodo} (r = {r})', fontsize=10)
            ax.set_xlabel('parole riferimento')
            ax.set_ylabel('parole generate')
            ax.set_xlim(0, 700)
            ax.set_ylim(0, 1200)
        for ax in axes.ravel()[len(campione):]:
            ax.axis('off')
        fig.suptitle(f'Adattamento alla lunghezza del cluster — ambito "{scope}"')
        fig.tight_layout()
        p = cartella / f'adattamento_{scope}.png'
        fig.savefig(p, dpi=150)
        plt.close(fig)
        scritte.append(p)

    for p in scritte:
        print(f'Figura: {p}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analisi per cluster delle lunghezze dei riassunti generati.")
    parser.add_argument('--scope', default='test',
                        help="ambito delle metriche da leggere (default: test; "
                             "usare test_budget per la corsa 'dopo')")
    parser.add_argument('--split', default='test',
                        help="split di complete.tab da cui prendere i riferimenti (default: test)")
    parser.add_argument('--tabella-budget', action='store_true',
                        help="ricalcola T(n_articoli) dalla split train e riscrive "
                             "scripts/budget_lunghezza.json (altrimenti si riusa quello esistente)")
    parser.add_argument('--figure', default=None, metavar='CARTELLA',
                        help="scrive le figure PNG nella cartella indicata")
    args = parser.parse_args()

    percorsi = su.percorsi_standard(REPO_ROOT)
    lunghezze = carica_lunghezze(percorsi['metrics_dir'], args.scope)
    if not lunghezze:
        raise SystemExit(f'Nessun file *_{args.scope}_per_example.csv in '
                         f'{percorsi["metrics_dir"]}')
    print(f'Metodi trovati per l\'ambito "{args.scope}": {len(lunghezze)}')

    serve_train = args.tabella_budget or not TABELLA_BUDGET_PATH.exists()
    splits = [args.split] + (['train'] if serve_train and args.split != 'train' else [])
    print(f'Scansione di {percorsi["complete_tab"].name} (split: {", ".join(splits)}) - '
          f'e\' il passaggio piu\' lento, ~1 minuto...')
    righe = scansiona_complete_tab(percorsi['complete_tab'], splits)

    if serve_train:
        tabella, conteggi = calcola_tabella_budget(righe['train'])
        contenuto = {
            'descrizione': 'T(n_articoli): mediana della lunghezza (parole) del riassunto di '
                           'riferimento per numero di articoli del cluster. Budget per-cluster '
                           'del contenimento delle lunghezze (issue #15).',
            'split_di_taratura': 'train',
            'bucket_massimo': BUCKET_MAX,
            'banda': list(BANDA),
            'n_cluster_per_bucket': conteggi,
            'tabella': tabella,
        }
        TABELLA_BUDGET_PATH.write_text(
            json.dumps(contenuto, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Tabella budget   : {TABELLA_BUDGET_PATH}')
    else:
        contenuto = json.loads(TABELLA_BUDGET_PATH.read_text(encoding='utf-8'))
        tabella, conteggi = contenuto['tabella'], contenuto['n_cluster_per_bucket']

    ris = analizza(lunghezze, righe[args.split], tabella)
    stampa(ris, tabella, conteggi, args.scope)

    if args.figure:
        scrivi_figure(ris, lunghezze, args.figure, args.scope)

    serie = ris.pop('_serie')
    ris.update(scope=args.scope, split=args.split, banda=list(BANDA),
               tabella_budget=tabella, split_taratura_budget='train')
    out = Path(percorsi['metrics_dir']) / f'analisi_lunghezze_{args.scope}.json'
    out.write_text(json.dumps(ris, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nAnalisi          : {out}')
    del serie


if __name__ == '__main__':
    sys.exit(main())
