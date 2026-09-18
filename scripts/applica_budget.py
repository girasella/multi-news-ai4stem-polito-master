#!/usr/bin/env python3
# coding=utf-8
"""Ceiling + re-scoring for the reference-length scope (issue #16, oracle-length protocol).

For every method slug this script builds the ``test_budgetref`` metrics in two steps:

1. **Ceiling.** Each summary is truncated to ``1.25 x B_i`` words, where ``B_i`` is the
   length of that cluster's reference summary (``summ_utils.budget_riferimento``). The
   source is the regenerated ``{slug}_test_budgetref.tsv`` when it exists (the 11
   extractive slugs and, once run, the LLMs), otherwise the committed ``{slug}_test.tsv``
   (the abstractive methods that only receive the ceiling; ``textrank``/``lexrank`` fall
   back to their ``_full.tsv`` if no budget run exists). Which file was used is recorded
   in the aggregate JSON (``config.ambito_budget``).
2. **Re-scoring.** ROUGE-1/2/L, BLEU, METEOR and ``parole_generate`` through the shared
   ``summ_utils.valuta_e_salva``, plus BERTScore (``calcola_bertscore_batch``) unless
   ``--senza-bertscore``. Output: ``results/metrics/{slug}_test_budgetref_per_example.csv``
   and ``..._aggregate.json``. The committed ``*_test_*`` files are never touched.

The notebooks that regenerate at budget also write ``{slug}_test_budgetref_*`` metrics on
the raw (un-truncated) output; this script overwrites them with the post-ceiling numbers,
which are the ones to read. Resumable per method: a slug whose aggregate JSON already
carries the ceiling marker is skipped unless ``--forza``.

Truncation only installs the ceiling: methods whose summaries are shorter than the
reference (``bart`` above all) stay short. Their share of rows inside the band is what
the final table reports, not something this script fixes.

Usage (from anywhere — paths resolve relative to this file):

    python scripts/applica_budget.py                      # 18 slugs, with BERTScore (GPU, ~1.5 h)
    python scripts/applica_budget.py --senza-bertscore    # lexical metrics only (minutes)
    python scripts/applica_budget.py --solo lda,lexrank   # subset
    python scripts/applica_budget.py --forza              # recompute slugs already done
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / 'notebooks'
sys.path.insert(0, str(NOTEBOOKS_DIR))
import summ_utils as su  # noqa: E402  (needs NOTEBOOKS_DIR on sys.path first)

SCOPE = 'test' + su.SUFFISSO_BUDGET          # test_budgetref
FATTORE_SOFFITTO = su.FATTORE_SOFFITTO_BUDGET  # 1.25: shared with notebook 14 (G-Eval judges the same text)

# Same 18 slugs as notebooks 05/13, in the driver's fastest-first order.
METODI = ['firstk_psr', 'firstk_nltk', 'lda', 'lsa', 'lsa_steinberger',
          'sbert_kmeans', 'sbert_agglom', 'centroid_mmr', 'centroid_mmr_bert',
          'lexrank', 'textrank',
          'bart', 'pegasus', 'primera', 'qwen', 'mistral', 'gemma', 'gpt5mini']


def sorgente_riassunti(summaries_dir, metodo):
    """(path, rigenerato): the budget run if present, else the committed test/full TSV."""
    for suffisso, rigenerato in ((SCOPE, True), ('test', False), ('full', False)):
        path = summaries_dir / f'{metodo}_{suffisso}.tsv'
        if path.exists():
            return path, rigenerato
    return None, False


def gia_fatto(metrics_dir, metodo):
    path = metrics_dir / f'{metodo}_{SCOPE}_aggregate.json'
    if not path.exists():
        return False
    config = json.loads(path.read_text(encoding='utf-8')).get('config', {})
    return 'soffitto' in config.get('ambito_budget', {})


def config_storico(metrics_dir, metodo):
    path = metrics_dir / f'{metodo}_test_aggregate.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8')).get('config', {})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--solo', default=None,
                        help='slug separati da virgola (default: tutti e 18)')
    parser.add_argument('--senza-bertscore', action='store_true',
                        help='solo metriche lessicali (minuti invece di ~1,5 h di GPU)')
    parser.add_argument('--forza', action='store_true',
                        help='ricalcola anche i metodi che hanno gia\' il soffitto applicato')
    parser.add_argument('--batch-size', type=int, default=64, help='batch del BERTScore')
    args = parser.parse_args()

    percorsi = su.percorsi_standard(REPO_ROOT)
    metodi = METODI if not args.solo else [m.strip() for m in args.solo.split(',') if m.strip()]
    sconosciuti = set(metodi) - set(METODI)
    if sconosciuti:
        parser.error(f'--solo: slug sconosciuti {sorted(sconosciuti)}')

    print(f'Ambito {SCOPE}: soffitto a {FATTORE_SOFFITTO} x lunghezza del riferimento, '
          f'{"senza" if args.senza_bertscore else "con"} BERTScore')
    print('Lettura dei riferimenti della split test...', flush=True)
    riferimenti = list(su.itera_split(percorsi['complete_tab'], 'test'))
    soffitto = {es['row_id']: su.soffitto_riferimento(es) for es in riferimenti}
    print(f'  {len(riferimenti)} righe')

    device = None if args.senza_bertscore else su.rileva_device()
    riepilogo = {}
    for metodo in metodi:
        if gia_fatto(percorsi['metrics_dir'], metodo) and not args.forza:
            print(f'\n== {metodo}: gia\' fatto, salto (--forza per ricalcolare) ==')
            continue
        path, rigenerato = sorgente_riassunti(percorsi['summaries_dir'], metodo)
        if path is None:
            print(f'\n== {metodo}: nessun TSV di riassunti trovato, salto ==')
            continue
        t0 = time.time()
        print(f'\n== {metodo} <- {path.name} ({"rigenerato a budget" if rigenerato else "solo soffitto"}) ==',
              flush=True)
        riassunti = su.carica_riassunti(path)
        troncati = {rid: su.tronca_parole(testo, soffitto[rid])
                    for rid, testo in riassunti.items() if rid in soffitto}
        n_tagliati = sum(len(riassunti[rid].split()) > soffitto[rid] for rid in troncati)
        print(f'  {len(troncati)} righe, {n_tagliati} troncate dal soffitto')

        config = dict(config_storico(percorsi['metrics_dir'], metodo))
        config['ambito_budget'] = {
            'budget': 'lunghezza del riassunto di riferimento (oracle-length, issue #16)',
            'soffitto': f'{FATTORE_SOFFITTO} x budget, su.tronca_parole',
            'sorgente_riassunti': path.name,
            'rigenerato_a_budget': rigenerato,
            'righe_troncate_dal_soffitto': n_tagliati,
        }

        extra, colonne_extra = None, None
        if not args.senza_bertscore:
            coppie = [(rif['row_id'], troncati[rif['row_id']], su.pulisci_riferimento(rif['summary']))
                      for rif in riferimenti if rif['row_id'] in troncati]
            print(f'  BERTScore su {len(coppie)} coppie ({device})...', flush=True)
            extra = su.calcola_bertscore_batch(coppie, device=device, batch_size=args.batch_size)
            colonne_extra = su.COLONNE_METRICHE_BERTSCORE

        _, aggregato = su.valuta_e_salva(riferimenti, troncati, metodo, SCOPE,
                                         percorsi['metrics_dir'], config,
                                         extra_metriche=extra, colonne_extra=colonne_extra)
        riepilogo[metodo] = aggregato['overall']
        print(f'  fatto in {time.time() - t0:.0f}s')

    if riepilogo:
        print(f'\n{"metodo":<20}{"parole test":>12}{"parole budget":>14}{"R1 rec":>8}{"R1 F1":>8}')
        for metodo, o in riepilogo.items():
            prima = percorsi['metrics_dir'] / f'{metodo}_test_aggregate.json'
            parole_prima = (json.loads(prima.read_text(encoding='utf-8'))['overall']['parole_generate']
                            if prima.exists() else float('nan'))
            print(f'{metodo:<20}{parole_prima:>12.1f}{o["parole_generate"]:>14.1f}'
                  f'{o["rouge1_r"]:>8.3f}{o["rouge1_f1"]:>8.3f}')


if __name__ == '__main__':
    sys.exit(main())
