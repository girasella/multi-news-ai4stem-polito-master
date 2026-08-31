"""Unattended driver for the G-Eval (LLM-as-a-Judge) backfill — notebook 14.

Executes notebooks/14_geval.ipynb via nbconvert with the GEVAL_* environment variables set,
so the notebook stays the single source of truth (and the documentary artifact) while this
script handles preflight, staged runs, cost reporting and logging.

Scope: the full test split judged by gpt-5.4-mini on Azure — ~100,600 judgments over 5,610 rows
(under 18x5,610 because coverage is uneven: firstk_psr, the centroid variants and the five
notebook 15-17 methods cover 5,588 rows, gpt5mini 5,471 after Azure's content filter). The
first 72,681 (the 13 original methods) are already in the committed cache; a relaunch judges
only what is missing. This costs real money and runs for hours, so the intended order is:
--righe 1 (smoke), then --pilota 20, then the full run.

Usage (from the repo root or anywhere — paths are resolved relative to this script):

    python scripts/run_geval.py --righe 1        # smoke: one call per not-yet-cached method
                                                 # on row 1, proves the prompt cache
    python scripts/run_geval.py --pilota 20      # pilot: 20 rows, measures cost/judgment
    python scripts/run_geval.py --budget 120                # hard stop once $120 is spent
    python scripts/run_geval.py --budget 7 --valuta EUR      # hard stop once EUR 7 is spent —
                                                              # --valuta MUST match the billing
                                                              # currency of your subscription,
                                                              # Azure's EUR price list is not a
                                                              # live FX conversion of the USD one
    python scripts/run_geval.py --righe 500 --thread 12
    python scripts/run_geval.py --solo-metriche  # rewrite CSV/JSON from the cache, ZERO calls
    python scripts/run_geval.py --costo          # cost report from the cache, ZERO calls
    python scripts/run_geval.py --riprova-errori # drop cached failures so they are retried
    python scripts/run_geval.py --no-05          # skip re-running notebook 05 at the end

--costo needs no notebook execution and touches no API: every cached judgment carries its own
token counts, so it can be run from a SECOND TERMINAL while a long run is in progress to see
spend-to-date and the projection to completion. This is the practical monitoring surface —
Azure has no real-time cost API (Cost Management lags 8-24 h; see scripts/README.md for the
next-day reconciliation command).

The run is resumable: each judgment is flushed to results/metrics/geval_cache_test.jsonl as
it arrives and a relaunch skips the (method, row_id) pairs already present, so Ctrl-C or a
budget stop never loses anything already paid for.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
NOTEBOOKS_DIR = REPO_ROOT / 'notebooks'
RESULTS_DIR = REPO_ROOT / 'results'
LOG_PATH = REPO_ROOT / 'run_geval.log'

sys.path.insert(0, str(NOTEBOOKS_DIR))
import summ_utils as su  # noqa: E402  (needs NOTEBOOKS_DIR on sys.path first)

NOTEBOOK = '14_geval.ipynb'
NOTEBOOK_CONFRONTO = '05_confronto.ipynb'

# Gli stessi 18 slug del notebook 05 (Vista 2), 13 e 14. Gli ultimi cinque (notebook
# 15-17) sono stati aggiunti col backfill dell'issue #12: la cache e' per (metodo,
# row_id), quindi allargare la lista fa giudicare SOLO i nuovi, senza ripagare i 13.
METODI = ['firstk_psr', 'firstk_nltk', 'centroid_mmr', 'centroid_mmr_bert',
          'textrank', 'lexrank', 'bart', 'pegasus', 'primera',
          'qwen', 'gemma', 'mistral', 'gpt5mini',
          'lsa', 'lsa_steinberger', 'sbert_kmeans', 'sbert_agglom', 'lda']


def log(message):
    stamped = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}'
    print(stamped, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(stamped + '\n')


def percorso_riassunti(metodo, scope):
    """textrank/lexrank hanno solo la corsa '_full.tsv' (vedi notebook 13/14)."""
    suffisso = 'full' if metodo in ('textrank', 'lexrank') else scope
    return RESULTS_DIR / 'summaries' / f'{metodo}_{suffisso}.tsv'


def percorso_cache(scope):
    return RESULTS_DIR / 'metrics' / f'geval_cache_{scope}.jsonl'


def giudizi_attesi(scope):
    """Totale dei giudizi previsti = somma della copertura per metodo sullo scope.

    Si legge dal CSV per-esempio STANDARD di ciascun metodo, che contiene
    esattamente le righe dello scope gia' valutate: contare invece le righe dei
    TSV dei riassunti darebbe 55.894 per textrank/lexrank (che hanno solo la
    corsa full sull'intero complete.tab), gonfiando la proiezione di 10 volte.
    """
    totale = 0
    for metodo in METODI:
        csv_path = RESULTS_DIR / 'metrics' / f'{metodo}_{scope}_per_example.csv'
        if csv_path.exists():
            with open(csv_path, encoding='utf-8') as f:
                totale += max(sum(1 for _ in f) - 1, 0)   # -1 = intestazione
    return totale


def conta_cache(scope):
    """Numero di giudizi gia' in cache (0 se il file non esiste)."""
    path = percorso_cache(scope)
    if not path.exists():
        return 0
    cache = su.CacheGiudizi(path)
    try:
        return len(cache)
    finally:
        cache.chiudi()


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(scope, deployment, con_api=True):
    """Fallisce subito su tutto cio' che altrimenti romperebbe ore dopo l'avvio.

    Il controllo decisivo e' il ping da 1 token sul deployment del giudice: un
    deployment inesistente o con il nome sbagliato deve costare secondi, non ore.
    """
    problemi = []

    if con_api:
        for chiave in ('AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY'):
            if not os.environ.get(chiave):
                problemi.append(f'Variabile d\'ambiente {chiave} non impostata.')

        if not problemi:
            try:
                from openai import OpenAI
                client = OpenAI(
                    base_url=os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/') + '/openai/v1/',
                    api_key=os.environ['AZURE_OPENAI_API_KEY'])
                client.chat.completions.create(
                    model=deployment,
                    messages=[{'role': 'user', 'content': 'ping'}],
                    max_completion_tokens=16)
                log(f'Preflight: deployment "{deployment}" raggiungibile.')
            except ImportError:
                problemi.append('Pacchetto openai non installato: '
                                'pip install -r requirements-notebooks.txt')
            except Exception as exc:
                problemi.append(f'Deployment "{deployment}" non utilizzabile ({exc!r}). '
                                'Verificare il nome nel portale Azure o impostare '
                                'AZURE_GEVAL_DEPLOYMENT.')

    complete_tab = REPO_ROOT / 'data' / 'tab' / 'complete.tab'
    if not complete_tab.exists():
        problemi.append(f'{complete_tab} non trovato (rigenerarlo con scripts/convert_to_tab.py).')

    mancanti = [m for m in METODI if not percorso_riassunti(m, scope).exists()]
    if mancanti:
        problemi.append(f'Riassunti mancanti per i metodi {mancanti}: eseguire prima '
                        'scripts/run_benchmark_test.py (e il notebook 12 per gpt5mini).')

    if shutil.which('jupyter') is None:
        try:
            import nbconvert  # noqa: F401
        except ImportError:
            problemi.append('ne\' il comando "jupyter" ne\' il pacchetto nbconvert sono '
                            'disponibili: installare requirements-notebooks.txt.')

    if problemi:
        log('Preflight fallito:')
        for p in problemi:
            log(f'  - {p}')
        return False
    log('Preflight OK.')
    return True


# ---------------------------------------------------------------------------
# Report di costo (offline, nessuna chiamata API)
# ---------------------------------------------------------------------------

def rapporto_costo(scope, prezzi=None, valuta='USD'):
    """Spesa a oggi e proiezione, ricalcolate dai soli conteggi salvati in cache.

    Non tocca l'API: si puo' lanciare da un secondo terminale mentre la corsa
    lunga e' in esecuzione. Azure non espone il costo in tempo reale (Cost
    Management ha 8-24 h di ritardo), quindi questa e' la fonte di verita' PER
    I TOKEN — ma la CIFRA dipende dal listino usato. `valuta` va fatta
    corrispondere alla valuta di fatturazione reale: il listino EUR di Azure
    non e' una conversione al cambio, e' un listino a se stante (verificato,
    ~0,8776x il numero USD su ogni meter). Con la valuta sbagliata i token
    contati restano esatti ma l'importo in cifra no.
    """
    path = percorso_cache(scope)
    if not path.exists():
        log(f'Nessuna cache in {path}: niente da riportare.')
        return
    simbolo = {'EUR': '€', 'USD': '$'}.get(valuta.upper(), valuta.upper() + ' ')
    prezzi = prezzi or su.prezzi_retail_azure(valuta=valuta)
    cache = su.CacheGiudizi(path)
    totali = cache.totali_token()
    errori = cache.errori()
    cache.chiudi()

    costi = su.costo_da_token(totali, prezzi)
    totale = sum(costi.values())
    n = totali['n_giudizi']
    n_pagati = max(totali['n_riusciti'], 1)

    rimanenti = max(giudizi_attesi(scope) - n, 0)

    log(f'--- Costo G-Eval (scope={scope}) ---')
    log(f'  giudizi in cache : {n:,} ({totali["n_riusciti"]:,} riusciti, '
        f'{len(errori):,} falliti)')
    log(f'  token input      : {totali["prompt_tokens"]:,} '
        f'({totali["cached_tokens"] / max(totali["prompt_tokens"], 1):.0%} in cache)')
    log(f'  token output     : {totali["completion_tokens"]:,} '
        f'({totali["reasoning_tokens"] / max(totali["completion_tokens"], 1):.0%} reasoning)')
    log(f'  SPESA A OGGI     : {simbolo}{totale:.2f}  '
        f'(input {simbolo}{costi["input"]:.2f} + cached {simbolo}{costi["cached"]:.2f} + '
        f'output {simbolo}{costi["output"]:.2f})')
    log(f'  costo/giudizio   : {simbolo}{totale / n_pagati:.5f}')
    if rimanenti:
        log(f'  giudizi rimasti  : ~{rimanenti:,}')
        log(f'  PROIEZIONE finale: ~{simbolo}{totale + totale / n_pagati * rimanenti:.2f}')
    log(f'  prezzi usati ({valuta.upper()}/1M token): {prezzi}')


# ---------------------------------------------------------------------------
# Esecuzione notebook
# ---------------------------------------------------------------------------

def esegui_notebook(nome, env_extra=None):
    """Esegue un notebook in-place via nbconvert con le variabili GEVAL_* impostate."""
    env = dict(os.environ, **(env_extra or {}))
    inizio = time.time()
    dettagli = ', '.join(f'{k}={v}' for k, v in sorted((env_extra or {}).items()))
    log(f'=== Avvio {nome} ({dettagli or "nessuna variabile extra"}) ===')

    risultato = subprocess.run(
        [sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
         '--inplace', '--ExecutePreprocessor.timeout=-1', nome],
        cwd=NOTEBOOKS_DIR, env=env, capture_output=True, text=True)
    durata = time.time() - inizio

    if risultato.returncode != 0:
        log(f'*** ERRORE in {nome} dopo {durata:.0f}s (returncode={risultato.returncode}) ***')
        log('--- stderr (ultime 40 righe) ---')
        for riga in risultato.stderr.splitlines()[-40:]:
            log(f'    {riga}')
        return False

    log(f'=== Completato {nome} in {durata:.0f}s ===')
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scope', default='test',
                        help='Split da giudicare (default: test).')
    parser.add_argument('--righe', type=int, default=None,
                        help='Limita la corsa alle prime N righe (13 giudizi ciascuna). '
                             '--righe 1 e\' lo smoke test end-to-end.')
    parser.add_argument('--pilota', type=int, default=None,
                        help='Esegue il pilota su N righe scelte con seed fisso, piu\' le '
                             'due righe PEGASUS patologiche, e stampa il costo estrapolato.')
    parser.add_argument('--thread', type=int, default=None,
                        help='Thread concorrenti (default: 8). Il collo di bottiglia vero '
                             'e\' la quota TPM del deployment, non questo numero.')
    parser.add_argument('--budget', type=float, default=None,
                        help='Tetto di spesa COMPLESSIVO (nella valuta di --valuta), non per '
                             'sessione: include quanto e\' gia\' costato cio\' che sta in '
                             'cache. Al superamento la corsa si ferma in modo pulito (basta '
                             'rilanciare con un tetto piu\' alto per riprendere).')
    parser.add_argument('--valuta', default=os.environ.get('GEVAL_VALUTA', 'USD'),
                        help='Valuta di --budget e dei report di costo: deve corrispondere '
                             'alla valuta di FATTURAZIONE della sottoscrizione, non e\' un '
                             'default innocuo — il listino EUR di Azure non e\' una '
                             'conversione al cambio del listino USD (default: USD, o '
                             'GEVAL_VALUTA se impostata).')
    parser.add_argument('--casuale', action='store_true',
                        help='Ordina le righe casualmente (seed 42) invece che per row_id. '
                             'Da usare SEMPRE insieme a --budget: se il tetto ferma la corsa '
                             'a meta\', quello che si e\' riusciti a giudicare e\' un campione '
                             'casuale della split test invece del suo primo tratto, quindi le '
                             'medie restano stime non distorte.')
    parser.add_argument('--ogni', type=int, default=None,
                        help='Ogni quanti giudizi stampare il blocco costo (default: 1500).')
    parser.add_argument('--solo-metriche', action='store_true',
                        help='Riscrive CSV/JSON dalla cache senza alcuna chiamata API.')
    parser.add_argument('--costo', action='store_true',
                        help='Stampa spesa e proiezione dalla cache ed esce. Nessuna '
                             'esecuzione di notebook, nessuna chiamata API.')
    parser.add_argument('--riprova-errori', action='store_true',
                        help='Rimuove dalla cache i giudizi falliti cosi\' vengono ritentati.')
    parser.add_argument('--no-05', action='store_true',
                        help='Non rieseguire il notebook 05 al termine.')
    args = parser.parse_args()

    if args.costo:
        rapporto_costo(args.scope, valuta=args.valuta)
        return

    deployment = os.environ.get('AZURE_GEVAL_DEPLOYMENT', 'gpt-5.4-mini')
    con_api = not args.solo_metriche

    log(f'--- Avvio G-Eval (scope={args.scope}, giudice={deployment}, '
        f'pilota={args.pilota}, righe={args.righe}, budget={args.budget} {args.valuta}) ---')
    if not preflight(args.scope, deployment, con_api=con_api):
        log('Interruzione: risolvere i problemi sopra prima di rilanciare.')
        sys.exit(1)

    prima = conta_cache(args.scope)
    log(f'Giudizi gia\' in cache: {prima:,}')

    env = {'GEVAL_SCOPE': args.scope, 'AZURE_GEVAL_DEPLOYMENT': deployment,
          'GEVAL_VALUTA': args.valuta}
    if args.righe is not None:
        env['GEVAL_RIGHE'] = str(args.righe)
    if args.pilota is not None:
        env['GEVAL_PILOTA'] = str(args.pilota)
    if args.thread is not None:
        env['GEVAL_THREAD'] = str(args.thread)
    if args.budget is not None:
        env['GEVAL_BUDGET'] = str(args.budget)
    if args.ogni is not None:
        env['GEVAL_OGNI'] = str(args.ogni)
    if args.casuale:
        env['GEVAL_ORDINE'] = 'casuale'
    if args.riprova_errori:
        env['GEVAL_RIPROVA_ERRORI'] = '1'
    if args.solo_metriche:
        env['GEVAL_SOLO_METRICHE'] = '1'

    ok = esegui_notebook(NOTEBOOK, env)

    dopo = conta_cache(args.scope)
    log(f'Giudizi in cache: {prima:,} -> {dopo:,} (+{dopo - prima:,})')
    rapporto_costo(args.scope, valuta=args.valuta)

    if ok and not args.no_05 and args.pilota is None:
        log('--- Riesecuzione notebook 05 (viste di confronto aggiornate) ---')
        ok = esegui_notebook(NOTEBOOK_CONFRONTO) and ok

    if not ok:
        log('Corsa terminata con errori. Rilanciare questo script per riprendere: '
            'i giudizi gia\' in cache non vengono ripagati.')
        sys.exit(1)
    log('--- Corsa completata senza errori ---')


if __name__ == '__main__':
    main()
