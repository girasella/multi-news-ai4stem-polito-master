"""Unattended driver for the SCOPE='test' benchmark stint.

Executes notebooks 03 (BART), 04 (PEGASUS), 07 (Qwen), 09 (Mistral), 08 (Gemma) and
06 (PRIMERA) — in that fastest-to-slowest order — with SUMM_SCOPE=test, so each generates
summaries and metrics for the full clean test split (5,610 rows) without any manual
babysitting between notebooks. TextRank/LexRank test-split metrics are derived by filtering
their already-committed SCOPE='full' per-example CSVs (no need to rerun them: metrics are
computed per example, so filtering is numerically identical to a fresh test-scope run).
Notebook 05 is re-executed at the end so the comparison views reflect the new results.

Usage (from the repo root or anywhere — paths are resolved relative to this script):

    python scripts/run_benchmark_test.py            # full run, all rows
    python scripts/run_benchmark_test.py --limit 2  # smoke test: 2 rows per method

A failed notebook is logged and does NOT stop the run — thanks to the shared resumable
generation loop (notebooks/summ_utils.py), simply re-running this script later completes
whatever is missing, skipping row_ids already written.

Progress: results/summaries/{method}_test.tsv row counts grow as each notebook runs;
a full log with per-notebook start/end times is appended to run_benchmark_test.log in the
repo root.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
NOTEBOOKS_DIR = REPO_ROOT / 'notebooks'
RESULTS_DIR = REPO_ROOT / 'results'
LOG_PATH = REPO_ROOT / 'run_benchmark_test.log'

sys.path.insert(0, str(NOTEBOOKS_DIR))
import summ_utils as su  # noqa: E402  (needs NOTEBOOKS_DIR on sys.path first)

# Fastest-to-slowest, estimated on a machine with a CUDA GPU and ollama already serving
# the three local models (see notebooks/README.md for per-method duration estimates).
NOTEBOOKS = [
    '03_bart.ipynb',
    '04_pegasus.ipynb',
    '07_qwen.ipynb',
    '09_mistral.ipynb',
    '08_gemma.ipynb',
    '06_primera.ipynb',
]

# Methods whose SCOPE='test' metrics are derived from an existing SCOPE='full' run
# instead of executing a notebook (TextRank/LexRank already cover the full dataset).
DERIVED_FROM_FULL = ['textrank', 'lexrank']

OLLAMA_TAGS_REQUIRED = [
    'qwen2.5:7b-instruct',
    'gemma4:latest',
    'mistral:7b-instruct-v0.3-q4_K_M',
]


def log(message):
    stamped = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}'
    print(stamped, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(stamped + '\n')


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight():
    """Fail fast on anything that would otherwise break hours into the stint."""
    problemi = []

    try:
        import torch
        if not torch.cuda.is_available():
            problemi.append('Nessuna GPU CUDA rilevata (torch.cuda.is_available() = False): '
                            'PRIMERA/PEGASUS/BART sarebbero impraticabili sull\'intera split test.')
    except ImportError:
        problemi.append('torch non importabile: installare requirements-notebooks.txt')

    try:
        with urllib.request.urlopen('http://localhost:11434/api/tags', timeout=5) as resp:
            tags = {m['name'] for m in json.load(resp)['models']}
        mancanti = [t for t in OLLAMA_TAGS_REQUIRED if t not in tags]
        if mancanti:
            problemi.append(f'Tag ollama mancanti: {mancanti} (disponibili: {sorted(tags)}). '
                            f'Eseguire "ollama pull <tag>" per ciascuno.')
    except (urllib.error.URLError, ConnectionError, OSError) as exc:
        problemi.append(f'ollama non raggiungibile su http://localhost:11434 ({exc}). '
                        'Eseguire "ollama serve".')

    complete_tab = REPO_ROOT / 'data' / 'tab' / 'complete.tab'
    if not complete_tab.exists():
        problemi.append(f'{complete_tab} non trovato (rigenerarlo con scripts/convert_to_tab.py).')

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
    log('Preflight OK: GPU, ollama, complete.tab, nbconvert tutti disponibili.')
    return True


# ---------------------------------------------------------------------------
# Esecuzione notebook
# ---------------------------------------------------------------------------

def esegui_notebook(nome, limit=None):
    """Esegue un notebook in-place via nbconvert con SUMM_SCOPE=test (e SUMM_LIMIT se dato)."""
    path = NOTEBOOKS_DIR / nome
    env = dict(os.environ, SUMM_SCOPE='test')
    if limit is not None:
        env['SUMM_LIMIT'] = str(limit)

    inizio = time.time()
    log(f'=== Avvio {nome} (SUMM_SCOPE=test{f", SUMM_LIMIT={limit}" if limit else ""}) ===')
    risultato = subprocess.run(
        [sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
         '--inplace', '--ExecutePreprocessor.timeout=-1', nome],
        cwd=NOTEBOOKS_DIR, env=env, capture_output=True, text=True)
    durata = time.time() - inizio

    if risultato.returncode != 0:
        log(f'*** ERRORE in {nome} dopo {durata:.0f}s (returncode={risultato.returncode}) ***')
        log(f'--- stderr (ultime 40 righe) ---')
        for riga in risultato.stderr.splitlines()[-40:]:
            log(f'    {riga}')
        return False

    if nome == '05_confronto.ipynb':
        log(f'=== Completato {nome} in {durata:.0f}s ===')
        return True

    metodo = nome.split('_', 1)[1].removesuffix('.ipynb').split('_')[0]
    tsv = RESULTS_DIR / 'summaries' / f'{metodo}_test.tsv'
    n_righe = len(su.carica_riassunti(tsv)) if tsv.exists() else 0
    log(f'=== Completato {nome} in {durata:.0f}s — {n_righe} riassunti in {tsv.name} ===')
    return True


# ---------------------------------------------------------------------------
# Derivazione metriche test per i metodi con corsa full gia' committata
# ---------------------------------------------------------------------------

def deriva_metriche_test(metodo):
    """Filtra {metodo}_full_per_example.csv sulla split test invece di rieseguire.

    Numericamente identico a una corsa test dedicata: le metriche sono per-esempio,
    quindi filtrare l'output di una corsa full equivale a valutare solo quelle righe.
    """
    metrics_dir = RESULTS_DIR / 'metrics'
    full_csv = metrics_dir / f'{metodo}_full_per_example.csv'
    full_json = metrics_dir / f'{metodo}_full_aggregate.json'
    if not full_csv.exists() or not full_json.exists():
        log(f'  {metodo}: metriche full non trovate ({full_csv.name}), salto la derivazione.')
        return

    import csv as csv_mod
    with open(full_csv, encoding='utf-8', newline='') as f:
        reader = csv_mod.reader(f)
        header = next(reader)
        righe = [r for r in reader if r[header.index('split')] == 'test']

    if not righe:
        log(f'  {metodo}: nessuna riga "test" in {full_csv.name}, salto.')
        return

    test_csv = metrics_dir / f'{metodo}_test_per_example.csv'
    with open(test_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv_mod.writer(f)
        writer.writerow(header)
        writer.writerows(righe)

    idx = {col: header.index(col) for col in su.COLONNE_METRICHE}
    medie = {col: sum(float(r[i]) for r in righe) / len(righe) for col, i in idx.items()}

    with open(full_json, encoding='utf-8') as f:
        full_agg = json.load(f)

    aggregato = {
        'metodo': metodo,
        'scope': 'test',
        'config': full_agg.get('config', {}),
        'timestamp': full_agg.get('timestamp'),
        'n_esempi': len(righe),
        'overall': medie,
        'per_split': {'test': {'n_esempi': len(righe), **medie}},
        'nota_derivazione': (f'Derivato filtrando {full_csv.name} sulla split test '
                             '(scripts/run_benchmark_test.py), non da una corsa dedicata.'),
    }
    test_json = metrics_dir / f'{metodo}_test_aggregate.json'
    with open(test_json, 'w', encoding='utf-8') as f:
        json.dump(aggregato, f, indent=2, ensure_ascii=False)

    log(f'  {metodo}: {len(righe)} righe derivate -> {test_csv.name}, {test_json.name}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--limit', type=int, default=None,
                        help='Limita ogni notebook a N esempi nuovi (smoke test end-to-end).')
    args = parser.parse_args()

    log(f'--- Avvio corsa benchmark test (limit={args.limit}) ---')
    if not preflight():
        log('Interruzione: risolvere i problemi sopra prima di rilanciare.')
        sys.exit(1)

    esiti = {}
    for nome in NOTEBOOKS:
        esiti[nome] = esegui_notebook(nome, limit=args.limit)

    log('--- Derivazione metriche test per textrank/lexrank (da corse full esistenti) ---')
    for metodo in DERIVED_FROM_FULL:
        deriva_metriche_test(metodo)

    log('--- Riesecuzione notebook 05 (viste di confronto aggiornate) ---')
    esiti['05_confronto.ipynb'] = esegui_notebook('05_confronto.ipynb')

    falliti = [n for n, ok in esiti.items() if not ok]
    log('--- Riepilogo ---')
    for nome, ok in esiti.items():
        log(f'  {"OK " if ok else "FALLITO"} {nome}')
    if falliti:
        log(f'{len(falliti)} notebook falliti: {falliti}. Rilanciare questo script per '
            'riprendere (il ciclo condiviso salta i row_id gia\' completati).')
        sys.exit(1)
    log('--- Corsa completata senza errori ---')


if __name__ == '__main__':
    main()
