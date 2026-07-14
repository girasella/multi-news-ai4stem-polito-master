# -*- coding: utf-8 -*-
"""Funzioni condivise dai notebook di benchmark della summarization (01-06).

Tutti i notebook dei metodi (TextRank, LexRank, BART, PEGASUS, PRIMERA) usano
le stesse routine per: caricare il campione, preparare i testi, scrivere i riassunti
generati in modo incrementale (con ripresa dopo interruzione) e calcolare le
metriche a partire dai file salvati — cosi' la valutazione e' identica e
ri-eseguibile senza rigenerare i riassunti.
"""

import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# I documenti piu' lunghi superano il limite di campo predefinito del modulo csv
csv.field_size_limit(2**31 - 1)

HEADER_LINES_TAB = 3          # intestazione formato Orange: nomi, tipi, flag
SEPARATORE_ARTICOLI = '|||||'  # separatore tra articoli dentro `document`

# Colonne del CSV per-esempio delle metriche, nell'ordine di scrittura
COLONNE_METRICHE = ['rouge1_f1', 'rouge1_p', 'rouge1_r',
                    'rouge2_f1', 'rouge2_p', 'rouge2_r',
                    'rougeL_f1', 'rougeL_p', 'rougeL_r',
                    'bleu', 'meteor', 'parole_generate']


# ---------------------------------------------------------------------------
# Percorsi e configurazione
# ---------------------------------------------------------------------------

def trova_base_dir():
    """Radice del progetto: la cartella sopra notebooks/ in locale, /content su Colab.

    Su Colab caricare accanto ai notebook i file necessari (campione TSV e, per
    SCOPE='full', data/tab/complete.tab) rispettando la stessa struttura di cartelle.
    """
    if 'google.colab' in sys.modules:
        return Path('/content')
    cwd = Path.cwd()
    return cwd.parent if cwd.name == 'notebooks' else cwd


def percorsi_standard(base_dir):
    """Dizionario dei percorsi usati da tutti i notebook."""
    return {
        'complete_tab': base_dir / 'data' / 'tab' / 'complete.tab',
        'sample_dir':   base_dir / 'results' / 'sample',
        'summaries_dir': base_dir / 'results' / 'summaries',
        'metrics_dir':  base_dir / 'results' / 'metrics',
    }


def rileva_device():
    """'cuda' se disponibile una GPU NVIDIA, altrimenti 'cpu'."""
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Caricamento dati
# ---------------------------------------------------------------------------

def itera_complete_tab(path):
    """Itera le righe dati di complete.tab come dict, in streaming (~658 MB).

    Ogni riga: {'row_id': int, 'split': str, 'document': str, 'summary': str}.
    """
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for _ in range(HEADER_LINES_TAB):
            next(reader)
        for i, row in enumerate(reader):
            document, summary, split = row
            yield {'row_id': i, 'split': split, 'document': document, 'summary': summary}


def carica_campione(path):
    """Carica il TSV del campione prodotto da 00_prepara_campione.ipynb."""
    esempi = []
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        assert header == ['row_id', 'split', 'document', 'summary'], f'Intestazione inattesa: {header}'
        for row in reader:
            esempi.append({'row_id': int(row[0]), 'split': row[1],
                           'document': row[2], 'summary': row[3]})
    return esempi


def prepara_documento(document):
    """Sostituisce il separatore ' ||||| ' tra articoli con un newline.

    Il token resterebbe altrimenti dentro le "frasi" viste dai summarizer,
    inquinando ranking e testo estratto.
    """
    return re.sub(r'\s*\|\|\|\|\|\s*', '\n', document).strip()


def pulisci_riferimento(summary):
    """Rimuove il trattino editoriale iniziale ('– ') dai riassunti di riferimento."""
    return summary.lstrip('–- ').strip()


# ---------------------------------------------------------------------------
# Scrittura incrementale dei riassunti generati (con ripresa)
# ---------------------------------------------------------------------------

def carica_riassunti(path):
    """Legge un TSV di riassunti gia' generati -> dict {row_id: riassunto}.

    File inesistente = nessun riassunto (ritorna dict vuoto): e' il caso della
    prima esecuzione.
    """
    fatti = {}
    path = Path(path)
    if not path.exists():
        return fatti
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # intestazione
        for row in reader:
            fatti[int(row[0])] = row[1]
    return fatti


class ScrittoreRiassunti:
    """Scrive i riassunti generati riga per riga (flush immediato).

    Se il file esiste gia', i row_id presenti vengono saltati: un'esecuzione
    interrotta riprende da dove era arrivata invece di ricominciare.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.completati = carica_riassunti(self.path)
        nuovo = not self.path.exists()
        self._f = open(self.path, 'a', encoding='utf-8', newline='')
        self._writer = csv.writer(self._f, delimiter='\t', quoting=csv.QUOTE_ALL)
        if nuovo:
            self._writer.writerow(['row_id', 'generated_summary'])
            self._f.flush()

    def gia_fatto(self, row_id):
        return row_id in self.completati

    def scrivi(self, row_id, riassunto):
        # Il riassunto e' testo libero: il modulo csv (QUOTE_ALL) gestisce
        # correttamente eventuali newline o tab incorporati
        self._writer.writerow([row_id, riassunto])
        self._f.flush()
        self.completati[row_id] = riassunto

    def chiudi(self):
        self._f.close()


def ciclo_summarization(esempi, scrittore, genera, limit=None, etichetta='',
                        prepara=prepara_documento):
    """Applica `genera(documento) -> riassunto` a ogni esempio, con ripresa e progresso.

    - `esempi`: iterabile di dict con chiavi row_id/document (campione o streaming full)
    - `scrittore`: ScrittoreRiassunti gia' aperto
    - `limit`: se impostato, si ferma dopo aver PROCESSATO limit esempi nuovi (smoke test)
    - `prepara`: pre-processing del documento prima di `genera`. Default:
      `prepara_documento` (separatore `|||||` -> newline). Il notebook 06 (PRIMERA)
      passa `str.strip` perche' deve vedere il separatore originale tra articoli.

    Gli errori su un singolo esempio vengono registrati e non fermano il ciclo.
    """
    t0 = time.time()
    processati, saltati, errori = 0, 0, []
    for es in esempi:
        if scrittore.gia_fatto(es['row_id']):
            saltati += 1
            continue
        if limit is not None and processati >= limit:
            break
        try:
            riassunto = genera(prepara(es['document']))
            scrittore.scrivi(es['row_id'], riassunto)
        except Exception as exc:  # difensivo: un esempio rotto non ferma la corsa
            errori.append((es['row_id'], repr(exc)))
            print(f"  ERRORE su row_id={es['row_id']}: {exc!r}")
        processati += 1
        if processati % 10 == 0 or processati <= 3:
            media = (time.time() - t0) / processati
            print(f'{etichetta}[{processati}] media {media:.1f} s/esempio '
                  f'(saltati {saltati} gia\' fatti)')
    print(f'{etichetta}Completato: {processati} nuovi, {saltati} saltati, '
          f'{len(errori)} errori, {time.time()-t0:.0f} s totali')
    return errori


# ---------------------------------------------------------------------------
# Metriche
# ---------------------------------------------------------------------------

def crea_valutatore():
    """Istanza di pyAutoSummarizer usata SOLO per le funzioni di metrica.

    Le metriche della libreria normalizzano i testi con le impostazioni di
    pre-processing dell'istanza: qui manteniamo stopword e numeri (come il
    ROUGE standard), minuscolizziamo e togliamo la punteggiatura. NOTA: il
    ROUGE-N della libreria usa insiemi di n-grammi UNICI (non conteggi
    "clipped" come il ROUGE ufficiale): i valori sono confrontabili tra i
    metodi di questo benchmark ma non identici a quelli della letteratura.
    """
    from pyAutoSummarizer.base import psr
    return psr.summarization('testo fittizio.', stop_words=[], lowercase=True,
                             rmv_accents=False, rmv_special_chars=True,
                             rmv_numbers=False, verbose=False)


def metriche_esempio(valutatore, generato, riferimento):
    """Tutte le metriche per una coppia (riassunto generato, riferimento)."""
    r1 = valutatore.rouge_N(generato, riferimento, n=1)
    r2 = valutatore.rouge_N(generato, riferimento, n=2)
    rl = valutatore.rouge_L(generato, riferimento)
    return {
        'rouge1_f1': r1[0], 'rouge1_p': r1[1], 'rouge1_r': r1[2],
        'rouge2_f1': r2[0], 'rouge2_p': r2[1], 'rouge2_r': r2[2],
        'rougeL_f1': rl[0], 'rougeL_p': rl[1], 'rougeL_r': rl[2],
        'bleu':   valutatore.bleu(generato, riferimento, n=4),
        'meteor': valutatore.meteor(generato, riferimento),
        'parole_generate': len(generato.split()),
    }


def valuta_e_salva(riferimenti, riassunti, metodo, scope, metrics_dir, config):
    """Calcola le metriche dai file salvati e scrive i due output.

    - `riferimenti`: iterabile di dict row_id/split/summary (campione o streaming
      di complete.tab) — vengono valutati solo i row_id presenti in `riassunti`
    - `riassunti`: dict {row_id: riassunto generato} (da carica_riassunti)
    - scrive {metodo}_{scope}_per_example.csv e {metodo}_{scope}_aggregate.json

    Ritorna (righe_per_esempio, aggregato).
    """
    valutatore = crea_valutatore()
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    righe = []
    for rif in riferimenti:
        row_id = rif['row_id']
        if row_id not in riassunti:
            continue
        m = metriche_esempio(valutatore, riassunti[row_id],
                             pulisci_riferimento(rif['summary']))
        righe.append({'row_id': row_id, 'split': rif['split'], **m})

    per_example_path = metrics_dir / f'{metodo}_{scope}_per_example.csv'
    with open(per_example_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row_id', 'split'] + COLONNE_METRICHE)
        for r in righe:
            writer.writerow([r['row_id'], r['split']] + [r[c] for c in COLONNE_METRICHE])

    def media(sottoinsieme):
        return {c: sum(r[c] for r in sottoinsieme) / len(sottoinsieme)
                for c in COLONNE_METRICHE}

    per_split = defaultdict(list)
    for r in righe:
        per_split[r['split']].append(r)

    aggregato = {
        'metodo': metodo,
        'scope': scope,
        'config': config,
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'n_esempi': len(righe),
        'overall': media(righe) if righe else {},
        'per_split': {s: {'n_esempi': len(rs), **media(rs)}
                      for s, rs in sorted(per_split.items())},
    }
    aggregate_path = metrics_dir / f'{metodo}_{scope}_aggregate.json'
    with open(aggregate_path, 'w', encoding='utf-8') as f:
        json.dump(aggregato, f, indent=2, ensure_ascii=False)

    print(f'Metriche per-esempio : {per_example_path} ({len(righe)} righe)')
    print(f'Metriche aggregate   : {aggregate_path}')
    return righe, aggregato


def mostra_esempi(riferimenti, riassunti, quanti=2, larghezza=500):
    """Stampa qualche coppia generato/riferimento per ispezione qualitativa."""
    mostrati = 0
    for rif in riferimenti:
        if rif['row_id'] not in riassunti:
            continue
        print(f"--- row_id={rif['row_id']} (split={rif['split']}) ---")
        print(f"GENERATO   : {riassunti[rif['row_id']][:larghezza]}")
        print(f"RIFERIMENTO: {pulisci_riferimento(rif['summary'])[:larghezza]}")
        print()
        mostrati += 1
        if mostrati >= quanti:
            break
