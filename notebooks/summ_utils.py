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
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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

# Colonne opzionali del BERTScore (13_bertscore.ipynb, backfill separato):
# tenute FUORI da COLONNE_METRICHE cosi' i notebook 01-04/06-12, che non le
# calcolano, restano invariati.
COLONNE_METRICHE_BERTSCORE = ['bertscore_f1', 'bertscore_p', 'bertscore_r']

# Colonne del G-Eval (14_geval.ipynb). A differenza del BERTScore NON vengono
# mai unite al CSV per-esempio standard: finiscono in file dedicati scritti da
# `salva_metriche_geval`. Il motivo e' in `valuta_e_salva`, la cui media somma
# OGNI colonna su OGNI riga: il giudice LLM lascia scoperte alcune righe
# (content filter di Azure, parsing fallito) e la media esploderebbe.
COLONNE_METRICHE_GEVAL = ['geval_coherence', 'geval_consistency',
                          'geval_fluency', 'geval_relevance', 'geval_media']

DIMENSIONI_GEVAL = ['coherence', 'consistency', 'fluency', 'relevance']


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


def itera_split(path, split):
    """Itera in streaming le sole righe di complete.tab con lo split richiesto.

    Usato per l'ambito 'test' (notebook 03-04, 06-12 via SUMM_SCOPE o SCOPE): i
    row_id restano gli indici globali di complete.tab, coerenti con gli altri ambiti.
    """
    for es in itera_complete_tab(path):
        if es['split'] == split:
            yield es


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


def calcola_bertscore_batch(coppie, model_type='roberta-large', device=None, batch_size=64):
    """BERTScore per una lista di coppie (row_id, candidato, riferimento), in un unico batch.

    A differenza di `psr.bert_score()` di pyAutoSummarizer (che ricarica il modello
    ad OGNI chiamata — inutilizzabile su migliaia di esempi), qui il modello viene
    caricato UNA SOLA volta (`bert_score.BERTScorer`) e tutte le coppie vengono
    valutate in un'unica chiamata batched. Ritorna {row_id: {'bertscore_f1': ...,
    'bertscore_p': ..., 'bertscore_r': ...}}.
    """
    from bert_score import BERTScorer

    row_ids = [rid for rid, _, _ in coppie]
    candidati = [c for _, c, _ in coppie]
    riferimenti_testo = [r for _, _, r in coppie]

    scorer = BERTScorer(model_type=model_type, lang='en',
                        device=device or rileva_device(), batch_size=batch_size)
    p, r, f1 = scorer.score(candidati, riferimenti_testo, batch_size=batch_size)

    return {rid: {'bertscore_f1': float(f1[i]), 'bertscore_p': float(p[i]),
                 'bertscore_r': float(r[i])}
           for i, rid in enumerate(row_ids)}


def valuta_e_salva(riferimenti, riassunti, metodo, scope, metrics_dir, config,
                   extra_metriche=None, colonne_extra=None):
    """Calcola le metriche dai file salvati e scrive i due output.

    - `riferimenti`: iterabile di dict row_id/split/summary (campione o streaming
      di complete.tab) — vengono valutati solo i row_id presenti in `riassunti`
    - `riassunti`: dict {row_id: riassunto generato} (da carica_riassunti)
    - `extra_metriche`: opzionale, {row_id: {colonna: valore}} da unire alle
      metriche standard (usato da 13_bertscore.ipynb per il backfill del BERTScore,
      gia' calcolato altrove in batch — vedi `calcola_bertscore_batch`)
    - `colonne_extra`: opzionale, lista di nomi colonna corrispondenti a
      `extra_metriche`, aggiunta in coda a COLONNE_METRICHE per intestazione,
      scrittura e media
    - scrive {metodo}_{scope}_per_example.csv e {metodo}_{scope}_aggregate.json

    Ritorna (righe_per_esempio, aggregato).
    """
    valutatore = crea_valutatore()
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    extra_metriche = extra_metriche or {}
    colonne = COLONNE_METRICHE + (colonne_extra or [])

    righe = []
    for rif in riferimenti:
        row_id = rif['row_id']
        if row_id not in riassunti:
            continue
        m = metriche_esempio(valutatore, riassunti[row_id],
                             pulisci_riferimento(rif['summary']))
        m.update(extra_metriche.get(row_id, {}))
        righe.append({'row_id': row_id, 'split': rif['split'], **m})

    per_example_path = metrics_dir / f'{metodo}_{scope}_per_example.csv'
    with open(per_example_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row_id', 'split'] + colonne)
        for r in righe:
            writer.writerow([r['row_id'], r['split']] + [r[c] for c in colonne])

    def media(sottoinsieme):
        return {c: sum(r[c] for r in sottoinsieme) / len(sottoinsieme)
                for c in colonne}

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


# ---------------------------------------------------------------------------
# G-Eval (LLM-as-a-Judge) — vedi 14_geval.ipynb
# ---------------------------------------------------------------------------
#
# pyAutoSummarizer espone gia' un G-Eval (`psr.summarization.g_eval`, v1.2.0) ma
# non e' riutilizzabile qui, per quattro motivi verificati sul sorgente:
#   1. costruisce `openai.OpenAI(api_key=...)` SENZA `base_url` -> non raggiunge Azure;
#   2. ha `max_tokens=5` e `temperature=0.0` cablati (fatali per un modello reasoning);
#   3. legge la sorgente da `self.full_txt` -> servirebbe una `psr.summarization(sorgente)`
#      per ogni esempio, e il pattern a istanza fittizia di `crea_valutatore()` non regge;
#   4. fa una chiamata API per dimensione (4x il costo).
# Le rubriche restano VERBATIM (vedi PROMPT_GEVAL_ORIGINALI), cosi' la metrica e'
# difendibile come "il G-Eval di pyAutoSummarizer, reimplementato per un endpoint
# non-OpenAI". Riferimento: Liu et al., 2023 (G-Eval / GPTEval).

# Copia letterale dei quattro template di psr.summarization.g_eval (v1.2.0).
# Non vengono usati a runtime: servono da provenienza e da sorgente delle rubriche.
PROMPT_GEVAL_ORIGINALI = {
    'coherence': (
        'Rate the COHERENCE of the summary below on a scale of 1 to 5, '
        'where 1 = completely incoherent and 5 = perfectly coherent and well-structured. '
        'Reply with a single digit only.\n\n'
        'Source:\n{src}\n\nSummary:\n{summ}'
    ),
    'consistency': (
        'Rate the FACTUAL CONSISTENCY of the summary below with respect to the source '
        'on a scale of 1 to 5, where 1 = many hallucinations or contradictions and '
        '5 = fully faithful to the source. Reply with a single digit only.\n\n'
        'Source:\n{src}\n\nSummary:\n{summ}'
    ),
    'fluency': (
        'Rate the FLUENCY and grammaticality of the summary below on a scale of 1 to 5, '
        'where 1 = unreadable and 5 = fluent and error-free. '
        'Reply with a single digit only.\n\nSummary:\n{summ}'
    ),
    'relevance': (
        'Rate the RELEVANCE of the summary below to the source text on a scale of 1 to 5, '
        'where 1 = captures nothing important and 5 = captures all key information. '
        'Reply with a single digit only.\n\n'
        'Source:\n{src}\n\nSummary:\n{summ}'
    ),
}

# Le rubriche vere e proprie, RICAVATE dai template originali tagliando la coda
# "Reply with a single digit only." (qui il formato di risposta e' un JSON unico)
# e il template Source/Summary, che nel prompt combinato sta nel messaggio user.
# Derivarle invece di ricopiarle rende la fedelta' al testo originale meccanica.
RUBRICHE_GEVAL = {
    dim: testo.split('Reply with a single digit only.')[0].strip()
    for dim, testo in PROMPT_GEVAL_ORIGINALI.items()
}

# Messaggio di sistema: IDENTICO su tutte le chiamate, quindi sempre in cache.
PROMPT_GEVAL_SISTEMA = (
    'You are an expert evaluator of automatically generated news summaries.\n'
    'Rate the summary provided by the user on each of the following four '
    'dimensions, on a scale of 1 to 5.\n\n'
    + '\n\n'.join(f'{dim.upper()}\n{RUBRICHE_GEVAL[dim]}' for dim in DIMENSIONI_GEVAL)
    + '\n\nReturn a single JSON object with exactly the four integer keys '
      '"coherence", "consistency", "fluency", "relevance". No other text.'
)

# Schema per le structured output. In modalita' `strict` OpenAI/Azure non accetta
# `minimum`/`maximum`: il vincolo 1-5 si esprime con `enum`.
SCHEMA_GEVAL = {
    'type': 'object',
    'properties': {dim: {'type': 'integer', 'enum': [1, 2, 3, 4, 5]}
                   for dim in DIMENSIONI_GEVAL},
    'required': list(DIMENSIONI_GEVAL),
    'additionalProperties': False,
}

# Troncamento della sorgente: 3.500 parole (~4.700 token) lasciano INTERO il
# 91,5% dei cluster della split test (p90 = 3.244 parole, p95 = 4.499) tenendo
# comunque un tetto sui casi estremi, che arrivano a 35.362 parole. Il costo
# marginale rispetto a un tetto piu' basso e' contenuto perche' quasi tutte le
# parole in piu' finiscono nel prefisso condiviso, pagato a tariffa cached:
# alzarlo da 2.100 a 3.500 sposta la copertura dal 76% al 91,5% e costa ~$5 in
# piu' sull'intera corsa. E' comunque molto piu' di quanto vedano BART/PEGASUS
# (1.024 token). Nota: la prompt cache di Azure non si attiva sotto i 1.024
# token di prefisso, quindi le righe con sorgente molto corta non ne beneficiano
# comunque, tetto o non tetto.
MAX_PAROLE_SORGENTE_GEVAL = 3500

# Prezzi unitari ($ per 1M token) del giudice gpt-5.4-mini, SKU GlobalStandard,
# verificati sulla Azure Retail Prices API. Usati come fallback quando la
# chiamata a `prezzi_retail_azure()` non riesce (rete assente, meter rinominati).
#
# ATTENZIONE VALUTA: la Retail Prices API, senza `currencyCode`, ritorna prezzi
# in USD — ma NON e' detto che la sottoscrizione Azure fatturi in USD. Verificato
# su questo account (fatturazione EUR): il listino EUR non e' una conversione al
# cambio del momento, e' un listino fisso, ~0,8776x il numero USD su OGNI meter
# (input/cached/output). La prima corsa completa e' stata tracciata in "$" con
# il listino USD (PREZZI_GEVAL sotto), ma il consumo REALE di credito, verificato
# sulla sottoscrizione 24h dopo, e' stato in EUR: usare `valuta='EUR'` in
# `prezzi_retail_azure()` per un tracciamento accurato da qui in avanti.
PREZZI_GEVAL = {'input': 0.75, 'cached': 0.075, 'output': 4.50}
PREZZI_GEVAL_EUR = {'input': 0.6582, 'cached': 0.0658, 'output': 3.9491}


def tronca_parole(testo, max_parole=MAX_PAROLE_SORGENTE_GEVAL):
    """Tronca `testo` alle prime `max_parole` parole (split su whitespace)."""
    parole = testo.split()
    if len(parole) <= max_parole:
        return testo
    return ' '.join(parole[:max_parole])


def costruisci_messaggi_geval(sorgente, riassunto, max_parole=MAX_PAROLE_SORGENTE_GEVAL):
    """Messaggi chat per UN giudizio G-Eval (tutte e quattro le dimensioni insieme).

    L'ordine e' un contratto, non uno stile: il `system` e' identico su tutte le
    chiamate e il `user` comincia con la sorgente troncata — la stessa per tutti
    e 13 i metodi di una riga — mettendo in fondo il riassunto, unica parte che
    cambia. Cosi' il primo giudizio della riga popola la prompt cache di Azure e
    gli altri dodici la riusano. Riordinare i messaggi azzera il risparmio.
    """
    return [
        {'role': 'system', 'content': PROMPT_GEVAL_SISTEMA},
        {'role': 'user',
         'content': f'Source:\n{tronca_parole(sorgente, max_parole)}\n\nSummary:\n{riassunto}'},
    ]


def estrai_punteggi_geval(testo):
    """Punteggi 1-5 dal JSON del giudice -> {'geval_coherence': int, ..., 'geval_media': float}.

    Tollera i code fence markdown e il testo attorno all'oggetto JSON. Solleva
    ValueError su JSON malformato, dimensioni mancanti o valori fuori da [1,5]:
    meglio registrare un errore che inventare un punteggio.
    """
    if not testo or not testo.strip():
        raise ValueError('risposta vuota dal giudice')
    grezzo = testo.strip()
    inizio, fine = grezzo.find('{'), grezzo.rfind('}')
    if inizio < 0 or fine <= inizio:
        raise ValueError(f'nessun oggetto JSON nella risposta: {grezzo[:120]!r}')
    try:
        dati = json.loads(grezzo[inizio:fine + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f'JSON non decodificabile ({exc}): {grezzo[:120]!r}') from exc
    if not isinstance(dati, dict):
        raise ValueError(f'JSON non e\' un oggetto: {grezzo[:120]!r}')

    punteggi = {}
    for dim in DIMENSIONI_GEVAL:
        if dim not in dati:
            raise ValueError(f'dimensione mancante: {dim} (ricevuto {sorted(dati)})')
        try:
            valore = int(dati[dim])  # accetta anche "4" come stringa
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{dim} non e\' un intero: {dati[dim]!r}') from exc
        if not 1 <= valore <= 5:
            raise ValueError(f'{dim} fuori scala 1-5: {valore}')
        punteggi[f'geval_{dim}'] = valore
    punteggi['geval_media'] = sum(punteggi.values()) / len(DIMENSIONI_GEVAL)
    return punteggi


def estrai_uso(uso):
    """Conteggi di token da un oggetto `usage` dell'SDK openai (o da un dict).

    Normalizza le due forme in cui li leggiamo: la risposta API durante la corsa
    e la riga di cache JSONL quando si ricalcola il costo offline.
    """
    if uso is None:
        return {'prompt_tokens': 0, 'cached_tokens': 0,
                'completion_tokens': 0, 'reasoning_tokens': 0}
    if isinstance(uso, dict):
        return {k: int(uso.get(k) or 0) for k in
                ('prompt_tokens', 'cached_tokens', 'completion_tokens', 'reasoning_tokens')}
    dettagli_p = getattr(uso, 'prompt_tokens_details', None)
    dettagli_c = getattr(uso, 'completion_tokens_details', None)
    return {
        'prompt_tokens': int(getattr(uso, 'prompt_tokens', 0) or 0),
        'cached_tokens': int(getattr(dettagli_p, 'cached_tokens', 0) or 0),
        'completion_tokens': int(getattr(uso, 'completion_tokens', 0) or 0),
        'reasoning_tokens': int(getattr(dettagli_c, 'reasoning_tokens', 0) or 0),
    }


def costo_da_token(totali, prezzi=None):
    """Costo in $ da un dict di conteggi token (vedi `estrai_uso`).

    I token di input in cache si pagano alla tariffa ridotta; `completion_tokens`
    include gia' i reasoning token, che quindi non vanno sommati a parte.
    """
    prezzi = prezzi or PREZZI_GEVAL
    cached = totali.get('cached_tokens', 0)
    non_cached = max(totali.get('prompt_tokens', 0) - cached, 0)
    return {
        'input': non_cached / 1e6 * prezzi['input'],
        'cached': cached / 1e6 * prezzi['cached'],
        'output': totali.get('completion_tokens', 0) / 1e6 * prezzi['output'],
    }


def prezzi_retail_azure(modello='5.4 mini', regione_meter='Gl', valuta='USD', timeout=15):
    """Prezzi unitari (1M token, nella VALUTA scelta) dalla Azure Retail Prices API
    (pubblica, senza auth).

    `valuta` va fatta corrispondere alla valuta di fatturazione della propria
    sottoscrizione, non lasciata al default USD: il listino EUR di Azure NON e'
    una conversione al cambio del momento del listino USD, e' un listino fisso
    a se stante (verificato: ~0,8776x il numero USD su ogni meter). Usare la
    valuta sbagliata non altera il conteggio dei token ma fa leggere un importo
    che non e' quello davvero addebitato sulla sottoscrizione.

    I meter hanno nomi molto abbreviati ('5.4 mini Inp Gl 1M Tokens',
    '5.4 mini cd Inp Gl 1M Tokens', '5.4 mini Opt Gl 1M Tokens') e convivono con
    un tier `pp` (priority) piu' caro e con i meter Batch e fine-tuning, che qui
    vanno esclusi. In caso di errore ritorna il fallback fisso nella valuta
    richiesta (PREZZI_GEVAL per USD, PREZZI_GEVAL_EUR per EUR): una corsa da ore
    non puo' dipendere dalla raggiungibilita' di un listino esterno.
    """
    import urllib.parse
    import urllib.request

    fallback = PREZZI_GEVAL_EUR if valuta.upper() == 'EUR' else PREZZI_GEVAL
    filtro = (f"contains(productName,'Azure OpenAI') and "
              f"contains(meterName,'{modello}')")
    url = ('https://prices.azure.com/api/retail/prices?'
           + urllib.parse.urlencode({'$filter': filtro, 'currencyCode': valuta.upper()}))
    attesi = {
        'input':  f'{modello} Inp {regione_meter} 1M Tokens',
        'cached': f'{modello} cd Inp {regione_meter} 1M Tokens',
        'output': f'{modello} Opt {regione_meter} 1M Tokens',
    }
    try:
        with urllib.request.urlopen(url, timeout=timeout) as risposta:
            voci = json.loads(risposta.read().decode('utf-8')).get('Items', [])
        prezzi = {}
        for chiave, nome in attesi.items():
            candidati = {v['unitPrice'] for v in voci if v.get('meterName') == nome}
            if len(candidati) != 1:
                raise ValueError(f'meter {nome!r}: {len(candidati)} prezzi distinti')
            prezzi[chiave] = float(candidati.pop())
        print(f'Prezzi da Azure Retail Prices API ({valuta.upper()}/1M token): {prezzi}')
        return prezzi
    except Exception as exc:  # rete, meter rinominati, formato cambiato
        print(f'Listino Azure non raggiungibile ({exc!r}); uso i prezzi fissati '
              f'({valuta.upper()}): {fallback}')
        return dict(fallback)


def errore_permanente(exc):
    """True se non ha senso ritentare `exc`.

    Il content filter di Azure respinge alcuni cluster in modo DETERMINISTICO
    (139 righe su 5.610 nella corsa gpt5mini del notebook 12): ritentarli a ogni
    rilancio brucerebbe chiamate a pagamento senza mai riuscire.
    """
    codice = getattr(exc, 'status_code', None)
    if codice in (400, 401, 403, 404):
        return True
    testo = repr(exc).lower()
    return 'content_filter' in testo or 'content management policy' in testo


def tipo_errore(exc):
    """Etichetta breve per raggruppare gli errori nelle diagnostiche."""
    testo = repr(exc).lower()
    if 'content_filter' in testo or 'content management policy' in testo:
        return 'content_filter'
    codice = getattr(exc, 'status_code', None)
    if codice:
        return f'http_{codice}'
    return type(exc).__name__


def _attesa_suggerita(exc):
    """Valore dell'header Retry-After, se il server lo fornisce."""
    risposta = getattr(exc, 'response', None)
    intestazioni = getattr(risposta, 'headers', None)
    if not intestazioni:
        return None
    try:
        return float(intestazioni.get('retry-after'))
    except (TypeError, ValueError):
        return None


def chiama_con_backoff(chiamata, tentativi=6, attesa=2.0, fattore=2.0, massimo=60.0):
    """Riprova `chiamata()` con backoff esponenziale + jitter sugli errori transitori.

    Rispetta l'header `Retry-After` quando c'e' (429 di Azure) e rilancia subito
    gli errori permanenti (`errore_permanente`). E' il primo helper di retry del
    repository: i notebook 07-09/12 fanno chiamate singole e sequenziali e non ne
    avevano bisogno, mentre qui si va in concorrenza contro una quota TPM.
    """
    ritardo = attesa
    for tentativo in range(1, tentativi + 1):
        try:
            return chiamata()
        except Exception as exc:
            if errore_permanente(exc) or tentativo == tentativi:
                raise
            pausa = _attesa_suggerita(exc)
            if pausa is None:
                pausa = min(ritardo, massimo) * (0.5 + random.random())
            time.sleep(pausa)
            ritardo = min(ritardo * fattore, massimo)


class CacheGiudizi:
    """Cache su disco dei giudizi G-Eval: una riga JSON per coppia (metodo, row_id).

    Stesso contratto di ScrittoreRiassunti (append + flush a ogni scrittura, chiavi
    gia' presenti ricaricate alla costruzione = ripresa), ma indicizzata su una
    COPPIA e protetta da un lock perche' ci scrive un pool di thread: 72.930
    chiamate a pagamento non possono essere all-or-nothing in memoria come
    `calcola_bertscore_batch`.

    Ogni riga porta con se' anche i conteggi di token, cosi' il costo si ricalcola
    offline dalla sola cache (`scripts/run_geval.py --costo`) anche a corsa in
    corso. I giudizi FALLITI in modo permanente vengono scritti con `errore`
    valorizzato, cosi' una riesecuzione non li ripaga; `dimentica_errori()` li
    rimuove quando si vuole ritentare.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._voci = self._leggi(self.path)
        self._f = open(self.path, 'a', encoding='utf-8')

    @staticmethod
    def _leggi(path):
        voci = {}
        if not Path(path).exists():
            return voci
        with open(path, encoding='utf-8') as f:
            for riga in f:
                riga = riga.strip()
                if not riga:
                    continue
                voce = json.loads(riga)
                voci[(voce['metodo'], int(voce['row_id']))] = voce
        return voci

    def __len__(self):
        return len(self._voci)

    def gia_fatto(self, metodo, row_id):
        return (metodo, int(row_id)) in self._voci

    def scrivi(self, metodo, row_id, punteggi=None, uso=None, errore=None):
        voce = {'metodo': metodo, 'row_id': int(row_id),
                **{c: None for c in COLONNE_METRICHE_GEVAL},
                'errore': errore, **estrai_uso(uso)}
        if punteggi:
            voce.update(punteggi)
        with self._lock:
            self._voci[(metodo, int(row_id))] = voce
            self._f.write(json.dumps(voce, ensure_ascii=False) + '\n')
            self._f.flush()

    def per_metodo(self, metodo):
        """{row_id: {colonna: valore}} dei soli giudizi RIUSCITI del metodo."""
        return {rid: {c: voce[c] for c in COLONNE_METRICHE_GEVAL}
                for (m, rid), voce in self._voci.items()
                if m == metodo and voce.get('errore') is None}

    def errori(self, metodo=None):
        """[(metodo, row_id, errore), ...] dei giudizi in cache falliti."""
        return [(m, rid, voce['errore'])
                for (m, rid), voce in sorted(self._voci.items())
                if voce.get('errore') is not None and (metodo is None or m == metodo)]

    def dimentica_errori(self):
        """Rimuove i giudizi falliti e riscrive il file, per ritentarli."""
        with self._lock:
            self._voci = {k: v for k, v in self._voci.items()
                          if v.get('errore') is None}
            self._f.close()
            with open(self.path, 'w', encoding='utf-8') as f:
                for voce in self._voci.values():
                    f.write(json.dumps(voce, ensure_ascii=False) + '\n')
            self._f = open(self.path, 'a', encoding='utf-8')
        return len(self._voci)

    def totali_token(self):
        """Conteggi di token cumulati su tutta la cache, piu' il numero di giudizi."""
        totali = Counter()
        riusciti = 0
        for voce in self._voci.values():
            totali.update(estrai_uso(voce))
            riusciti += voce.get('errore') is None
        return {**{k: totali[k] for k in ('prompt_tokens', 'cached_tokens',
                                          'completion_tokens', 'reasoning_tokens')},
                'n_giudizi': len(self._voci), 'n_riusciti': riusciti}

    def chiudi(self):
        self._f.close()


class ContatoreCosti:
    """Accumulatore thread-safe di token e costo sui giudizi effettivamente pagati.

    Non esiste un'API Azure che riporti il costo in tempo reale (Cost Management
    ha 8-24 h di ritardo): la fonte di verita' e' l'oggetto `usage` di ogni
    risposta, che qui viene sommato e moltiplicato per i prezzi unitari. I
    reasoning token sono contati a parte perche' sono il termine di costo
    dominante e meno prevedibile.
    """

    def __init__(self, prezzi=None):
        self.prezzi = prezzi or PREZZI_GEVAL
        self._lock = threading.Lock()
        self.totali = Counter()
        self.giudizi = 0
        self.t0 = time.time()

    def aggiungi(self, uso):
        conteggi = estrai_uso(uso)
        with self._lock:
            self.totali.update(conteggi)
            self.giudizi += 1

    def riepilogo(self):
        durata = max(time.time() - self.t0, 1e-9)
        totali = {k: self.totali[k] for k in ('prompt_tokens', 'cached_tokens',
                                              'completion_tokens', 'reasoning_tokens')}
        costi = costo_da_token(totali, self.prezzi)
        totale = sum(costi.values())
        return {
            'giudizi': self.giudizi,
            'durata_s': durata,
            'giudizi_al_s': self.giudizi / durata,
            'tpm_osservato': (totali['prompt_tokens'] + totali['completion_tokens']) / durata * 60,
            **totali,
            'quota_cached': totali['cached_tokens'] / max(totali['prompt_tokens'], 1),
            'quota_reasoning': totali['reasoning_tokens'] / max(totali['completion_tokens'], 1),
            'costo': costi,
            'costo_totale': totale,
            'costo_per_giudizio': totale / max(self.giudizi, 1),
        }

    def proiezione(self, rimanenti):
        """Costo e tempo stimati a fine corsa, estrapolando i giudizi gia' pagati."""
        r = self.riepilogo()
        return {
            'costo_finale': r['costo_totale'] + r['costo_per_giudizio'] * rimanenti,
            'ore_rimanenti': rimanenti / max(r['giudizi_al_s'], 1e-9) / 3600,
        }

    def stampa(self, fatti, saltati, rimanenti, errori=None, prefisso=''):
        r = self.riepilogo()
        p = self.proiezione(rimanenti)
        c = r['costo']
        print(f"{prefisso}--- {fatti} giudizi pagati ({saltati} da cache, "
              f"{rimanenti} rimasti) ---")
        print(f"{prefisso}  ritmo   : {r['giudizi_al_s']:.2f} giudizi/s, "
              f"TPM osservato {r['tpm_osservato']:,.0f}, "
              f"ETA {p['ore_rimanenti']:.1f} h")
        print(f"{prefisso}  input   : {r['prompt_tokens']:,} token "
              f"({r['quota_cached']:.0%} in cache)")
        print(f"{prefisso}  output  : {r['completion_tokens']:,} token "
              f"({r['quota_reasoning']:.0%} di reasoning)")
        print(f"{prefisso}  costo   : ${r['costo_totale']:.2f} "
              f"(input ${c['input']:.2f} + cached ${c['cached']:.2f} + "
              f"output ${c['output']:.2f}) = ${r['costo_per_giudizio']:.5f}/giudizio")
        print(f"{prefisso}  PROIEZIONE a fine corsa: ${p['costo_finale']:.2f}")
        if errori:
            print(f"{prefisso}  errori  : " +
                  ', '.join(f'{k}={v}' for k, v in sorted(errori.items())))


def giudica_geval_concorrente(righe, giudica_uno, cache, n_thread=8, etichetta='',
                              ogni=1500, prezzi=None, budget_massimo=None,
                              speso_precedente=0.0, propaga_content_filter=True):
    """Esegue i giudizi G-Eval con un pool di thread, saltando quelli gia' in cache.

    - `righe`: LISTA di (row_id, sorgente, [(metodo, riassunto), ...]), ordinata per
      row_id. L'unita' di lavoro e' la RIGA: i suoi giudizi girano IN SEQUENZA nello
      stesso thread, cosi' il primo popola la prompt cache di Azure con il prefisso
      sistema+sorgente e gli altri la riusano; il parallelismo e' TRA righe diverse.
      Parallelizzare per singolo giudizio farebbe partire insieme le 13 chiamate
      della stessa riga, mancando la cache tutte quante.
    - `giudica_uno(sorgente, riassunto) -> (punteggi, uso)`: una chiamata API.
    - `cache`: CacheGiudizi aperta; ogni esito viene scritto e flushato subito.
    - `ogni`: ogni quanti giudizi pagati stampare il blocco costo/avanzamento.
    - `budget_massimo`: tetto di spesa COMPLESSIVO in $, non per singola sessione.
      Al superamento la corsa si ferma in modo pulito (la cache e' gia' su disco:
      basta rilanciare con un tetto piu' alto).
    - `speso_precedente`: quanto e' gia' costato quello che sta in cache. Va
      sommato al costo di questa sessione prima di confrontarlo con
      `budget_massimo`, altrimenti ogni rilancio ripartirebbe da zero e un tetto
      di $38 diventerebbe $38 A CORSA invece che $38 in totale.
    - `propaga_content_filter`: la sorgente troncata e' identica per tutti i metodi
      di una riga, quindi se il filtro contenuti respinge il primo giudizio
      respingerebbe anche gli altri: li si marca subito invece di pagarli.

    Ritorna un dict di riepilogo (fatti, saltati, errori, token, costo, durata).
    """
    contatore = ContatoreCosti(prezzi)
    totale_giudizi = sum(len(coppie) for _, _, coppie in righe)
    stato = {'fatti': 0, 'saltati': 0, 'errori': Counter(),
             'fermato': False, 'prossimo_report': ogni}
    lock = threading.Lock()

    def controlla_budget():
        """Ferma la corsa se il totale (cache + questa sessione) supera il tetto.

        Va chiamata dopo OGNI giudizio, non a fine riga: altrimenti la granularita'
        sarebbe di 13 chiamate per riga, moltiplicate per le righe in volo.
        """
        if budget_massimo is None or stato['fermato']:
            return
        speso = speso_precedente + contatore.riepilogo()['costo_totale']
        if speso >= budget_massimo:
            stato['fermato'] = True
            print(f'{etichetta}BUDGET COMPLESSIVO di ${budget_massimo:.2f} raggiunto '
                  f'(${speso_precedente:.2f} gia\' in cache + '
                  f'${speso - speso_precedente:.2f} in questa sessione): arresto '
                  f'pulito, nulla di pagato viene perso (rilanciare con un tetto '
                  f'piu\' alto per riprendere).')

    controlla_budget()   # gia' oltre il tetto? non spendere nemmeno una chiamata

    def lavora_riga(riga):
        row_id, sorgente, coppie = riga
        locali = {'fatti': 0, 'saltati': 0, 'errori': Counter()}
        for metodo, riassunto in coppie:
            if stato['fermato']:
                break
            if cache.gia_fatto(metodo, row_id):
                locali['saltati'] += 1
                continue
            try:
                punteggi, uso = giudica_uno(sorgente, riassunto)
                cache.scrivi(metodo, row_id, punteggi=punteggi, uso=uso)
                contatore.aggiungi(uso)
                locali['fatti'] += 1
                controlla_budget()
            except Exception as exc:
                etichetta_err = tipo_errore(exc)
                locali['errori'][etichetta_err] += 1
                if errore_permanente(exc):
                    cache.scrivi(metodo, row_id, errore=repr(exc))
                if etichetta_err == 'content_filter' and propaga_content_filter:
                    for altro, _ in coppie:
                        if not cache.gia_fatto(altro, row_id):
                            cache.scrivi(altro, row_id,
                                         errore='content_filter (propagato: stessa sorgente)')
                            locali['errori']['content_filter_propagato'] += 1
                    break
        return locali

    def assorbi(locali):
        with lock:
            stato['fatti'] += locali['fatti']
            stato['saltati'] += locali['saltati']
            stato['errori'].update(locali['errori'])
            if stato['fatti'] >= stato['prossimo_report']:
                stato['prossimo_report'] = stato['fatti'] + ogni
                rimasti = totale_giudizi - stato['fatti'] - stato['saltati']
                contatore.stampa(stato['fatti'], stato['saltati'], rimasti,
                                 stato['errori'], prefisso=etichetta)
        controlla_budget()

    with ThreadPoolExecutor(max_workers=n_thread) as pool:
        iteratore = iter(righe)
        pendenti = set()

        def rifornisci(quante):
            for _ in range(quante):
                try:
                    pendenti.add(pool.submit(lavora_riga, next(iteratore)))
                except StopIteration:
                    return

        rifornisci(2 * n_thread)
        while pendenti:
            concluse, pendenti = wait(pendenti, return_when=FIRST_COMPLETED)
            for futuro in concluse:
                assorbi(futuro.result())
            if not stato['fermato']:
                rifornisci(len(concluse))

    riepilogo = contatore.riepilogo()
    rimasti = totale_giudizi - stato['fatti'] - stato['saltati']
    contatore.stampa(stato['fatti'], stato['saltati'], rimasti, stato['errori'],
                     prefisso=etichetta)
    return {'fatti': stato['fatti'], 'saltati': stato['saltati'],
            'rimasti': rimasti, 'totale_giudizi': totale_giudizi,
            'fermato_per_budget': stato['fermato'],
            'errori': dict(stato['errori']), **riepilogo}


def salva_metriche_geval(punteggi, metodo, scope, metrics_dir, config, diagnostica=None):
    """Scrive i due file G-Eval del metodo, SEPARATI dalle metriche standard.

    - `punteggi`: {row_id: {colonna: valore}} dei soli giudizi riusciti
      (da `CacheGiudizi.per_metodo`)
    - scrive {metodo}_{scope}_geval_per_example.csv (intestazione `row_id` +
      COLONNE_METRICHE_GEVAL; nessuna colonna `split`, lo scope `test` e'
      monosplit) e {metodo}_{scope}_geval_aggregate.json

    NON tocca {metodo}_{scope}_per_example.csv ne' ..._aggregate.json: le righe
    senza giudizio romperebbero la media di `valuta_e_salva`, che somma OGNI
    colonna su OGNI riga, e riscrivere quei file cambierebbe medie e n_esempi
    gia' pubblicati delle altre metriche.

    Ritorna (righe_per_esempio, aggregato).
    """
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    righe = [{'row_id': rid, **punteggi[rid]} for rid in sorted(punteggi)]

    per_example_path = metrics_dir / f'{metodo}_{scope}_geval_per_example.csv'
    with open(per_example_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row_id'] + COLONNE_METRICHE_GEVAL)
        for r in righe:
            writer.writerow([r['row_id']] + [r[c] for c in COLONNE_METRICHE_GEVAL])

    aggregato = {
        'metodo': metodo,
        'scope': scope,
        'config': config,
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'n_esempi': len(righe),
        'overall': ({c: sum(r[c] for r in righe) / len(righe)
                     for c in COLONNE_METRICHE_GEVAL} if righe else {}),
        'diagnostica': diagnostica or {},
    }
    aggregate_path = metrics_dir / f'{metodo}_{scope}_geval_aggregate.json'
    with open(aggregate_path, 'w', encoding='utf-8') as f:
        json.dump(aggregato, f, indent=2, ensure_ascii=False)

    print(f'G-Eval per-esempio : {per_example_path} ({len(righe)} righe)')
    print(f'G-Eval aggregato   : {aggregate_path}')
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
