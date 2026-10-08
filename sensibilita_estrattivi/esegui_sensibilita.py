#!/usr/bin/env python3
# coding=utf-8
"""Studio di sensibilita' degli iperparametri dei metodi estrattivi (vedi RIEPILOGO_sensibilita.md).

Rigenera i riassunti della split test (5.610 righe) a LUNGHEZZA DEL RIFERIMENTO (stesso
protocollo dell'ambito `test_budgetref`, issue #16) per una griglia di valori di un
iperparametro, poi calcola le metriche lessicali di pyAutoSummarizer (ROUGE-1/2/L, BLEU,
METEOR) sul testo post-tetto, come `scripts/applica_budget.py`. Niente BERTScore ne' G-Eval:
lo studio e' un'esplorazione qualitativa a basso costo.

Nessun file di `results/` viene scritto: tutto finisce in questa cartella.

Step implementati:
  centroid_mmr   lambda MMR per `centroid_mmr` (TF-IDF) e `centroid_mmr_bert` (MiniLM)
  lsa            K_LATENTE per `lsa` (top-k) e `lsa_steinberger` (deflazione)
  sbert          N_CLUSTER per `sbert_kmeans` e `sbert_agglom`, linkage per `sbert_agglom`
  lda            N_TOPICS x alpha (eta fisso) e seme per `lda`

Uso (da qualunque cartella):

    python sensibilita_estrattivi/esegui_sensibilita.py --metodo centroid_mmr --limit 20
    python sensibilita_estrattivi/esegui_sensibilita.py --metodo centroid_mmr
    python sensibilita_estrattivi/esegui_sensibilita.py --metodo centroid_mmr --solo-metriche

La generazione riprende da dove si era fermata (una riga e' fatta quando e' presente in tutti
i TSV della griglia). Le metriche si ricalcolano sempre da zero sui TSV salvati.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

CARTELLA = Path(__file__).resolve().parent
REPO_ROOT = CARTELLA.parent
sys.path.insert(0, str(REPO_ROOT / 'notebooks'))
import summ_utils as su  # noqa: E402  (serve notebooks/ su sys.path)

SCOPE = 'test' + su.SUFFISSO_BUDGET           # test_budgetref: stesse righe del test, budget attivo
RIASSUNTI_DIR = CARTELLA / 'riassunti'
METRICHE_DIR = CARTELLA / 'risultati' / 'metriche'
SELEZIONI_DIR = CARTELLA / 'risultati' / 'selezioni'
P = su.percorsi_standard(REPO_ROOT)


def slug_config(metodo, parametro, valore):
    """Nome di una configurazione: `centroid_mmr__lambda-0.7`."""
    return f'{metodo}__{parametro}-{valore}'


# =============================================================================
# Notebook 11 — Centroid+MMR
# =============================================================================
# Copia VERBATIM di segmenta / vettorizza_tfidf / vettorizza_bert / analizza_mmr dalla
# cella 4 di notebooks/11_centroid_mmr.ipynb, con le costanti del notebook. `analizza_mmr`
# serve da riferimento: il greedy usato nella griglia e' `ordine_mmr`, la stessa regola
# vettorizzata (il ciclo Python costa O(n*k^2) e qui va ripetuto 7 volte per variante),
# verificata uguale ad `analizza_mmr` riga per riga con --verifica-greedy.

N_SENTENCES = 11
LAMBDA = 0.7
K_ORDINE_BUDGET = 50
STOP_WORDS = ['en']
ST_MODEL = 'all-MiniLM-L6-v2'
SEED = 42

LAMBDA_GRIGLIA = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
METODO_TFIDF = 'centroid_mmr'
METODO_BERT = 'centroid_mmr_bert'

from pyAutoSummarizer.base import psr  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402


def segmenta(documento):
    """Frasi del cluster con la segmentazione di psr.summarization (come TextRank/LexRank).

    `s.original` e' la lista delle frasi originali in ordine di documento (split
    abbreviation-aware). Si scartano le stringhe vuote.
    """
    s = psr.summarization(documento, stop_words=STOP_WORDS)
    return [f.strip() for f in getattr(s, 'original', []) if f and f.strip()]


# --- Due vettorizzatori di frasi (unica differenza tra le varianti) ---------
def vettorizza_tfidf(frasi):
    """Matrice TF-IDF sparsa (n, vocab), righe L2-normalizzate."""
    return TfidfVectorizer(stop_words='english').fit_transform(frasi)


_bert = {}


def vettorizza_bert(frasi):
    """Embeddings Sentence-BERT densi (n, dim), L2-normalizzati. Modello caricato una volta."""
    if 'model' not in _bert:
        from sentence_transformers import SentenceTransformer
        _bert['model'] = SentenceTransformer(ST_MODEL, device=DEVICE)
    return _bert['model'].encode(frasi, normalize_embeddings=True, show_progress_bar=False)


VETTORIZZATORI = {METODO_TFIDF: vettorizza_tfidf, METODO_BERT: vettorizza_bert}


def analizza_mmr(frasi, vettorizza=vettorizza_tfidf, lam=None, k=None):
    """Esegue Centroid+MMR restituendo gli artefatti intermedi (per spiegazione/plot).

    Ritorna un dict con: matrice dei vettori `X` (TF-IDF o embeddings), `centroide`,
    `rilevanza` di ogni frase (cosine col centroide), matrice `sim` frase-frase e `scelti`
    (indici nell'ordine di selezione greedy). Con `vettorizza`/`lam`/`k` si riesegue sullo
    stesso pool con rappresentazione o parametri diversi (usato dalle figure di spiegazione).
    """
    lam = LAMBDA if lam is None else lam
    k = N_SENTENCES if k is None else k
    X = vettorizza(frasi)
    centroide = np.asarray(X.mean(axis=0)).reshape(1, -1)          # (1, dim)
    rilevanza = cosine_similarity(X, centroide).ravel()            # S(frase, centroide)
    sim = cosine_similarity(X)                                     # R(frase_i, frase_j), n x n

    scelti, candidati = [], list(range(len(frasi)))
    while len(scelti) < k and candidati:
        migliore, punteggio_migliore = None, -np.inf
        for i in candidati:
            ridondanza = max((sim[i][j] for j in scelti), default=0.0)
            punteggio = lam * rilevanza[i] - (1.0 - lam) * ridondanza
            if punteggio > punteggio_migliore:
                punteggio_migliore, migliore = punteggio, i
        scelti.append(migliore)
        candidati.remove(migliore)
    return {'X': X, 'centroide': centroide, 'rilevanza': rilevanza, 'sim': sim, 'scelti': scelti}


# --- Fine della copia verbatim ------------------------------------------------

def artefatti_mmr(frasi, vettorizza):
    """Rilevanza e similarita' di `analizza_mmr`, calcolate una volta per tutti i lambda."""
    X = vettorizza(frasi)
    centroide = np.asarray(X.mean(axis=0)).reshape(1, -1)
    return cosine_similarity(X, centroide).ravel(), cosine_similarity(X)


def ordine_mmr(rilevanza, sim, lam, k):
    """Greedy MMR di `analizza_mmr`, vettorizzato: stesso ordine, a parita' compresa.

    Il ciclo originale scorre i candidati in ordine di indice e tiene il primo punteggio
    STRETTAMENTE maggiore: e' il primo massimo, cioe' `np.argmax`. La ridondanza vale 0 finche'
    non c'e' nessuna frase scelta, poi e' il massimo delle similarita' con le scelte (che con
    BERT puo' essere negativo: per questo non si parte da 0 ma dalla prima colonna scelta).
    """
    n = len(rilevanza)
    scelti = []
    disponibile = np.ones(n, dtype=bool)
    ridondanza = np.zeros(n, dtype=sim.dtype)
    while len(scelti) < k and disponibile.any():
        punteggio = lam * rilevanza - (1.0 - lam) * ridondanza
        punteggio = np.where(disponibile, punteggio, -np.inf)
        migliore = int(np.argmax(punteggio))
        ridondanza = sim[:, migliore].copy() if not scelti else np.maximum(ridondanza, sim[:, migliore])
        scelti.append(migliore)
        disponibile[migliore] = False
    return scelti


def ridondanza_interna(sim, indici):
    """Similarita' coseno media fra le coppie di frasi scelte (None se meno di 2 frasi)."""
    indici = list(indici)
    if len(indici) < 2:
        return None
    sotto = sim[np.ix_(indici, indici)]
    return float(sotto[np.triu_indices(len(indici), k=1)].mean())


def configurazioni_centroid():
    return [(metodo, 'lambda', lam) for metodo in (METODO_TFIDF, METODO_BERT) for lam in LAMBDA_GRIGLIA]


def genera_centroid(documento, budget, verifica_greedy=False):
    """{(metodo, 'lambda', lam): (riassunto, indici scelti, ridondanza interna)} per una riga.

    Replica `make_genera(...)` del notebook 11 nell'ambito a budget: ordine MMR lungo
    min(n, K_ORDINE_BUDGET), taglio al budget, frasi in ordine di documento; vocabolario
    TF-IDF vuoto -> ordine = range(k).
    """
    frasi = segmenta(documento)
    k = min(len(frasi), K_ORDINE_BUDGET)
    uscita = {}
    for metodo in (METODO_TFIDF, METODO_BERT):
        try:
            rilevanza, sim = artefatti_mmr(frasi, VETTORIZZATORI[metodo])
        except ValueError:            # vocabolario TF-IDF vuoto su testi minimi
            rilevanza, sim = None, None
        for lam in LAMBDA_GRIGLIA:
            if rilevanza is None:
                ordine = list(range(k))
            else:
                ordine = ordine_mmr(rilevanza, sim, lam, k)
                if verifica_greedy:
                    atteso = analizza_mmr(frasi, vettorizza=VETTORIZZATORI[metodo], lam=lam, k=k)['scelti']
                    assert ordine == atteso, f'{metodo} lambda={lam}: greedy diverso da analizza_mmr'
            scelti = su.seleziona_per_budget(frasi, ordine, budget)
            testo = ' '.join(frasi[i] for i in sorted(scelti))
            rid = ridondanza_interna(sim, scelti) if sim is not None else None
            uscita[(metodo, 'lambda', lam)] = (testo, sorted(scelti), rid)
    return uscita


# =============================================================================
# Notebook 15 — LSA
# =============================================================================
# Copia VERBATIM di seleziona_topk / seleziona_deflazione dalla cella 4 di
# notebooks/15_lsa.ipynb (segmenta e' identica a quella del notebook 11, sopra). La
# pipeline di `make_genera` (TF-IDF -> k cappato al rango -> TruncatedSVD con random_state)
# e' replicata in `genera_lsa` con K_LATENTE variabile. La SVD si rifa' per ogni K: quella
# randomizzata di scikit-learn non da' per K piccolo le prime colonne della soluzione a K grande.

EPS = 1e-12   # soglia sotto la quale un vettore latente e' considerato esaurito
K_LATENTE = 11
K_GRIGLIA = [1, 2, 3, 5, 8, 11, 15, 20]
METODO_LSA = 'lsa'
METODO_STEIN = 'lsa_steinberger'

from sklearn.decomposition import TruncatedSVD  # noqa: E402


def seleziona_topk(Z, n_sentences):
    """Variante `lsa`: le n_sentences frasi con vettore latente piu' lungo.

    E' la regola di Gong & Liu / Steinberger senza update, la stessa di sumy.LsaSummarizer:
    nessuna memoria di cosa e' gia' stato scelto.
    """
    salienza = np.linalg.norm(Z, axis=1)
    return [int(i) for i in np.argsort(-salienza)[:n_sentences]]


def seleziona_deflazione(Z, n_sentences):
    """Variante `lsa_steinberger`: greedy con aggiornamento della matrice approssimata.

    Ad ogni passo si prende la frase con vettore residuo piu' lungo e si sottrae la sua
    direzione da tutti i vettori rimanenti (proiezione ortogonale). Il contenuto gia'
    coperto sparisce dal residuo, quindi una frase che ripete la precedente vale ~0 e non
    viene riscelta: e' l'anti-ridondanza della variante multi-documento di Steinberger,
    assente in sumy.

    Ritorna (scelti, storia) dove `storia[t]` sono le lunghezze residue di TUTTE le frasi
    prima della t-esima scelta (serve alle figure della sezione di spiegazione).
    """
    R = np.array(Z, dtype=float, copy=True)     # matrice residua, si consuma passo dopo passo
    scelti, storia = [], []
    for _ in range(min(n_sentences, len(R))):
        lunghezze = np.linalg.norm(R, axis=1)
        storia.append(lunghezze.copy())
        lunghezze[scelti] = -1.0                # una frase gia' scelta non torna candidata
        i = int(np.argmax(lunghezze))
        if lunghezze[i] <= EPS:                 # spazio residuo esaurito: niente da aggiungere
            break
        scelti.append(i)
        u = R[i] / np.linalg.norm(R[i])         # direzione appena "spesa"
        R -= np.outer(R @ u, u)                 # z_j <- z_j - (z_j . u) u, per ogni frase
    # se la deflazione si e' fermata presto, completo con la salienza ORIGINALE
    if len(scelti) < min(n_sentences, len(Z)):
        for i in np.argsort(-np.linalg.norm(Z, axis=1)):
            if int(i) not in scelti:
                scelti.append(int(i))
            if len(scelti) >= min(n_sentences, len(Z)):
                break
    return scelti, storia


SELETTORI_LSA = {METODO_LSA: lambda Z, n: seleziona_topk(Z, n),
                 METODO_STEIN: lambda Z, n: seleziona_deflazione(Z, n)[0]}

# --- Fine della copia verbatim ------------------------------------------------


def configurazioni_lsa():
    return [(metodo, 'k', k) for metodo in (METODO_LSA, METODO_STEIN) for k in K_GRIGLIA]


def genera_lsa(documento, budget, verifica_greedy=False):
    """{(metodo, 'k', K): (riassunto, indici scelti, ridondanza interna)} per una riga.

    Replica `make_genera(...)` del notebook 15 nell'ambito a budget con K_LATENTE = K.
    La ridondanza interna e' misurata nello spazio TF-IDF (righe L2-normalizzate, quindi
    X X^T = coseno), lo stesso per tutti i K, cosi' i valori sono confrontabili.
    """
    frasi = segmenta(documento)
    n_ordine = min(len(frasi), K_ORDINE_BUDGET)
    try:
        X = TfidfVectorizer(stop_words='english').fit_transform(frasi)
        sim = (X @ X.T).toarray()
    except ValueError:            # vocabolario TF-IDF vuoto su testi minimi
        X, sim = None, None
    uscita = {}
    for k_lat in K_GRIGLIA:
        Z = None
        if X is not None:
            k = max(1, min(k_lat, min(X.shape) - 1))
            Z = TruncatedSVD(n_components=k, random_state=SEED).fit_transform(X)
        for metodo in (METODO_LSA, METODO_STEIN):
            ordine = range(n_ordine) if Z is None else SELETTORI_LSA[metodo](Z, n_ordine)
            scelti = su.seleziona_per_budget(frasi, ordine, budget)
            testo = ' '.join(frasi[i] for i in sorted(scelti))
            rid = ridondanza_interna(sim, scelti) if sim is not None else None
            uscita[(metodo, 'k', k_lat)] = (testo, sorted(scelti), rid)
    return uscita


# =============================================================================
# Notebook 16 — clustering di embedding SBERT
# =============================================================================
# Copia VERBATIM di codifica / cluster_kmeans / cluster_agglomerativo /
# centroidi_e_medoidi / analizza_clustering / ordine_rappresentanti dalla cella 4 di
# notebooks/16_sbert_clustering.ipynb (segmenta e' quella del notebook 11, sopra).
# `cluster_agglomerativo` e' l'unica funzione con il parametro esposto (linkage), con
# default 'average' identico al notebook.

N_CLUSTER = 11
N_CLUSTER_GRIGLIA = [3, 5, 8, 11, 15, 20]
LINKAGE_GRIGLIA = ['average', 'complete', 'single']
METODO_KMEANS = 'sbert_kmeans'
METODO_AGGLOM = 'sbert_agglom'

from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: E402

_sbert = {}


def codifica(frasi):
    """Embedding SBERT delle frasi, (n, dim), L2-normalizzati. Modello caricato una volta.

    La normalizzazione rende il prodotto scalare uguale alla cosine similarity: tutte le
    "vicinanze" del notebook sono quindi la stessa cosa, misurata in un modo solo.
    """
    if 'model' not in _sbert:
        from sentence_transformers import SentenceTransformer
        _sbert['model'] = SentenceTransformer(ST_MODEL, device=DEVICE)
    return _sbert['model'].encode(frasi, normalize_embeddings=True, show_progress_bar=False)


# --- I due algoritmi di clustering (unica differenza tra le varianti) -------
def cluster_kmeans(E, k):
    """K-means sugli embedding -> etichette di cluster. Distanza euclidea."""
    return KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(E)


def cluster_agglomerativo(E, k, linkage='average'):
    """Clustering gerarchico agglomerativo, cosine + average linkage -> etichette."""
    return AgglomerativeClustering(n_clusters=k, metric='cosine',
                                   linkage=linkage).fit_predict(E)


def centroidi_e_medoidi(E, etichette):
    """Centroide di ogni cluster e frase piu' vicina a esso (medoide).

    Il centroide e' la media degli embedding del cluster, ri-normalizzata: e' un punto
    "medio" che non corrisponde ad alcuna frase reale. Il medoide e' la frase VERA piu'
    vicina a quel punto, cioe' il rappresentante piu' tipico del sottotema; e' quella che
    finisce nel riassunto (un riassunto estrattivo puo' solo copiare frasi esistenti).
    """
    centroidi, medoidi, vicinanze = {}, {}, {}
    for c in sorted(set(int(x) for x in etichette)):
        membri = np.where(etichette == c)[0]
        centro = E[membri].mean(axis=0)
        norma = np.linalg.norm(centro)
        centro = centro / norma if norma > 0 else centro
        sim = E[membri] @ centro               # cosine, gli embedding sono normalizzati
        centroidi[c] = centro
        medoidi[c] = int(membri[int(np.argmax(sim))])
        vicinanze[c] = sim
    return centroidi, medoidi, vicinanze


def analizza_clustering(frasi, algoritmo=None, k=None, E=None):
    """Pipeline completa su un pool di frasi, con tutti gli artefatti intermedi.

    Ritorna `E` (embedding), `etichette`, `centroidi`, `medoidi` ({cluster: indice frase}),
    `vicinanze` e `k`. `E` puo' essere passata dall'esterno per non ricodificare le stesse
    frasi quando si confrontano i due algoritmi (lo fa la sezione di spiegazione).
    """
    algoritmo = cluster_kmeans if algoritmo is None else algoritmo
    k = N_CLUSTER if k is None else k
    E = codifica(frasi) if E is None else E
    k = max(1, min(k, len(frasi)))
    etichette = algoritmo(E, k)
    centroidi, medoidi, vicinanze = centroidi_e_medoidi(E, etichette)
    return {'E': E, 'etichette': etichette, 'centroidi': centroidi, 'medoidi': medoidi,
            'vicinanze': vicinanze, 'k': k}


def ordine_rappresentanti(art):
    """Ordine di selezione a giri per l'ambito a budget.

    1o giro: il medoide di ogni cluster, cluster piu' grandi prima; 2o giro: la seconda
    frase piu' vicina al centroide di ogni cluster; ... Il primo giro coincide con la
    selezione storica (un medoide per cluster), quindi a budget "piccolo" il metodo
    e' lo stesso di sempre; a budget grande continua con i rappresentanti successivi
    dei sottotemi invece di fermarsi a 11 frasi.
    """
    per_cluster = {}
    for c, sim in art['vicinanze'].items():
        membri = np.where(art['etichette'] == c)[0]
        per_cluster[c] = [int(membri[j]) for j in np.argsort(-sim)]
    cluster_ordinati = sorted(per_cluster, key=lambda c: -len(per_cluster[c]))
    ordine, giro = [], 0
    while any(giro < len(per_cluster[c]) for c in cluster_ordinati):
        ordine += [per_cluster[c][giro] for c in cluster_ordinati if giro < len(per_cluster[c])]
        giro += 1
    return ordine

# --- Fine della copia verbatim ------------------------------------------------


def configurazioni_sbert():
    return ([(m, 'n_cluster', k) for m in (METODO_KMEANS, METODO_AGGLOM) for k in N_CLUSTER_GRIGLIA]
            + [(METODO_AGGLOM, 'linkage', l) for l in LINKAGE_GRIGLIA])


def genera_sbert(documento, budget, verifica_greedy=False):
    """{config: (riassunto, indici scelti, ridondanza interna)} per una riga.

    Replica `make_genera(...)` del notebook 16 nell'ambito a budget: un'unica codifica
    MiniLM per riga, poi un clustering per configurazione e l'ordine a giri tagliato al
    budget. La configurazione ('sbert_agglom', 'linkage', 'average') ripete N_CLUSTER = 11
    con nomi propri, cosi' lo studio del linkage ha il suo default. La ridondanza interna e'
    misurata nello spazio degli embedding (E E^T = coseno).
    """
    frasi = segmenta(documento)
    try:
        E = codifica(frasi)
        sim = E @ E.T
    except ValueError:
        E, sim = None, None
    algoritmi = {(METODO_KMEANS, 'n_cluster', k): (cluster_kmeans, k) for k in N_CLUSTER_GRIGLIA}
    algoritmi.update({(METODO_AGGLOM, 'n_cluster', k): (cluster_agglomerativo, k)
                      for k in N_CLUSTER_GRIGLIA})
    algoritmi.update({(METODO_AGGLOM, 'linkage', l):
                      (lambda E_, k_, l=l: cluster_agglomerativo(E_, k_, linkage=l), N_CLUSTER)
                      for l in LINKAGE_GRIGLIA})
    uscita = {}
    for config, (algoritmo, k) in algoritmi.items():
        try:
            if E is None:
                raise ValueError('embedding non disponibili')
            art = analizza_clustering(frasi, algoritmo=algoritmo, k=k, E=E)
            ordine = ordine_rappresentanti(art)
        except ValueError:    # clustering non applicabile (es. embedding degeneri/nulli)
            ordine = range(len(frasi))
        scelti = su.seleziona_per_budget(frasi, ordine, budget)
        testo = ' '.join(frasi[i] for i in sorted(scelti))
        rid = ridondanza_interna(sim, scelti) if sim is not None else None
        uscita[config] = (testo, sorted(scelti), rid)
    return uscita


# =============================================================================
# Notebook 17 — LDA
# =============================================================================
# Copia VERBATIM di seleziona_per_quote_parole dalla cella 4 di notebooks/17_lda.ipynb
# (segmenta e' quella del notebook 11). Nell'ambito a budget il notebook usa di
# `analizza_lda` solo `theta`, ottenuta da CountVectorizer -> LatentDirichletAllocation con
# k = max(1, min(N_TOPICS, n_frasi)) e random_state = SEED: `theta_lda` replica quella parte
# con k, alpha e seme espliciti.
#
# I priori. Con doc_topic_prior = topic_word_prior = None scikit-learn pone alpha = eta = 1/k,
# quindi nel notebook cambiare N_TOPICS cambia anche i priori. Nella griglia alpha e' esplicito
# ed eta e' fissato a 0,2 (= 1/5, il valore del default), cosi' N_TOPICS e alpha si separano.
# La configurazione di default (k = 5, alpha = 0,2) e lo studio del seme usano invece None/None
# come il notebook: e' identico a 0,2/0,2 salvo nelle righe con meno di 5 frasi, dove k viene
# cappato e 1/k cambia.

N_TOPICS = 5
N_TOPICS_GRIGLIA = [2, 3, 5, 8, 11]
ALFA_GRIGLIA = [0.05, 0.2, 0.5, 1.0]
ETA_FISSO = 0.2
SEMI = [42, 0, 1, 2, 3]
METODO_LDA = 'lda'

from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from sklearn.decomposition import LatentDirichletAllocation  # noqa: E402


def seleziona_per_quote_parole(frasi, theta, budget_parole):
    """Selezione LDA a budget di PAROLE: quota proporzionale al peso di ogni topic.

    Per ogni topic (dal piu' pesante) si scorrono le frasi per appartenenza decrescente
    accumulando parole fino alla quota del topic; la frase che sfora entra se ne entra
    piu' della meta'. Se, per frasi gia' prese da altri topic, il totale resta sotto
    l'80% del budget, si completa con le frasi piu' salienti (max di theta) fino al
    budget residuo, come fa gia' la versione a frasi.
    """
    peso = theta.sum(axis=0)
    quote = (budget_parole * peso / peso.sum() if peso.sum() > 0
             else np.full(len(peso), budget_parole / len(peso)))
    scelti, gia = [], set()
    for t in np.argsort(-peso):
        totale = 0
        for i in np.argsort(-theta[:, t]):
            i = int(i)
            if i in gia:
                continue
            n = len(frasi[i].split())
            if totale + n <= quote[t]:
                scelti.append(i)
                gia.add(i)
                totale += n
                continue
            if quote[t] - totale > n / 2:
                scelti.append(i)
                gia.add(i)
            break
    parole_scelte = sum(len(frasi[i].split()) for i in scelti)
    if parole_scelte < 0.8 * budget_parole:
        residuo = [int(i) for i in np.argsort(-theta.max(axis=1)) if int(i) not in gia]
        if residuo:
            scelti += su.seleziona_per_budget(frasi, residuo, budget_parole - parole_scelte)
    return scelti

# --- Fine della copia verbatim ------------------------------------------------


def configurazioni_lda():
    return ([(METODO_LDA, 'k_alfa', f'{k}_{a}') for k in N_TOPICS_GRIGLIA for a in ALFA_GRIGLIA]
            + [(METODO_LDA, 'seme', s) for s in SEMI])


def parametri_lda(parametro, valore):
    """(k, alpha, eta, seme) di una configurazione; None = default di scikit-learn (1/k)."""
    if parametro == 'seme':
        return N_TOPICS, None, None, valore
    k, a = valore.split('_')
    k, a = int(k), float(a)
    if k == N_TOPICS and a == 0.2:              # il default del notebook, priori impliciti
        return k, None, None, SEED
    return k, a, ETA_FISSO, SEED


def theta_lda(X, n_frasi, k, alfa, eta, seme):
    k = max(1, min(k, n_frasi))
    lda = LatentDirichletAllocation(n_components=k, doc_topic_prior=alfa,
                                    topic_word_prior=eta, random_state=seme)
    return lda.fit_transform(X)


def genera_lda(documento, budget, verifica_greedy=False):
    """{config: (riassunto, indici scelti, ridondanza interna)} per una riga.

    Replica `genera(documento, budget)` del notebook 17. Configurazioni con gli stessi
    (k, alpha, eta, seme) condividono un unico addestramento (default = seme 42). La
    ridondanza interna e' il coseno fra i vettori di conteggio delle frasi (spazio di LDA).
    """
    frasi = segmenta(documento)
    try:
        X = CountVectorizer(stop_words='english').fit_transform(frasi)
        sim = cosine_similarity(X)
    except ValueError:            # vocabolario CountVectorizer vuoto su testi minimi
        X, sim = None, None
    cache, uscita = {}, {}
    for config in configurazioni_lda():
        params = parametri_lda(config[1], config[2])
        if X is None:
            scelti = su.seleziona_per_budget(frasi, range(len(frasi)), budget)
        else:
            if params not in cache:
                cache[params] = seleziona_per_quote_parole(
                    frasi, theta_lda(X, len(frasi), *params), budget)
            scelti = cache[params]
        testo = ' '.join(frasi[i] for i in sorted(scelti))
        rid = ridondanza_interna(sim, scelti) if sim is not None else None
        uscita[config] = (testo, sorted(scelti), rid)
    return uscita


# Registro degli step: configurazioni, generatore per riga, riassunto di default da
# confrontare con il file pubblicato, descrizione per l'aggregato JSON.
STEP = {
    'centroid_mmr': {
        'configurazioni': configurazioni_centroid,
        'genera': genera_centroid,
        'default': {(METODO_TFIDF, 'lambda', LAMBDA): METODO_TFIDF,
                    (METODO_BERT, 'lambda', LAMBDA): METODO_BERT},
        'config': lambda metodo, parametro, valore: {
            'lambda_mmr': valore, 'k_ordine_budget': K_ORDINE_BUDGET,
            'segmentazione': 'psr.summarization',
            'vettorizzazione': (f'sentence-BERT ({ST_MODEL})' if metodo == METODO_BERT
                                else 'sklearn TfidfVectorizer (stop_words=english)'),
            'rilevanza': 'cosine vs centroide (MEAD)', 'selezione': 'MMR greedy'},
    },
    'lsa': {
        'configurazioni': configurazioni_lsa,
        'genera': genera_lsa,
        'default': {(METODO_LSA, 'k', K_LATENTE): METODO_LSA,
                    (METODO_STEIN, 'k', K_LATENTE): METODO_STEIN},
        'config': lambda metodo, parametro, valore: {
            'k_latente': valore, 'k_ordine_budget': K_ORDINE_BUDGET,
            'segmentazione': 'psr.summarization',
            'vettorizzazione': 'sklearn TfidfVectorizer (stop_words=english)',
            'svd': f'sklearn TruncatedSVD(random_state={SEED}), k cappato a min(X.shape)-1',
            'selezione': ('top-k per norma del vettore latente' if metodo == METODO_LSA
                          else 'greedy con deflazione (Steinberger)')},
    },
    'sbert': {
        # Il clustering e' instabile a perturbazioni di 1e-6 degli embedding (CPU contro la
        # GPU della corsa pubblicata): sulle prime 20 righe un rumore di 1e-6 cambia la
        # selezione in 9, quasi sempre scambiando frasi quasi duplicate fra articoli. Il testo
        # esatto non puo' coincidere; l'equivalenza si giudica sulle metriche medie
        # (verifica_metriche_default), non sui riassunti.
        'quota_diverse_max': 1.0,
        'configurazioni': configurazioni_sbert,
        'genera': genera_sbert,
        'default': {(METODO_KMEANS, 'n_cluster', N_CLUSTER): METODO_KMEANS,
                    (METODO_AGGLOM, 'n_cluster', N_CLUSTER): METODO_AGGLOM,
                    (METODO_AGGLOM, 'linkage', 'average'): METODO_AGGLOM},
        'config': lambda metodo, parametro, valore: {
            'n_cluster': valore if parametro == 'n_cluster' else N_CLUSTER,
            'segmentazione': 'psr.summarization', 'encoder': ST_MODEL,
            'clustering': ('KMeans(n_init=10, random_state=42)' if metodo == METODO_KMEANS
                           else 'AgglomerativeClustering(metric=cosine, linkage='
                                + (valore if parametro == 'linkage' else 'average') + ')'),
            'selezione': 'medoidi a giri, cluster piu grandi prima, tagliati al budget'},
    },
    'lda': {
        'configurazioni': configurazioni_lda,
        'genera': genera_lda,
        'default': {(METODO_LDA, 'k_alfa', f'{N_TOPICS}_0.2'): METODO_LDA,
                    (METODO_LDA, 'seme', SEED): METODO_LDA},
        'config': lambda metodo, parametro, valore: dict(zip(
            ['n_topics', 'doc_topic_prior', 'topic_word_prior', 'random_state'],
            [p if p is not None else 'None (1/k)' for p in parametri_lda(parametro, valore)]),
            segmentazione='psr.summarization',
            vettorizzazione='sklearn CountVectorizer (stop_words=english)',
            selezione='quote di parole proporzionali al peso dei topic'),
    },
}


# =============================================================================
# Driver
# =============================================================================

class ScrittoreSelezioni:
    """JSONL per configurazione: {row_id, scelti, ridondanza}, con ripresa come i TSV."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.completati = set()
        if self.path.exists():
            with open(self.path, encoding='utf-8') as f:
                self.completati = {json.loads(r)['row_id'] for r in f if r.strip()}
        self._f = open(self.path, 'a', encoding='utf-8')

    def scrivi(self, row_id, scelti, ridondanza):
        self._f.write(json.dumps({'row_id': row_id, 'scelti': scelti, 'ridondanza': ridondanza}) + '\n')
        self._f.flush()
        self.completati.add(row_id)

    def chiudi(self):
        self._f.close()


def genera(step, limit=None, verifica_greedy=False):
    configurazioni = STEP[step]['configurazioni']()
    scrittori = {c: su.ScrittoreRiassunti(RIASSUNTI_DIR / f'{slug_config(*c)}_{SCOPE}.tsv')
                 for c in configurazioni}
    selezioni = {c: ScrittoreSelezioni(SELEZIONI_DIR / f'{slug_config(*c)}_{SCOPE}.jsonl')
                 for c in configurazioni}
    t0, processati, saltati, errori = time.time(), 0, 0, []
    for es in su.itera_split(P['complete_tab'], 'test'):
        rid = es['row_id']
        if all(s.gia_fatto(rid) and rid in selezioni[c].completati for c, s in scrittori.items()):
            saltati += 1
            continue
        if limit is not None and processati >= limit:
            break
        try:
            uscita = STEP[step]['genera'](su.prepara_documento(es['document']),
                                          su.budget_riferimento(es), verifica_greedy)
            for c, (testo, scelti, rid_interna) in uscita.items():
                if not scrittori[c].gia_fatto(rid):
                    scrittori[c].scrivi(rid, testo)
                if rid not in selezioni[c].completati:
                    selezioni[c].scrivi(rid, scelti, rid_interna)
        except Exception as exc:  # difensivo, come su.ciclo_summarization
            errori.append((rid, repr(exc)))
            print(f'  ERRORE su row_id={rid}: {exc!r}', flush=True)
        processati += 1
        if processati % 100 == 0 or processati <= 3:
            media = (time.time() - t0) / processati
            print(f'[{processati}] {media:.2f} s/riga (saltate {saltati})', flush=True)
    for s in list(scrittori.values()) + list(selezioni.values()):
        s.chiudi()
    print(f'Generazione: {processati} nuove, {saltati} saltate, {len(errori)} errori, '
          f'{time.time() - t0:.0f} s')
    return errori


def verifica_default(step):
    """Il default rigenerato deve coincidere con i riassunti pubblicati in results/summaries/.

    Tolleranza: fino a QUOTA_DIVERSE_MAX delle righe comuni. Gli embedding MiniLM ricalcolati
    (qui su CPU) differiscono dagli originali nelle ultime cifre e, dove il greedy MMR e' a un
    quasi-pareggio, una frase puo' cambiare; il TF-IDF invece e' deterministico. Le righe
    diverse sono stampate e il loro numero entra nel riepilogo.
    """
    esito = True
    for c, pubblicato in STEP[step]['default'].items():
        nostri = su.carica_riassunti(RIASSUNTI_DIR / f'{slug_config(*c)}_{SCOPE}.tsv')
        loro = su.carica_riassunti(P['summaries_dir'] / f'{pubblicato}_{SCOPE}.tsv')
        comuni = set(nostri) & set(loro)
        # Il checkout git (autocrlf) porta a CRLF i newline dentro i riassunti pubblicati
        diversi = sorted(r for r in comuni if nostri[r] != loro[r].replace(chr(13), ''))
        solo_nostri = sorted(set(nostri) - set(loro))
        print(f'Equivalenza {slug_config(*c)} vs {pubblicato}_{SCOPE}.tsv: {len(comuni)} righe comuni, '
              f'{len(diversi)} diverse, {len(solo_nostri)} solo qui, {len(set(loro) - set(nostri))} solo li\'')
        if diversi:
            print(f'  prime diverse: {diversi[:10]}')
        if len(diversi) > STEP[step].get('quota_diverse_max', QUOTA_DIVERSE_MAX) * max(len(comuni), 1):
            esito = False
    return esito


QUOTA_DIVERSE_MAX = 0.01


def metriche(step, forza=False):
    riferimenti = list(su.itera_split(P['complete_tab'], 'test'))
    tetto = {es['row_id']: su.tetto_riferimento(es) for es in riferimenti}
    for c in STEP[step]['configurazioni']():
        slug = slug_config(*c)
        if not forza and (METRICHE_DIR / f'{slug}_{SCOPE}_aggregate.json').exists():
            print(f'  {slug}: metriche presenti, salto (--forza per ricalcolare)')
            continue
        t0 = time.time()
        riassunti = su.carica_riassunti(RIASSUNTI_DIR / f'{slug}_{SCOPE}.tsv')
        troncati = {rid: su.tronca_parole(t, tetto[rid]) for rid, t in riassunti.items()
                    if rid in tetto}
        n_tagliati = sum(len(riassunti[rid].split()) > tetto[rid] for rid in troncati)
        config = STEP[step]['config'](*c)
        config['ambito_budget'] = {
            'budget': 'lunghezza del riassunto di riferimento (oracle-length, issue #16)',
            'tetto': f'{su.FATTORE_TETTO_BUDGET} x budget, su.tronca_parole',
            'righe_troncate_dal_tetto': n_tagliati}
        _, agg = su.valuta_e_salva(riferimenti, troncati, slug, SCOPE, METRICHE_DIR, config)
        o = agg['overall']
        print(f'  {slug}: n={agg["n_esempi"]} R1 F1 {o["rouge1_f1"]:.4f} R2 F1 {o["rouge2_f1"]:.4f} '
              f'parole {o["parole_generate"]:.1f} ({time.time() - t0:.0f} s)', flush=True)


def verifica_metriche_default(step):
    """Le metriche del default devono coincidere con l'aggregato pubblicato post-tetto."""
    for c, pubblicato in STEP[step]['default'].items():
        nostro = json.loads((METRICHE_DIR / f'{slug_config(*c)}_{SCOPE}_aggregate.json')
                            .read_text(encoding='utf-8'))
        loro = json.loads((P['metrics_dir'] / f'{pubblicato}_{SCOPE}_aggregate.json')
                          .read_text(encoding='utf-8'))
        diff = max(abs(nostro['overall'][m] - loro['overall'][m])
                   for m in su.COLONNE_METRICHE if m != 'parole_generate')
        print(f'Metriche {slug_config(*c)} vs pubblicate: n {nostro["n_esempi"]} vs {loro["n_esempi"]}, '
              f'differenza massima fra le medie delle metriche {diff:.2e}')


def main():
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--metodo', required=True, choices=sorted(STEP))
    parser.add_argument('--limit', type=int, default=None, help='righe nuove da generare (smoke test)')
    parser.add_argument('--solo-metriche', action='store_true', help='salta la generazione')
    parser.add_argument('--forza', action='store_true',
                        help='ricalcola anche le metriche presenti')
    parser.add_argument('--verifica-greedy', action='store_true',
                        help='confronta ogni greedy vettorizzato con analizza_mmr (lento)')
    args = parser.parse_args()

    DEVICE = su.rileva_device()
    print(f'Step {args.metodo}, ambito {SCOPE}, device {DEVICE}', flush=True)
    if not args.solo_metriche:
        genera(args.metodo, args.limit, args.verifica_greedy)
    if not verifica_default(args.metodo):
        print('ATTENZIONE: il default rigenerato non coincide con i riassunti pubblicati.')
        return 1
    if args.limit is None:
        metriche(args.metodo, args.forza)
        verifica_metriche_default(args.metodo)
    return 0


DEVICE = 'cpu'

if __name__ == '__main__':
    sys.exit(main())
