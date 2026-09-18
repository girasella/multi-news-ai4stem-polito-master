#!/usr/bin/env python3
# coding=utf-8
"""Paired second-judge pilot: does the G-Eval ranking survive a judge from another lineage?

Background. The committed G-Eval (`test` scope) is judged by **gpt-5.4-mini**, an OpenAI
GPT-5-family model — the same family as one of the *judged* methods, `gpt5mini`. `CLAUDE.md`
only ever claimed *model-level* independence ("no self-judging"), not family-level. The
existing numbers are suggestive: among the four LLMs the G-Eval order is exactly inverted
w.r.t. BERTScore and ROUGE-1 F1 — G-Eval ranks gpt5mini > gemma > mistral > qwen while both
reference-anchored metrics rank mistral > qwen > gpt5mini > gemma, and `gpt5mini` is 3rd of
four on those yet 1st overall on G-Eval (4.89).

Two explanations fit that pattern equally well:
  (a) family bias — an OpenAI judge favours an OpenAI generator;
  (b) no bias — G-Eval scores coherence/fluency/relevance *to the source*, not similarity to
      the reference, and gpt5mini may genuinely write better prose while matching the
      reference less closely.
One judge cannot separate them. A judge of a different lineage can.

Design. Re-judge the SAME summaries (scope `test`, already generated and already judged by
gpt-5.4-mini) with DeepSeek, on a random sample of rows, then compare the two judges'
rankings method by method. It is a paired comparison on identical text, so any systematic
difference is attributable to the judge.

  - methods: the 4 LLMs (where bias would show) + 3 non-LLM controls, so a general
    "judge likes LLM prose" effect can be told apart from an OpenAI-specific one;
  - rows: random (seeded), so a partial run stays an unbiased sample — the same property the
    August run demonstrated when its budget cap stopped it at ~60%;
  - cost: ~EUR 0.0013/judgment, so 7 methods x 1,000 rows is about EUR 9.

Results go to a SEPARATE cache and JSON: nothing about the committed gpt-5.4-mini G-Eval is
touched, and the two judges stay independently inspectable.

Usage (needs AZURE_OPENAI_ENDPOINT/API_KEY of the account hosting the DeepSeek deployment):

    python scripts/pilota_giudice_deepseek.py --righe 1000
    python scripts/pilota_giudice_deepseek.py --righe 1000 --budget 12   # tetto di spesa
    python scripts/pilota_giudice_deepseek.py --solo-metriche            # riepilogo, zero chiamate
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / 'notebooks'
sys.path.insert(0, str(NOTEBOOKS_DIR))
import summ_utils as su  # noqa: E402

SCOPE = 'test'
SEED = 42
# 4 LLM + 3 controlli non-LLM presi a quote diverse della graduatoria gpt-5.4-mini
METODI_LLM = ['gpt5mini', 'gemma', 'qwen', 'mistral']
METODI_CONTROLLO = ['primera', 'firstk_psr', 'lexrank']
METODI = METODI_LLM + METODI_CONTROLLO

CACHE = REPO_ROOT / 'results' / 'metrics' / f'geval_cache_{SCOPE}_deepseek.jsonl'
OUT = REPO_ROOT / 'results' / 'metrics' / f'confronto_giudici_{SCOPE}.json'
# DeepSeek non ha prompt cache: input a tariffa piena (EUR/1M token, listino Azure EUR)
PREZZI = {'input': 0.50, 'cached': 0.50, 'output': 1.60}


def percorso_riassunti(metodo):
    """textrank/lexrank non hanno una corsa `_test.tsv`: i riassunti sono nel `_full.tsv`."""
    suffisso = 'full' if metodo in ('textrank', 'lexrank') else SCOPE
    return REPO_ROOT / 'results' / 'summaries' / f'{metodo}_{suffisso}.tsv'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--righe', type=int, default=1000, help='righe campionate (default 1000)')
    parser.add_argument('--budget', type=float, default=None,
                        help='tetto di spesa COMPLESSIVO in EUR (somma quanto e\' gia\' in cache)')
    parser.add_argument('--thread', type=int, default=2,
                        help='thread; il deployment e\' a 20 RPM, alzarlo non accelera (default 2)')
    parser.add_argument('--deployment', default=os.environ.get('AZURE_GEVAL_DEPLOYMENT',
                                                               'deepseek-giudice'))
    parser.add_argument('--solo-metriche', action='store_true',
                        help='riepiloga dalla cache senza fare chiamate')
    args = parser.parse_args()

    percorsi = su.percorsi_standard(REPO_ROOT)
    riassunti = {m: su.carica_riassunti(percorso_riassunti(m)) for m in METODI}
    mancanti = [m for m, r in riassunti.items() if not r]
    if mancanti:
        raise SystemExit(f'riassunti mancanti per: {mancanti}')

    # campione di righe riproducibile: le righe coperte da TUTTI i metodi del pilota
    comuni = sorted(set.intersection(*(set(r) for r in riassunti.values())))
    random.Random(SEED).shuffle(comuni)
    scelte = set(comuni[:args.righe])
    print(f'Metodi   : {len(METODI)} ({", ".join(METODI)})')
    print(f'Righe    : {len(scelte)} campionate su {len(comuni)} comuni (seed {SEED})')
    print(f'Giudizi  : {len(scelte) * len(METODI):,} previsti')

    if not args.solo_metriche:
        from openai import OpenAI
        client = OpenAI(base_url=os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/') + '/openai/v1/',
                        api_key=os.environ['AZURE_OPENAI_API_KEY'])
        formato = {'type': 'json_schema',
                   'json_schema': {'name': 'geval', 'strict': True, 'schema': su.SCHEMA_GEVAL}}

        def giudica_uno(sorgente, riassunto):
            """Una chiamata al giudice DeepSeek -> (punteggi, usage).

            Stessa forma della chiamata a gpt-5.4-mini (messaggi identici, `response_format`
            strict): verificato che DeepSeek la accetta e tollera `reasoning_effort`. Senza
            `response_format` il modello discorre invece di emettere JSON.
            """
            r = su.chiama_con_backoff(lambda: client.chat.completions.create(
                model=args.deployment,
                messages=su.costruisci_messaggi_geval(sorgente, riassunto),
                max_completion_tokens=1500, response_format=formato))
            return su.estrai_punteggi_geval(r.choices[0].message.content or ''), r.usage

        righe = []
        for es in su.itera_split(percorsi['complete_tab'], SCOPE):
            if es['row_id'] not in scelte:
                continue
            coppie = [(m, riassunti[m][es['row_id']]) for m in METODI
                      if es['row_id'] in riassunti[m]]
            righe.append((es['row_id'], su.prepara_documento(es['document']), coppie))
        righe.sort(key=lambda r: r[0])

        cache = su.CacheGiudizi(CACHE)
        speso_prima = sum(su.costo_da_token(cache.totali_token(), PREZZI).values())
        print(f'In cache : {len(cache):,} giudizi (EUR {speso_prima:.2f} gia\' spesi)\n')
        su.giudica_geval_concorrente(righe, giudica_uno, cache, n_thread=args.thread,
                                     etichetta='DeepSeek ', ogni=500, prezzi=PREZZI,
                                     budget_massimo=args.budget, speso_precedente=speso_prima)
        cache.chiudi()

    # ---- confronto fra i due giudici ----
    ds = su.CacheGiudizi(CACHE)
    gpt = su.CacheGiudizi(percorsi['metrics_dir'] / f'geval_cache_{SCOPE}.jsonl')
    confronto, righe_out = {}, []
    for m in METODI:
        a, b = ds.per_metodo(m), gpt.per_metodo(m)
        comuni_m = sorted(set(a) & set(b))          # solo righe giudicate da ENTRAMBI
        if not comuni_m:
            continue
        md = sum(a[r]['geval_media'] for r in comuni_m) / len(comuni_m)
        mg = sum(b[r]['geval_media'] for r in comuni_m) / len(comuni_m)
        confronto[m] = {'n': len(comuni_m), 'deepseek': round(md, 3),
                        'gpt_5_4_mini': round(mg, 3), 'delta': round(md - mg, 3),
                        'famiglia': 'LLM' if m in METODI_LLM else 'controllo'}
        righe_out.append((m, confronto[m]))
    ds.chiudi(); gpt.chiudi()

    if righe_out:
        print(f'\n{"metodo":<14}{"n":>6}{"DeepSeek":>10}{"gpt-5.4-mini":>14}{"delta":>8}   tipo')
        for m, d in sorted(righe_out, key=lambda x: -x[1]['deepseek']):
            print(f'{m:<14}{d["n"]:>6}{d["deepseek"]:>10.2f}{d["gpt_5_4_mini"]:>14.2f}'
                  f'{d["delta"]:>+8.2f}   {d["famiglia"]}')
        rank_ds = [m for m, _ in sorted(righe_out, key=lambda x: -x[1]['deepseek'])]
        rank_gpt = [m for m, _ in sorted(righe_out, key=lambda x: -x[1]['gpt_5_4_mini'])]
        print(f'\nordine DeepSeek     : {" > ".join(rank_ds)}')
        print(f'ordine gpt-5.4-mini : {" > ".join(rank_gpt)}')
        if 'gpt5mini' in confronto:
            d = confronto['gpt5mini']
            print(f'\ngpt5mini: DeepSeek {d["deepseek"]:.2f} vs gpt-5.4-mini {d["gpt_5_4_mini"]:.2f} '
                  f'({d["delta"]:+.2f})')
            print('delta negativo piu\' marcato che sugli altri LLM = indizio di family bias;')
            print('delta simile su tutti = la preferenza e\' per la prosa LLM, non per la famiglia.')
        OUT.write_text(json.dumps({'scope': SCOPE, 'giudice_b': args.deployment,
                                   'seed': SEED, 'righe_campionate': args.righe,
                                   'per_metodo': confronto}, ensure_ascii=False, indent=2),
                       encoding='utf-8')
        print(f'\nConfronto salvato: {OUT}')


if __name__ == '__main__':
    sys.exit(main())
