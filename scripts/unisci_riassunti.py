#!/usr/bin/env python3
# coding=utf-8
"""Unisce i TSV di riassunti prodotti da piu' macchine (corsa divisa con SUMM_PARTIZIONE).

    python scripts/unisci_riassunti.py DESTINAZIONE SORGENTE [SORGENTE ...]

- DESTINAZIONE e' il TSV "ufficiale" (es. results/summaries/qwen_fewshot_test_budgetref.tsv):
  se esiste, le sue righe vengono per prime e hanno la precedenza.
- Le SORGENTI (i TSV copiati dalle altre macchine) aggiungono solo i row_id mancanti.
- Un row_id presente in piu' file con testo diverso (es. una riga del pilota rigenerata
  anche dall'altra macchina: il campionamento non e' deterministico) tiene la PRIMA
  occorrenza e viene contato fra i duplicati; il testo scartato non entra nei risultati.

Scrive la destinazione con lo stesso formato di summ_utils.ScrittoreRiassunti (QUOTE_ALL),
passando da un file temporaneo: un'interruzione non lascia un TSV a meta'. Dopo l'unione,
un'esecuzione del notebook SENZA partizione riprende saltando le righe presenti, completa
eventuali buchi e calcola le metriche su tutte le righe.
"""
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'notebooks'))
import summ_utils as su  # noqa: E402


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    destinazione = Path(sys.argv[1])
    sorgenti = [Path(p) for p in sys.argv[2:]]

    unione, provenienza = {}, {}
    duplicati_uguali, duplicati_diversi = 0, 0
    for path in ([destinazione] if destinazione.exists() else []) + sorgenti:
        righe = su.carica_riassunti(path)
        nuove = 0
        for rid, testo in righe.items():
            if rid not in unione:
                unione[rid], provenienza[rid] = testo, path.name
                nuove += 1
            elif unione[rid] == testo:
                duplicati_uguali += 1
            else:
                duplicati_diversi += 1
        print(f'{path}: {len(righe)} righe, {nuove} nuove')

    tmp = destinazione.with_suffix('.tsv.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_ALL)
        w.writerow(['row_id', 'generated_summary'])
        for rid in sorted(unione):
            w.writerow([rid, unione[rid]])
    tmp.replace(destinazione)
    print(f'-> {destinazione}: {len(unione)} righe '
          f'(duplicati identici {duplicati_uguali}, duplicati diversi scartati {duplicati_diversi})')


if __name__ == '__main__':
    main()
