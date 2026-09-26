#!/usr/bin/env python3
# coding=utf-8
"""Extract the figures cited by ``docs/guida_studio.md`` into ``docs/resources/``.

The study guide embeds a selection of the charts the notebooks produce. They live in two
places: the "how it works" figures of notebooks 11 and 15-17 are already saved as PNGs under
``results/figures/`` (copied as-is), while the comparison charts of 05b/05d/18/19 exist only
as ``image/png`` outputs inside the committed ``.ipynb`` files (decoded from base64).

A notebook output is located by the markdown heading of the section that precedes its code
cell plus the image's ordinal inside that cell — not by raw cell index, so inserting a cell
elsewhere in the notebook does not silently pick the wrong chart. A missing heading or
ordinal is a hard error: re-run this script after re-executing a notebook, and fix the
mapping below if a section was renamed.

Usage::

    python scripts/estrai_figure_guida.py

The PDF of the guide is not built by this script and must be re-printed afterwards.
"""
import base64
import json
import shutil
import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parent.parent
DESTINAZIONE = RADICE / 'docs' / 'resources'

# nome file in docs/resources -> ('file', percorso) oppure ('notebook', nb, titolo, ordinale)
FIGURE = {
    'centroid_mmr_pca.png': ('file', 'results/figures/centroid_mmr/scatter_pca_centroid_mmr_row425.png'),
    'lsa_salienza.png': ('file', 'results/figures/lsa/salienza_row425.png'),
    'sbert_cluster_pca.png': ('file', 'results/figures/sbert_clustering/pca_cluster_row425.png'),
    'lda_peso_topic.png': ('file', 'results/figures/lda/peso_topic_row425.png'),
    'test_rouge.png': ('notebook', '05b_confronto_test.ipynb', '## Tabella e grafici', 1),
    'test_geval_dimensioni.png': ('notebook', '05b_confronto_test.ipynb', '## Tabella e grafici', 5),
    'lunghezze_dispersione_cluster.png': ('notebook', '18_analisi_lunghezze.ipynb', '## Vista 1', 1),
    'lunghezza_vs_recall.png': ('notebook', '05d_confronto_prima_dopo.ipynb', '## Grafici prima/dopo', 5),
    'lunghezza_riferimento_per_cluster.png': ('notebook', '18_analisi_lunghezze.ipynb', '## Vista 2', 1),
    'lunghezze_banda_per_metodo.png': ('notebook', '18_analisi_lunghezze.ipynb', '## Vista 3b', 1),
    'lunghezze_adattamento.png': ('notebook', '18_analisi_lunghezze.ipynb', '## Vista 4', 1),
    'prima_dopo_rouge1_f1.png': ('notebook', '05d_confronto_prima_dopo.ipynb', '## Grafici prima/dopo', 2),
    'prima_dopo_geval.png': ('notebook', '05d_confronto_prima_dopo.ipynb', '## Grafici prima/dopo', 3),
    'giudici_confronto.png': ('notebook', '19_confronto_giudici.ipynb', '## Vista 1', 1),
    'giudici_family_bias.png': ('notebook', '19_confronto_giudici.ipynb', '## Vista 2', 1),
}


def immagini_sezione(nb, titolo):
    """PNG (bytes) delle celle di codice fra l'intestazione `titolo` e la successiva."""
    celle = json.loads((RADICE / 'notebooks' / nb).read_text(encoding='utf-8'))['cells']
    dentro, trovata, png = False, False, []
    for c in celle:
        testo = ''.join(c['source'])
        if c['cell_type'] == 'markdown' and testo.lstrip().startswith('#'):
            dentro = testo.lstrip().startswith(titolo)
            trovata |= dentro
            continue
        if dentro and c['cell_type'] == 'code':
            for o in c.get('outputs', []):
                dati = o.get('data', {}).get('image/png')
                if dati:
                    png.append(base64.b64decode(''.join(dati) if isinstance(dati, list) else dati))
    if not trovata:
        sys.exit(f'{nb}: sezione "{titolo}" non trovata')
    return png


def main():
    DESTINAZIONE.mkdir(parents=True, exist_ok=True)
    for nome, sorgente in FIGURE.items():
        uscita = DESTINAZIONE / nome
        if sorgente[0] == 'file':
            shutil.copyfile(RADICE / sorgente[1], uscita)
            print(f'{nome:40s} <- {sorgente[1]}')
            continue
        _, nb, titolo, ordinale = sorgente
        png = immagini_sezione(nb, titolo)
        if len(png) < ordinale:
            sys.exit(f'{nb} "{titolo}": {len(png)} immagini, richiesta la n. {ordinale}')
        uscita.write_bytes(png[ordinale - 1])
        print(f'{nome:40s} <- {nb} "{titolo}" #{ordinale}')


if __name__ == '__main__':
    main()
