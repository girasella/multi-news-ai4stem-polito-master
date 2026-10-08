# Studio di sensibilità degli iperparametri — metodi estrattivi

Cartella autonoma con lo studio di sensibilità degli iperparametri dei metodi estrattivi
del benchmark. **Il documento da leggere è [`RIEPILOGO_sensibilita.md`](RIEPILOGO_sensibilita.md)**:
cosa si è fatto, perché, risultati e conclusioni.

Lo studio non tocca nulla di `results/`: i numeri pubblicati del benchmark restano quelli dei
notebook originali. Procede a step, un notebook alla volta.

| Step | Notebook | Slug | Iperparametro | Stato |
|---|---|---|---|---|
| 1 | 11 Centroid+MMR | `centroid_mmr`, `centroid_mmr_bert` | λ | fatto |
| 2 | 15 LSA | `lsa`, `lsa_steinberger` | `K_LATENTE` | fatto |
| 3 | 16 SBERT clustering | `sbert_kmeans`, `sbert_agglom` | `N_CLUSTER`, linkage | fatto |
| 4 | 17 LDA | `lda` | `N_TOPICS` × α, seme | da fare |

## Contenuto

```
RIEPILOGO_sensibilita.md    documento riassuntivo (metodo, risultati, conclusioni)
esegui_sensibilita.py       driver: rigenera i riassunti per la griglia e calcola le metriche
analisi_sensibilita.ipynb   tabelle, figure e riepilogo.json (legge solo risultati/, non genera)
risultati/
  metriche/                 {slug}__{parametro}-{valore}_test_budgetref_{per_example.csv,aggregate.json}
  selezioni/                indici delle frasi scelte e ridondanza interna, per riga (JSONL)
  figure/                   PNG usati dal riepilogo
  riepilogo.json            tutte le configurazioni: medie, differenze con IC, Jaccard, banda
riassunti/                  TSV generati (non versionati, rigenerabili)
```

## Come rieseguire

```bash
# smoke test su 20 righe (con verifica del greedy contro la copia verbatim del notebook)
python sensibilita_estrattivi/esegui_sensibilita.py --metodo centroid_mmr --limit 20 --verifica-greedy

# corsa completa sulla split test (riprende da dove si era fermata)
python sensibilita_estrattivi/esegui_sensibilita.py --metodo centroid_mmr

# solo metriche, dai TSV gia' generati
python sensibilita_estrattivi/esegui_sensibilita.py --metodo centroid_mmr --solo-metriche

# tabelle e figure
jupyter nbconvert --to notebook --execute --inplace sensibilita_estrattivi/analisi_sensibilita.ipynb
```

Il driver contiene una copia verbatim delle funzioni di generazione dei notebook (stesso
schema dei notebook didattici) e controlla che il valore di default riproduca i riassunti
pubblicati in `results/summaries/{slug}_test_budgetref.tsv`.
