# Cante Style Embeddings — Cantaor Identification

**Can a metric-learned embedding space capture a flamenco singer's interpretive style
independently of which style family (*palo*) they perform?** This project investigates that
question by applying transfer learning and triplet-loss metric learning to a small corpus of
historical recordings — using the same conceptual framework as speaker verification in speech,
but applied to the expressive identity of flamenco cantaores.

**Result**: AUC 0.80, EER ~30% on held-out closed-set pairwise verification — a modest but
genuine, reproducible improvement over the frozen OpenL3 pretrained baseline. The improvement
is real; its practical significance at this dataset scale is limited and is reported as such.

**This project interrogates its own results.** `05_confound_probe.ipynb` is a post-hoc
diagnostic probe, included as a core deliverable, that tests whether the embedding space is
capturing cantaor-level style or partially collapsing onto domain and dataset artefacts.
Findings are documented there and summarised under [Limitations](#limitations) below.

---

## Method

Given a combined dataset of 109 recordings across 79 unique cantaores — of whom only 31 have
two or more recordings — training a deep audio encoder from scratch is impractical. Instead:

1. **Pretrained OpenL3** embeddings (`torchopenl3`, 512-d) provide a general-purpose audio
   representation without exposure to this corpus.
2. A **projection head** (512 → 256 → 128, L2-normalized) is fine-tuned with **triplet
   margin loss**. Window-level embeddings serve as training examples, multiplying the
   effective number of same-cantaor pairs from the sparse recording count.
3. Triplets are sampled by first choosing cantaores **uniformly at random** (to prevent
   high-recording-count cantaores from dominating by volume), then drawing windows randomly
   within each cantaor. No distance-based mining is used.
4. Evaluation uses a **closed-set pairwise verification** protocol (ROC/AUC, EER): 31 test
   recordings are compared against 78 training recordings (2,418 pairs). The protocol choice
   is made explicit and justified in `01_conteo_cantaores.ipynb` given the dataset size.
5. A small open-set appendix in `04_train_encoder.ipynb` tests generalization to cantaores
   unseen during training, reported with appropriate caution.

## Limitations

- **Domain confound — real and strong at the representation level.** A logistic regression
  probe trained on the final 128-d embeddings separates TONAS from cante100 recordings with
  balanced accuracy 0.979 (permutation test, p < 0.002). The embedding space encodes
  dataset origin as a strong signal, alongside — or entangled with — cantaor style.

- **Effect on the main evaluation — not statistically detectable at this scale.** Restricting
  the pairwise evaluation to cante100-only pairs produces AUC 0.813 vs. 0.801 full-set and
  EER 25.4% vs. 32.1%, but bootstrap confidence intervals overlap substantially. This is not
  evidence of no effect; it reflects the available statistical power at this sample size. The
  limitation is one of interpretive validity, particularly for the 10 cantaores present in
  both datasets.

- **Palo confound — real but modest.** A probe predicting palo (10 classes, cante100 only)
  achieves balanced accuracy 0.243 — 2.4× chance level, permutation test p < 0.002. The
  embedding space carries some style-family information, though far less than dataset-domain
  information.

- **Convergent evidence from confusion geometry.** The closest cross-domain pair in embedding
  space sits at cosine distance 0.1185; the closest within-TONAS pair sits at 0.0455. The
  acoustic homogeneity of TONAS recordings (a cappella, similar era) compresses that
  neighbourhood of the space in ways that interact with cantaor proximity estimates.

- **Palo imbalance in the split**: the closed-set protocol does not guarantee palo balance
  per cantaor; a cantaor represented only by *Martinetes* in test and only by *Debla* in
  train faces a harder verification pair for reasons unrelated to identity.

Full probe methodology and numbers: `05_confound_probe.ipynb`.

## Future work

- **Distance-based negative mining**: the current training uses purely random negative
  selection (cantaores drawn uniformly, windows drawn at random within each). Hard-negative
  mining — selecting negatives already close in embedding space — was not explored and would
  likely strengthen discriminative fine-tuning at this scale.
- **Domain-specific baselines**: comparison against COFLA-literature descriptors (fundamental
  frequency contours, melisma rate, ornamentation density) as musicologically grounded
  baselines, rather than relying solely on general-purpose audio embeddings.
- **Musicological annotation of confused pairs**: `06_confusion_pairs_for_annotation.ipynb`
  exports the most confusable cantaor pairs for expert annotation — ongoing work to understand
  whether proximity in embedding space reflects genuine stylistic affinity or artefact.

## Repository structure

```
01_conteo_cantaores.ipynb               — dataset audit, protocol justification
02_train_test_split.ipynb               — stratified split design
03_audio_preprocessing.ipynb            — OpenL3 window-level extraction
04_train_encoder.ipynb                  — projection head training, evaluation, open-set appendix
05_confound_probe.ipynb                 — post-hoc domain-confound diagnostic
06_confusion_pairs_for_annotation.ipynb — pairwise confusion analysis for expert annotation
data/
  raw/                                  — audio (not included; see Data & licensing)
  processed/
    confusion_pairs_for_annotation.csv  — 75 most confusable pairs (14 columns)
    [embeddings, model weights]         — not included; regenerable from notebooks 03–04
```

## Portfolio context

This project is one half of a computational musicology portfolio built for a PhD candidacy
application. The companion project — **[Flamenco Palos Classifier](<COMPANION_REPO_URL>)** —
addresses the orthogonal task of style-family classification using classical ML and
clustering, without deep learning. Together they demonstrate breadth across paradigms: one
classical, one deep metric learning; one closed-vocabulary classification, one open-ended
similarity learning.

## Reproducing this project

Audio files and derived artifacts (embeddings, model weights) are **not included** —
see *Data & licensing* for the reason. Full reproduction requires the original audio:

1. Obtain TONAS and cante100 audio directly from MTG-UPF (links below).
2. Place audio at `data/raw/TONAS/` and `data/raw/Cante100audio/` respectively.
3. `pip install -r requirements.txt`
4. Run notebooks in order (`01` → `04`). Embedding extraction in `03` is compute-heavy —
   see the notebook's environment note regarding GPU/Colab.

## Data & licensing

This project uses two datasets from the **COFLA (COmputational analysis of FLAmenco music)**
research project, Universidad de Sevilla / Music Technology Group, Universitat Pompeu Fabra:

- **TONAS**: 72 a cappella flamenco recordings (Deblas, Martinetes).  
  Mora, J., Gómez, F., Gómez, E., Escobar-Borrego, F.J., Díaz-Báñez, J.M. (2010).
  *Melodic Characterization and Similarity in A Cappella Flamenco Cantes.* 11th ISMIR.
  © 2012 COFLA project, Universidad de Sevilla. Distribution rights granted to MTG-UPF.  
  Source: [mtg.upf.edu/download/datasets/tonas](https://www.upf.edu/web/mtg/tonas)

- **cante100**: 100 flamenco recordings across 10 style families. Audio sourced from
  commercial flamenco anthology CDs; third-party rights apply.  
  Kroher, N., Díaz-Báñez, J.M., Mora, J., Gómez, E. (2015). *Corpus COFLA: A research
  corpus for the Computational study of Flamenco Music.* arXiv:1510.04029. © COFLA 2015.  
  Source: [zenodo.org/records/1322542](https://zenodo.org/records/1322542)

**Terms of use (identical for both datasets — COFLA/MTG-UPF):**
> "The provided datasets are offered free of charge for internal non-commercial use.
> We do not grant any rights for redistribution or modification. All data collections
> were gathered by the COFLA team. © COFLA. All rights reserved."

**What is and is not included in this repository:**

- Audio files: **not included** (excluded via `.gitignore`).
- Derived artifacts — embeddings (`recording_embeddings.npz`, `final_results.npz`) and
  model weights (`projection_head_final.pt`): **not included**. The terms prohibit
  redistribution and modification; the status of non-reversible derived features is not
  explicitly addressed, and this project takes the conservative position of not distributing
  them. Files are on the author's local machine and regenerable from notebooks `03` and `04`.
- Catalog metadata CSVs (`cante100_catalogo.csv`, `tonas_catalogo.csv`): **included** —
  bibliographic metadata only (name, palo, filename, duration, sample rate); no audio
  features or musically derived content.
- Processed split and count tables (`train_test_split.csv`,
  `conteo_cantaores_combinado.csv`): **included** — derived entirely from catalog metadata,
  no audio features.
