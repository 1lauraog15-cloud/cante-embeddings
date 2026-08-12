# Cante Style Embeddings - Cantaor Identification

A computational musicology project learning an embedding space that captures a flamenco cantaor's **interpretive style** — not *which* palo they sing, but *how* they sing it — using the same logic as speaker embeddings in speech recognition.

This project demonstrates transfer learning and metric learning (triplet loss) applied to a low-resource audio task, complementing a companion project (flamenco palos classifier, classical ML) as part of a computational musicology portfolio.

## Approach

Given a small combined dataset (109 recordings, 79 unique cantaores, 31 with 2+ recordings), training a deep audio encoder from scratch is impractical. Instead:

1. Pretrained **OpenL3** embeddings (via `torchopenl3`) provide a general audio representation.
2. A small projection head (512→256→128) is trained on top with **triplet loss**, using window-level embeddings as training examples to multiply the effective number of same-cantaor training pairs.
3. Evaluation follows a **closed-set pairwise verification** protocol (EER, ROC/AUC) — the primary methodological decision, made explicit and justified given the dataset's sample size (see `01_conteo_cantaores.ipynb`).
4. A small exploratory open-set appendix (`04_train_encoder.ipynb`) tests generalization to cantaores never seen in training, reported with appropriate caution given the tiny sample involved.

**Result**: AUC 0.80, EER ~30% on held-out closed-set pairs — a modest but genuine, reproducible improvement over the pretrained baseline, honestly contextualized against the dataset's limitations throughout.

## Repository structure

## Reproducing this project

Audio files are **not included** in this repository (see Data & licensing below). To reproduce from scratch:

1. Obtain TONAS and cante100 audio directly from MTG-UPF (see links below).
2. Place them at `data/raw/TONAS/` and `data/raw/Cante100audio/` respectively.
3. Install dependencies: `pip install -r requirements.txt`
4. Run notebooks in order (`01` → `04`). Note: embedding extraction in `03` is compute-heavy — see the notebook's environment note regarding GPU/Colab.

Alternatively, `data/processed/recording_embeddings.npz` and `projection_head_final.pt` (already included) allow reproducing evaluation and analysis (`04`) without re-running extraction.

## Data & licensing

This project uses two datasets from the **COFLA (COmputational analysis of FLAmenco music)** research project, Universidad de Sevilla / Music Technology Group, Universitat Pompeu Fabra:

- **TONAS**: 72 a cappella flamenco recordings (Deblas, Martinetes).
  Mora, J., Gómez, F., Gómez, E., Escobar-Borrego, F.J., Díaz-Báñez, J.M. (2010). *Melodic Characterization and Similarity in A Cappella Flamenco Cantes.* 11th ISMIR. © 2012 COFLA project. [mtg.upf.edu/download/datasets/tonas](https://www.upf.edu/web/mtg/tonas)

- **cante100**: 100 flamenco recordings across 10 style families.
  Kroher, N., Díaz-Báñez, J.M., Mora, J., Gómez, E. (2015). *Corpus COFLA: A research corpus for the Computational study of Flamenco Music.* arXiv:1510.04029. © COFLA 2015. [zenodo.org/records/1322542](https://zenodo.org/records/1322542)

**Audio files are not redistributed in this repository**, per the datasets' own terms of use ("no rights for redistribution or modification" — COFLA/MTG-UPF). Only derived, non-reversible artifacts (catalog metadata, learned embeddings, trained model weights) are included. Anyone wishing to reproduce this work should obtain the audio directly from the sources above.

## Portfolio context

This project is part of a broader computational musicology portfolio built for a PhD candidacy application. A companion project — a flamenco palos classifier using classical ML and clustering — addresses a distinct MIR task (genre/style classification) and lives in a separate repository, deliberately keeping the two pieces differentiated to demonstrate breadth across ML paradigms.