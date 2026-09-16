<div align="center">

# Likelihood-Coupled Graph-Prototype Refinement  
## for Passenger-Requirement Discovery

**Research code and reproducibility resources for discovering passenger requirements from transit-related social media**

**Fatima Ashraf · Muhammad Ayub Sabir · Jiaxin Deng · Junbiao Pang · Haitao Yu**

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-194%20passed-success)
![Research Code](https://img.shields.io/badge/status-research%20code-6f42c1)

</div>

---

## Overview

This repository contains the implementation and reproducibility resources for the manuscript:

> **Likelihood-Coupled Graph-Prototype Refinement for Discovering Passenger Requirements from Social Media**

The framework is designed for **unsupervised passenger-requirement discovery from short, sparse transit-related social-media text**. It combines a frozen neural topic-model parent with post-fit graph-prototype refinement while preserving a strict fit/evaluation separation.

The released workflow includes:

- lexical graph construction from fitting-corpus statistics;
- disjoint prototype-core extraction and topic alignment;
- minimum-distortion affinity transport;
- fitting-corpus moment calibration;
- pooled lexical backoff for sparse posts;
- frozen internal evaluation and ablation analysis;
- blinded post-fit human evaluation;
- independent English architecture replication; and
- frozen parameter-sensitivity reporting.

No labels, human judgments, or city identifiers are used for model fitting or document inference. City information is used only for split auditing, stratified evaluation, and descriptive analysis.

---

## Quick Navigation

| Resource | Purpose |
|---|---|
| [`docs/DATASETS.md`](docs/DATASETS.md) | Dataset provenance, corpus statistics, local placement, privacy, and access notes |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Complete command-line reproduction guide from Step 01 through Step 14 |
| [`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md) | What is included and intentionally excluded from this code release |
| [`inputs/README.md`](inputs/README.md) | Rules for local restricted-data placement |
| [`artifacts/README.md`](artifacts/README.md) | Guidance for publication-approved aggregate artifacts |
| [`scripts/`](scripts/) | Canonical PowerShell stage launchers |
| [`configs/v6/`](configs/v6/) | Frozen stage configurations |
| [`tests/`](tests/) | Mathematical, contract, and integration tests |

---

## Method at a Glance

The framework follows a **fit-and-freeze** design. Corpus-dependent quantities are estimated only from the applicable fitting partition and are then frozen before held-out evaluation.

### Core components

1. **Frozen neural parent**  
   The parent topic model provides the initial latent representation and topic-word affinity structure.

2. **Lexical graph construction**  
   A positive-NPMI lexical graph is constructed from the fitting corpus.

3. **Prototype extraction and alignment**  
   Disjoint prototype cores are extracted and aligned one-to-one with the native parent topics.

4. **Affinity transport**  
   The native topic affinity is projected toward the graph-derived prototype structure using a minimum-distortion transport step.

5. **Moment calibration**  
   Fitting-corpus moments are used to correct decoder shift introduced by the transported affinity.

6. **Pooled lexical backoff**  
   Sparse-document predictions are mixed with a corpus-level lexical distribution using the frozen backoff policy.

7. **Frozen evaluation**  
   Internal confirmation, ablations, human assessment, external replication, and sensitivity reporting are performed without reopening the fitting boundary.

---

## Repository Structure

```text
passenger-graph-refinement/
│
├── README.md
├── main_v6.py
├── pyproject.toml
├── requirements-step*.txt
│
├── src/
│   └── v6/
│       ├── core.py
│       ├── graph_prototype_adapter.py
│       ├── transport_calibration.py
│       ├── city_backoff.py
│       ├── evaluation.py
│       ├── predictive_evaluation.py
│       ├── comparators/
│       └── step*.py
│
├── configs/
│   └── v6/
│       └── step*.yaml
│
├── scripts/
│   └── run_step*.ps1
│
├── tests/
│   └── test_*.py
│
├── docs/
│   ├── DATASETS.md
│   ├── REPRODUCIBILITY.md
│   ├── RELEASE_SCOPE.md
│   └── frozen preregistration / technical records
│
├── inputs/
│   └── README.md
│
└── artifacts/
    └── README.md
```

Some implementation paths retain the historical `v6` identifier because renaming frozen interfaces, configuration paths, hashes, or contract-bound resources could compromise reproducibility. The public repository name therefore uses the method-oriented title while the internal release structure remains stable.

---

## Requirements

The released code requires:

- **Python 3.10 or later**;
- stage-specific dependencies from the corresponding `requirements-step*.txt` files;
- authorized local copies of required datasets;
- prerequisite outputs produced by earlier stages; and
- the locked parent implementation where required.

The clean release was validated under **Python 3.11.7**.

Because some stages use different scientific and deep-learning dependencies, the repository keeps **stage-specific requirement files** rather than forcing every optional dependency into one monolithic environment.

For complete environment and stage instructions, see:

**[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)**

---

## Quick Verification

From the repository root:

```powershell
python --version
python .\main_v6.py --help
python -m pytest -q tests
```

Release validation:

```text
194 passed
```

A passing unit-test suite verifies the released code contracts; full experimental reproduction additionally requires the authorized datasets and stage prerequisites documented in the reproducibility guide.

---

## Canonical Experimental Pipeline

The authoritative pathway uses **Step 04R2, Step 05R2, and Step 06R2**. Historical Step 04/06 compatibility routes are not part of the final scientific pathway, and there is no canonical Step 05 route.

```mermaid
flowchart LR
    S01["Step 01"] --> S02["Step 02"]
    S02 --> S03["Step 03"]
    S03 --> S04["Step 04R2"]
    S04 --> S05["Step 05R2"]
    S05 --> S06["Step 06R2"]
    S06 --> S07["Step 07"]
    S07 --> S08["Step 08"]
    S08 --> S09["Step 09"]
    S09 --> S10A["Step 10<br/>preparation gate"]
    S10A --> S11["Step 11"]
    S11 --> S11B["Step 11B"]
    S11B --> S13["Step 13<br/>external replication"]
    S13 --> XFER["Transfer external evidence"]
    XFER --> S10B["Step 10<br/>final pass"]
    S10B --> S12["Step 12"]
    S12 --> S14["Step 14"]
```

The final dependency-aware execution order is therefore:

```text
01 → 02 → 03 → 04R2 → 05R2 → 06R2 → 07 → 08 → 09
→ 10 → 11 → 11B → 13 → external-evidence transfer → 10 → 12 → 14
```

The ordering near Steps 10–14 is intentionally dependency-aware rather than strictly numerical.

---

## Running the Pipeline

The recommended public entry points are the PowerShell runners in [`scripts/`](scripts/).

Examples:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_step01.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_step04r2.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_step07.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_step13.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_step14.ps1
```

Do **not** run the stages as an undifferentiated batch on a fresh system. Several stages require outputs or restricted inputs produced/provided earlier in the pipeline.

The complete stage-by-stage commands, prerequisites, Step-13 → Step-10 evidence transfer, direct Python CLI equivalents, and final verification procedure are documented in:

**[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)**

---

## Datasets

Two independent transit social-media corpora are used.

### Internal Chinese corpus

The primary experiment uses Chinese transit-related social-media posts from:

- **Beijing**
- **Shanghai**
- **Xiamen**

After the frozen study-specific screening and preprocessing pipeline, the modeled corpus contains **772 documents**:

| City | Modeled documents |
|---|---:|
| Beijing | 432 |
| Shanghai | 181 |
| Xiamen | 159 |
| **Total** | **772** |

The frozen experimental allocation contains:

- 620 original-training documents;
- 75 spent-development documents;
- 77 held-out modeled test documents;
- 695 documents in the final fitting set; and
- a fixed vocabulary of 1,148 terms.

### External English corpus

Independent replication uses **Transit Tweet Data**, associated with:

> *Learning About the Transit Passenger Experience from Microblogging Posts* (2025)

The audited source contains **559 valid, distinct posts** after excluding two blank trailing CSV rows. The frozen chronological protocol uses:

- 391 original-training posts;
- 84 spent-development posts;
- 475 posts in the final fitting corpus; and
- 84 temporally held-out test posts.

The English experiment is an **architecture-level replication with language-compatible refitting**, not zero-shot transfer of the fitted Chinese model state.

### Where are the datasets?

Raw research datasets are **not stored in this GitHub repository**.

Authorized users should place required local data under the ignored `inputs/` structure according to the detailed instructions in:

**[`docs/DATASETS.md`](docs/DATASETS.md)**

For the external Step-13 experiment, the audited archive is expected locally at:

```text
inputs/v6/step13/Transit-Tweet-Data-main.zip
```

Dataset hashes, source-corpus statistics, split definitions, privacy restrictions, and local-placement instructions are all documented in `docs/DATASETS.md`.

---

## Reproducibility

Reproducibility is organized around frozen stage configurations, explicit runners, automated tests, and fail-closed evaluation contracts.

The repository provides:

- frozen YAML configurations;
- canonical PowerShell stage launchers;
- mathematical and integration tests;
- stage-specific dependency files;
- source/data contract checks;
- frozen preregistration and technical records; and
- detailed command-line instructions.

Full reproduction requires authorized access to the underlying datasets and restricted post-fit human-evaluation inputs where applicable.

Start here:

**[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)**

---

## Parent Topic Model

The framework uses **EnCOT** as the neural parent implementation.

Official repository:

```text
https://github.com/manhdo249/EnCOT
```

Locked revision used by the released workflow:

```text
8ac3592165cc7851676be2776b314eb6e44e9388
```

The parent is frozen before the graph-prototype refinement operations used by this framework.

---

## Evaluation Design

The release separates semantic quality, topic separation, and predictive reliability rather than collapsing them into one aggregate score.

The evaluation framework includes:

- NPMI-based topic coherence;
- \(C_v\) coherence;
- topic diversity;
- pairwise descriptor redundancy;
- document-completion negative log-likelihood;
- city-stratified internal evaluation;
- paired multi-seed comparison;
- component ablation;
- blinded human assessment;
- external temporal replication; and
- pre-test parameter-sensitivity reporting.

Human judgments are used only as **post-fit evaluation evidence**. They are never used to fit the model, tune document inference, or construct the held-out predictions.

---

## Data and Privacy

This repository intentionally excludes:

- raw social-media posts;
- user identifiers and profile information;
- geographic coordinates;
- serialized comments and other account-level metadata;
- individual human-annotation response files;
- model checkpoints;
- generated experiment-output directories;
- result archives and local caches; and
- local third-party source checkouts.

Only code, configurations, tests, aggregate/non-identifying documentation, and approved reproducibility resources are intended for release.

See [`docs/DATASETS.md`](docs/DATASETS.md) and [`docs/RELEASE_SCOPE.md`](docs/RELEASE_SCOPE.md) for details.

---

## Outputs and Artifacts

Generated experiment outputs are intentionally not versioned as ordinary source files.

Publication-approved aggregate artifacts may be placed under:

```text
artifacts/
```

See [`artifacts/README.md`](artifacts/README.md) for the release policy.

Local experimental outputs, caches, checkpoints, raw result archives, and restricted evidence should remain outside Git tracking.

---

## Authors

**Fatima Ashraf**  
**Muhammad Ayub Sabir**  
**Jiaxin Deng**  
**Junbiao Pang**  
**Haitao Yu**

---
