# Likelihood-Coupled Graph-Prototype Refinement for Passenger-Requirement Discovery

Research code and reproducibility resources for the manuscript:

**Likelihood-Coupled Graph-Prototype Refinement for Discovering Passenger Requirements from Social Media**

**Authors:** Fatima Ashraf, Muhammad Ayub Sabir, Jiaxin Deng, Junbiao Pang, and Haitao Yu.

## Overview

This repository implements an unsupervised framework for discovering coherent passenger requirements from transit-related social-media text.

The framework combines a frozen neural topic-model parent with likelihood-coupled graph-prototype refinement, lexical graph construction, prototype alignment, affinity transport, moment calibration, and pooled lexical backoff for sparse documents.

No labels, human judgments, or city identifiers are used for model fitting or document inference. City information is used only for stratified evaluation and descriptive analysis.

## Verification

Validated under Python 3.11.7.

**Test status: 194 passed.**

Run:

    python -m pytest -q tests

## Repository structure

- src/v6/ — model, refinement, evaluation, and comparator implementation
- configs/v6/ — frozen experiment configurations
- scripts/ — stage runners
- tests/ — mathematical, contract, and integration tests
- docs/ — preregistration and reproducibility documentation
- inputs/ — local-data placeholder; raw datasets are not distributed
- artifacts/ — publication-approved aggregate artifacts

## Primary experimental pipeline

Step 01 → Step 02 → Step 03 → Step 04R2 → Step 05R2 → Step 06R2 → Step 07 → Step 08 → Step 09 → Step 10 → Step 11/11B → Step 13 → Step 12 → Step 14

## Parent model

The framework uses EnCOT as the parent topic model.

Locked EnCOT commit:

8ac3592165cc7851676be2776b314eb6e44e9388

## Data availability

Raw social-media data, user identifiers, coordinates, individual human-annotation responses, checkpoints, and private experiment outputs are not distributed.

See docs/DATASETS.md and docs/REPRODUCIBILITY.md.

