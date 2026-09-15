# Dataset Documentation

This repository uses two independent transit social-media datasets:

1. an internal Chinese Weibo corpus from Beijing, Shanghai, and Xiamen; and
2. an external English **Transit Tweet Data** corpus used for independent architecture replication.

The datasets are **not merged**. The Chinese fitted model state is not transferred directly to the English vocabulary. City information is used only for leakage-safe partition auditing, stratified evaluation, and descriptive analysis; it is not supplied to model fitting or document inference.

## 1. Dataset overview

| Dataset | Language | Source-level records | Experimental corpus | Role |
|---|---|---:|---:|---|
| Three-city Weibo corpus | Chinese | 2,073 raw source records; 2,072 nonempty; 2,054 distinct nonempty posts | 772 modeled posts | Internal development, final fitting, held-out evaluation, human assessment, and city-stratified analysis |
| Transit Tweet Data | English | 561 physical CSV rows; 559 valid and distinct posts | 559 retained posts | Independent external architecture replication under a frozen chronological protocol |

These counts refer to different stages of the data pipeline and must not be reported interchangeably.

---

## 2. Internal Chinese Weibo corpus

### 2.1 Source-corpus characteristics reported in the manuscript

Statistics were recomputed from the three supplied city files before the modeling pipeline.

| City | Raw records | Nonempty post bodies | Distinct nonempty post bodies | Recorded date coverage | Post length, median [IQR] characters |
|---|---:|---:|---:|---|---:|
| Beijing | 812 | 811 | 804 | 2019-09-07 to 2019-11-19 | 77 [46, 126] |
| Shanghai | 588 | 588 | 580 | 2017-01-16 to 2019-11-16 | 73 [40, 130] |
| Xiamen | 673 | 673 | 671 | 2016-08-13 to 2019-10-31* | 91 [54, 136] |
| **All cities** | **2,073** | **2,072** | **2,054** | **2016-08-13 to 2019-11-19** | **81 [46, 132]** |

\* For 373 Xiamen records, the publication-date field omits the year. Their 2019 placement follows the preserved reverse-chronological ordering and should therefore be treated as inferred rather than directly observed.

Character length excludes only exported outer brackets and surrounding whitespace; it is not the same as model-token length. The combined distinct count is computed across all cities.

### 2.2 Processed repository snapshot

The processed files available to the experimental pipeline have a common 17-column schema.

| City | Processed file | Rows | Unique users | Parseable timestamp coverage |
|---|---|---:|---:|---|
| Beijing | `beijing_processed.csv` | 805 | 664 | 2019-09-07 to 2019-11-19 |
| Shanghai | `shanghai_processed.csv` | 588 | 518 | 2017-01-16 to 2019-11-16 |
| Xiamen | `xiamen_processed.csv` | 671 | 444 | 2016-08-13 to 2019-11-02 |
| **Total** | — | **2,064** | — | 2016-08-13 to 2019-11-19 |

Unique-user counts are calculated within each file from non-null `user_id` values and are not summed across cities because the same account could occur in more than one city file.

**Provenance note.** The 2,073-row source-corpus table and the 2,064-row processed-file snapshot describe different upstream stages/snapshots. The public repository should not imply that they are the same quantity. The exact upstream removal or transformation accounting for the nine-row difference should be documented from the original preprocessing provenance if it is needed for the final archival release.

### 2.3 Processed source schema

| Column | Type | Experimental use |
|---|---|---|
| `seq_id` | integer/string | Internal row identifier; not a lexical model feature |
| `screen_name` | string | Identifying metadata; excluded from model input |
| `user_id` | string/integer | Restricted identifier; excluded from lexical model input |
| `user_bio` | string | Profile metadata; excluded |
| `verification` | integer/Boolean | Account metadata; excluded |
| `timestamp` | string/datetime | Used for provenance/auditing where applicable |
| `influence` | float | Legacy field; not part of the released method |
| `followers` | integer | Engagement/account metadata; excluded |
| `following` | integer | Engagement/account metadata; excluded |
| `likes` | integer | Engagement metadata; excluded |
| `comments_count` | integer | Engagement metadata; excluded |
| `text` | string | Source for the lexical representation |
| `comments_list` | string/JSON | Excluded |
| `comments_24h` | integer | Legacy metadata; excluded |
| `comment_times` | string/JSON | Excluded |
| `timestamp_raw` | string | Original timestamp for provenance |
| `city` | string | Split auditing, stratified evaluation, and descriptive analysis only |

The released framework is a lexical topic-modeling pipeline. Legacy influence scores, follower counts, biographies, verification status, engagement statistics, city labels, and human judgments do not enter model fitting or document inference.

### 2.4 Frozen modeled corpus

Following the frozen source-screening procedure, 778 internal records received split assignments. Six records were subsequently excluded by the predefined model-inclusion rules, leaving **772 modeled documents**.

| City | Modeled posts |
|---|---:|
| Beijing | 432 |
| Shanghai | 181 |
| Xiamen | 159 |
| **Total** | **772** |

The experimental allocation is:

| Partition / quantity | Documents |
|---|---:|
| Original training partition | 620 |
| Spent-development partition | 75 |
| Held-out modeled test partition | 77 |
| **Total modeled corpus** | **772** |
| Final fitting set: training + spent development | **695** |
| Test documents eligible for document-completion scoring | 75 |
| Fixed fitting vocabulary | 1,148 terms |

Duplicate groups are kept within a single partition. After development decisions were completed, the 75-document development partition was classified as spent and combined with the 620 training documents for the final 695-document fit.

The final-fit matrix contains 6,944 retained tokens, or 9.99 retained tokens per document on average. The fixed 1,148-term vocabulary was constructed without using held-out test terms.

The 77-document test partition remains outside model fitting, graph construction, moment calibration, pooled-backoff estimation, and hyperparameter selection. Two test documents do not satisfy the frozen document-completion eligibility rule, so predictive NLL is calculated on 75 eligible test documents.

### 2.5 Privacy and availability

The raw or processed Chinese social-media files are **not distributed** in this repository.

The public release must not contain raw post text, `screen_name`, `user_id`, biographies, account or post identifiers, serialized comments, URLs, or other account-level metadata unless the authors have explicit redistribution permission and an appropriate ethics/privacy basis.

The repository may distribute non-identifying aggregate counts, schemas, preprocessing code, split logic, and approved integrity metadata.

Before changing this repository from private to public, the authors should verify and document:

- the factual source citation and access route for the Chinese corpus;
- collection/access conditions and applicable platform terms;
- dataset redistribution rights;
- ethics/consent basis and privacy safeguards; and
- any retention or authorized-access procedure required by the institution or source.

---

## 3. External English Transit Tweet Data

### 3.1 Source

**Dataset:** Transit Tweet Data  
**Associated article:** Egbe-Etu Etu, Asha Weinstein Agrawal, Jordan Larot, Chaitanya Tatipigari, and Imokhai Theophilus Tenebe, “Learning About the Transit Passenger Experience from Microblogging Posts,” *Findings*, 2025.  
**DOI:** https://doi.org/10.32866/001c.140966  
**Public dataset repository:** https://github.com/etujnr/Transit-Tweet-Data

The associated publication reports that the final dataset contains 559 transit-experience tweets from 2020–2023 and identifies the GitHub repository above as the data-availability location.

The public dataset repository contains the dataset CSV and README but, as verified for this release documentation, does not expose an explicit standalone license file. Therefore, this repository does **not** redistribute the English CSV or archive. Users should obtain the dataset from the original source and comply with the source's applicable terms.

### 3.2 Audited local input

| Item | Expected value |
|---|---|
| Local archive path | `inputs/v6/step13/Transit-Tweet-Data-main.zip` |
| Archive SHA-256 | `47910a2c136e61b54605ebc23d81d6cdbc58aa9bc00633f3b58144ec620d159e` |
| CSV member | `Transit-Tweet-Data-main/Transit Tweet Data.csv` |
| CSV SHA-256 | `3e386a2d6e0ba4023559992415880687326a43c86a7be36d226d960e1a39ec9d` |
| Physical CSV rows | 561, including two blank trailing rows |
| Retained records | 559 unique, nonblank posts |
| Recorded period | 2020-01-01 01:55:17 UTC to 2023-03-31 12:43:02 UTC |

The source fields are `Created_at`, `Year`, `Month`, `Text`, `latitude`, and `longitude`. The experimental loader uses only the date/year/month/text information required for integrity checking and modeling. Latitude and longitude are not model inputs and are not written to generated evidence.

### 3.3 Frozen chronological protocol

Records are ordered by parsed UTC timestamp and then by original CSV row.

| Partition | Documents | Experimental use |
|---|---:|---|
| Original training | 391 | Initial external fitting basis |
| Spent development | 84 | Added to training after development was closed |
| **Final fitting set** | **475** | First 475 chronological records |
| Held-out test | 84 | Final 84 chronological records |
| **Total retained records** | **559** | — |

The English vocabulary, semantic/context representation, graph, calibration quantities, pooled lexical prior, and fitted model parameters are estimated using only the English fitting corpus. The vocabulary is capped at 1,148 terms.

The 84 held-out documents undergo deterministic 50/50 token-level document completion: only the observed half is supplied for inference and the target half is reserved for scoring.

### 3.4 Interpretation of the external experiment

The English evaluation is an **architecture-level external replication with language-compatible refitting**. It is not zero-shot transfer of the fitted Chinese model state.

The dataset has no ground-truth topic labels and no audited city identifier appropriate for the internal macro-city protocol. Consequently, label-based classification metrics and macro-city NLL are not reported. Geographic coordinates are not used.

---

## 4. Relationship between the two datasets

The two evaluations answer different questions:

- the Chinese corpus supports internal model development, final fitting, confirmation, human interpretation, and city-stratified analysis;
- the English corpus evaluates whether the same frozen architecture and hyperparameter policy can be refitted and evaluated in a temporally and linguistically distinct corpus;
- the external experiment does not establish zero-shot cross-language transfer;
- neither experiment uses city identifiers as model inputs; and
- human judgments are post-fit evaluation evidence only and are never used to fit the model.

---

## 5. Local data placement

Raw data are intentionally ignored by Git. Authorized users should place local files under `inputs/` according to the stage-specific configuration.

For the external dataset:

```powershell
New-Item -ItemType Directory -Force .\inputs\v6\step13 | Out-Null
Copy-Item -LiteralPath "C:\path\to\Transit-Tweet-Data-main.zip" `
          -Destination .\inputs\v6\step13\Transit-Tweet-Data-main.zip

Get-FileHash .\inputs\v6\step13\Transit-Tweet-Data-main.zip -Algorithm SHA256
```

The reported SHA-256 value should match the audited archive hash above. Do not commit the archive or extracted raw CSV to this repository.

---

## 6. Public-release checklist

Before making the repository public, confirm that:

- [ ] the Chinese source citation and access route are documented;
- [ ] Chinese data collection/access terms and redistribution rights are verified;
- [ ] ethics/privacy statements are factually supported;
- [ ] no raw posts, identifiers, coordinates, profiles, comments, annotator-response files, archives, or checkpoints are tracked by Git;
- [ ] the source/processed-count provenance is reconciled or explicitly retained as separate pipeline stages;
- [ ] the external dataset is obtained from its original repository rather than redistributed here; and
- [ ] displayed aggregate counts and hashes match the frozen release artifacts.

---

## 7. Citation

For the external English dataset, cite:

> Etu, E.-E., Agrawal, A. W., Larot, J., Tatipigari, C., & Tenebe, I. T. (2025). Learning About the Transit Passenger Experience from Microblogging Posts. *Findings*. https://doi.org/10.32866/001c.140966

For the Chinese corpus, insert the verified source citation in the manuscript and repository documentation before public release.
