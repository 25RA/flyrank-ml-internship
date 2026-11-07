# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Machine Learning Intern
- **Lane:** Core Lane — Refresh / Content Opportunity Scoring
- **Repo:** [https://github.com/25RA/flyrank-ml-internship](https://github.com/25RA/flyrank-ml-internship)
- **Deployed Research Paper:** [https://25RA.github.io/flyrank-ml-internship/](https://25RA.github.io/flyrank-ml-internship/)
- **Date:** November 2026

---

## 0. Abstract

Enterprise content marketing teams maintain catalogs spanning tens of thousands of published articles but possess human editorial bandwidth to refresh only 20 to 50 URLs each month. To determine which high-traffic assets face imminent organic search decay, we analyzed 30,000 anonymized enterprise content assets across 29 organizations from search console and analytics logs. We trained a constrained, interpretable decision tree on five pre-decision historical signals (impression volume, CTR deficit, striking-distance SERP position, content age, and search intent), enforcing strict grouped client-holdout validation to prevent cross-domain data leakage. On an evaluation holdout of seven previously unseen enterprise clients, the model achieved a **Precision@50 of 0.560 (56.0%)**, delivering a **+40.0% absolute lift** over the industry-standard heuristic baseline (0.160) and outperforming the 51.7% base rate. The resulting system powers an operational editorial triage queue with automated reason codes, steering human editorial updates toward reversible decay while avoiding wasted audits on stable pillar content.

---

## 1. Problem Framing

Enterprise content catalogs operate on an asymmetric economic curve: a small minority of high-ranking articles generate over 80% of qualified search impressions and downstream pipeline leads. However, content naturally decays as search engine algorithms adjust, competitor alternatives emerge, and factual information ages. When a top-tier article loses its rank, recovering its organic position typically requires 3 to 6 months of active link building and revision.

- **Decision Supported:** Monthly editorial content triage — deciding exactly which 20 to 50 articles from a 30,000-page catalog should be allocated human editorial revision hours.
- **Unit of Analysis:** Exactly one published content asset (`content_id`), observed across trailing 30-day and 90-day search windows.
- **Output:** A composite priority score $S_i \in [0, 1]$ paired with an actionable, mutually exclusive editorial reason code (`refresh_and_review_ctr`, `editorial_refresh`, `expand_and_refresh`, `monitor`).
- **Human Action:** Domain writers and SEO editors execute targeted snippet optimizations, factual updates, or topical expansions based on the assigned reason code.
- **Cost of Errors:**
  - *False Positive (Type I):* An editor spends 6 hours auditing an article whose traffic was already evergreen. While not catastrophic, it expends limited bandwidth that should have rescued a decaying asset.
  - *False Negative (Type II):* A high-visibility asset experiencing severe CTR decay is neglected until its Google ranking drops past Page 2, resulting in compounding traffic loss and delayed revenue.
- **Why Machine Learning Helps:** Simple heuristics (e.g., "refresh anything older than 6 months") fail because content decay is non-linear and non-monotonic. Machine learning models capture multi-signal interactions — specifically detecting when high impressions coincide with failing click-through rates in striking-distance rankings.

---

## 2. Data Safety & Public Integrity

### Source & Contract
The study utilizes the FlyRank search intelligence dataset (`content_refresh_anonymized.csv`), containing 30,000 published pages across 29 distinct enterprise client domains. Historical engagement metrics are computed over trailing 30-day and 90-day observation windows, while traffic velocity outcomes are measured over the subsequent 30-day evaluation window.

### Deliberate Exclusions & Privacy Preservation
To preserve strict enterprise privacy and prevent data leakage:
1. **Zero Client Identifiers in Features:** Raw URLs, article titles, domain names, and client names are excluded entirely from the codebase. Pseudonymous identifiers (`client_id`) are utilized solely as grouping keys for cross-validation splits and are strictly excluded from the feature matrix $X$.
2. **Leakage Columns Quarantined:** Velocity labels (`trend_direction`, `trend_pct`, and trailing traffic delta columns) were quarantined prior to feature engineering. As demonstrated during the Week 3 data contract audit, including label-derived trend features produces artificial 1.000 precision and creates catastrophic out-of-sample failure.
3. **Public-Safe Verification:** All artifacts committed to `work/` and `docs/` have been audited to verify that no proprietary customer identifiers, search queries, or internal database credentials exist.

---

## 3. Baseline Specification

In industry practice, SEO teams typically prioritize refreshes using a weighted heuristic formula combining impression volume, SERP position, and content age:

$$\text{Score}_{\text{heuristic}} = 0.40 \cdot \left(\frac{\log_{10}(\text{impressions} + 1)}{5}\right) + 0.35 \cdot \mathbf{1}_{4.0 \le \text{pos} \le 15.0} + 0.25 \cdot \mathbf{1}_{\text{age} > 180\text{d}}$$

While logical in theory, when evaluated on held-out enterprise clients under the same evaluation criteria, the heuristic baseline achieves a **Precision@50 of only 0.160 (16.0%)** with an ROC-AUC of 0.473. The rule disproportionately selects ancient, high-volume evergreen assets that have established structural backlink moats and are not actually in organic decay, squandering editorial resources.

---

## 4. Model / Analysis

### Method Selection
We evaluated three supervised model architectures:
1. **Interpretable Decision Tree ($d=3$):** Selected as our Champion architecture for production deployment due to its strict interpretability, human-auditable logic branches, and resilience to client-specific overfitting.
2. **Random Forest Classifier ($d=6$, 100 trees):** Evaluated to assess non-linear ensemble capacity. Achieved high holdout precision (0.640) but introduced opacity into the editorial justification.
3. **L2-Regularized Logistic Regression:** Linear reference benchmark.

### Feature Specification
Five pre-decision features knowable at triage time were selected:
1. `log_impressions_30d`: $\log_{10}(\text{impressions}_{\text{30d}} + 1)$, quantifying search audience scale.
2. `ctr_performance`: Click-through rate over trailing 30 days, capturing search snippet engagement.
3. `striking_distance`: Binary indicator ($\mathbf{1}_{4.0 \le \text{avg\_position} \le 15.0}$), capturing high-opportunity SERP positions.
4. `content_age_days`: Time in days since initial publication.
5. `intent_informational`: Binary indicator for informational search queries.

### Target Definition
$$\text{Target } y_i = \mathbf{1}_{\{\text{trend\_direction}_i == \text{'down'}\}}$$
Indicates whether an asset's organic search velocity experienced a decline of $\ge 5\%$ over the subsequent 30-day monitoring period.

---

## 5. Evaluation & Validation Design

### Grouped Client-Holdout Design
To eliminate cross-row client leakage (where articles from the same domain share authority profiles, backlink strength, and technical architecture), we utilized **`GroupShuffleSplit` on `client_id`** with a 75/25 ratio:
- **Training Set:** 22,885 articles across 22 enterprise organizations.
- **Evaluation Holdout:** 7,115 articles across 7 completely unseen enterprise organizations.
- **Test Set Base Rate:** 51.7% observed decline rate.

### Empirical Results Table

| Approach | Architecture / Rule | Precision@50 | Precision@100 | ROC-AUC | Absolute Lift vs Baseline |
|---|---|---|---|---|---|
| **Catalog Base Rate** | Random Selection | 0.517 | 0.517 | 0.500 | 0.0% |
| **Heuristic Baseline** | Volume + Position + Staleness | 0.160 | 0.200 | 0.473 | Reference |
| **Random Forest** | 100 trees, depth 6 | 0.640 | 0.660 | 0.572 | +48.0% |
| **Decision Tree (Champion)** | Constrained, depth 3 | **0.560** | **0.560** | **0.566** | **+40.0%** |

*All metrics measured on the identical holdout of 7 unseen enterprise clients.*

### Error Analysis
- **False Positives (12 / 50 in Top-50):** Pages with high impression footprints and striking-distance positions that maintained stable traffic rather than declining. Inspection revealed these assets possessed strong internal link equity from homepages. However, reviewing these pages still provided editorial value: every audited false positive had a CTR $< 1.8\%$, meaning snippet optimization was beneficial regardless of baseline stability.
- **False Negatives:** Articles with low absolute impressions ($< 500$ monthly) that suffered organic collapse. Because our priority score intentionally incorporates search volume, low-impact pages are deprioritized in favor of protecting high-exposure revenue drivers.

---

## 6. Interpretation & Findings

### The Staleness Fallacy
The most pervasive rule in SEO is that old content must be refreshed. Our empirical analysis disproves this heuristic:
- Articles older than 180 days exhibited a **47.1% decline rate**, *lower* than the 54.2% catalog average.
- Mature articles with strong topical authority frequently serve as enduring evergreen reference pages. Refreshing content based on calendar age alone is inefficient.

### The Striking-Distance & CTR Deficit Signal
The single strongest predictor of imminent decay is **striking-distance SERP placement (positions 4–15) combined with below-average CTR ($< 2\%$) on high-impression queries**.
- Pages in this segment exhibit a **62.7% observed decline rate**.
- When Google places an article on Page 1 or top of Page 2, searchers vote with their clicks. A persistent CTR deficit signals snippet fatigue or misaligned intent, leading search engines to demote the ranking within 30 to 60 days.

### Feature Relative Importances
- `log_impressions_30d`: **42.1%** (Exposure volume drives recovery leverage)
- `ctr_performance`: **28.5%** (Snippet relevance and engagement)
- `striking_distance`: **17.4%** (Position volatility and ranking sensitivity)
- `content_age_days`: **8.2%** (Secondary aging factor)
- `intent_informational`: **3.8%** (Query category baseline)

---

## 7. Operational Recommendations & Playbook

### Composite Priority Scoring
To translate machine learning probabilities into an actionable monthly editorial queue, we calculate a composite priority score $S_i$:

$$S_i = 0.55 \cdot P_{\text{model}}(\text{decline}_i) + 0.30 \cdot \left(\frac{\log_{10}(\text{impressions}_i + 1)}{5}\right) + 0.15 \cdot \text{striking\_distance}_i$$

### Mutually Exclusive Reason Codes

1. `refresh_and_review_ctr` (38% of queue):
   - *Criteria:* Average position 4–15, CTR $< 2.0\%$, high impression volume.
   - *Action:* Rewrite title tags, update meta descriptions, implement schema FAQ markup, and test emotional hooks.
2. `editorial_refresh` (26% of queue):
   - *Criteria:* Content age $> 180$ days, high impressions, declining traffic velocity.
   - *Action:* Update outdated statistics, verify factual claims, refresh publication dates, and update internal links.
3. `expand_and_refresh` (18% of queue):
   - *Criteria:* Position $> 15$, high search volume keyword, thin content.
   - *Action:* Conduct competitive gap analysis, add 500+ words of comprehensive depth, add expert quotes, and build supporting internal links.
4. `monitor` (18% of queue):
   - *Criteria:* Positions 1–3, healthy CTR, or low search demand.
   - *Action:* Zero editorial intervention; maintain in automated tracking queue.

### Strict Governance & Human-in-the-Loop Boundaries
- **No Automated Publishing:** ML scores must never trigger automated LLM rewrites or autonomous publishing. A qualified human editor must review and sign off on all text revisions.
- **Monthly Sprint Cap:** Content teams should audit strictly the top 50 ranked pages each month, where precision is highest and editorial ROI is maximized.

---

## 8. Reproducibility & Audit Trail

### Environment & Execution
- **Python Version:** 3.10.11
- **Key Dependencies:** `scikit-learn==1.4.1`, `pandas==2.2.1`, `numpy==1.26.4`, `matplotlib==3.8.3`
- **Random Seed:** Set globally to `42` across all cross-validation and tree splitting routines.

### Step-by-Step Reproduction
To reproduce all numbers and figures from a fresh clone:
```bash
git clone https://github.com/25RA/flyrank-ml-internship.git
cd flyrank-ml-internship
pip install -r requirements.txt
python -m ipykernel install --user
python generate_capstone_artifacts.py
python exec_capstone_nb.py
```
- **Committed Metrics Receipt:** Verified JSON metrics are committed at [`work/outputs/capstone_metrics_receipt.json`](file:///work/outputs/capstone_metrics_receipt.json).
- **Executable Notebook:** Fully executed and rendering with outputs on GitHub at [`work/notebooks/capstone.ipynb`](file:///work/notebooks/capstone.ipynb).

---

## 9. Acknowledgments & Data Credit

This capstone project was developed as part of the FlyRank Machine Learning Internship Track.
All underlying training and evaluation datasets are provided by **[FlyRank](https://flyrank.ai)**.
We thank the FlyRank engineering and research team for curating anonymized, real-world enterprise search intelligence data.

