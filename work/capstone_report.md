# Capstone Report — Lane 2: Refresh / Content Opportunity Scoring

- **Author:** abuhussein1504
- **Lane:** Lane 2 — Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/abuhussein1504/flyrank-ml-internship-starter
- **Date:** September 2026

> This file mirrors the deployed paper at `docs/index.html`. Every number below has a receipt
> in `work/outputs/*.json`; see Section 8.

## 0. Abstract

FlyRank researches, writes, and publishes content into client websites at scale, then watches
search performance over time — and content that ranks well quietly decays: rankings slip,
clicks drop, and teams often notice too late. With thousands of live pages and far more of them
than any team can manually re-check each cycle, the question this paper answers is: which pages
should a human look at first? Working from FlyRank's anonymized starter export — 30,000 pseudonymized pages across 32 clients, restricted to the
28,795 pages with real ranking signal — a transparent staleness-and-visibility rule is compared
against a client-grouped, leakage-checked Logistic Regression predicting an *observed* decline
outcome, evaluated at Precision@50 to match a realistic 50-page review budget. The learned model
ranks pages substantially better than both the rule and random ordering (Precision@50 ≈ 0.72 vs.
0.44 vs. a 0.564 base rate under a client-held-out split), while a naive random split on the same
data would have overstated that number by 0.14 points — a gap that is itself reported as a
finding. The output is a monthly, reason-coded review queue built for a human reviewer: decision
support, not an automated refresh trigger or a causal claim about what fixes a decline. A
parallel data-contract exercise against a 9.84-million-row production warehouse partition shows
the same grain and leakage discipline holds at scale, even though the headline numbers above come
from the smaller starter export.

## 1. Problem framing

This is FlyRank's own case study: FlyRank runs content as infrastructure — research, writing,
and publishing into a client's site, then ongoing search-performance monitoring — across a
portfolio far larger than any team can manually re-check every month, and content that ranks
well quietly decays if nobody is watching. **Unit of analysis:** one pseudonymized content page,
as it appears in a trailing-90-day snapshot. **Output:** a priority score and a reason-coded suggested action. **Who acts on it:** a
FlyRank content strategist / editor with a fixed monthly review budget (K = 50 pages, chosen in
`w02_ml_task_framing.ipynb` before any model was trained). **Cost of a wrong call:** a false
positive (flagging a healthy page) wastes ~10–15 minutes of review time; a false negative
(missing a genuinely declining page) lets the decline run unnoticed for another full cycle — an
asymmetric cost that favors erring toward review over silence. **Why ML helps:** FlyRank's
existing product flags are fixed-threshold rules on one or two signals at a time, and this
project's own signal audit (`w04_baseline_score.ipynb`) found the most obvious one — staleness —
is a **MIXED** predictor of decline on its own (decline rate rises through the 91–180-day
freshness tier, then drops at 181+ days). A model that weighs several signals jointly, rather
than one hand-tuned threshold, is positioned to close that gap.

## 2. Data safety

**Working dataset:** `data/raw/content_refresh_anonymized.csv` — 30,000 rows, 44 columns, one row
per pseudonymized content page, trailing-90-day snapshot, 32 clients. This is what the baseline,
model, validation, and action queue are built and measured on.

**Warehouse rehearsal (separate from the model above):** `FlyRank/internship-warehouse` on
Hugging Face — a data contract was verified against its March 2026 partition: 9,841,378 rows,
331,437 content items, 55 clients, 2026-03-01–2026-03-31, a slice of the full ~78.8M-row release.
This proves the same grain/field/leakage discipline holds at production scale but did **not**
feed the model below (`w03_data_contract.ipynb`).

**Excluded, and why:**

| Field | Why excluded |
|---|---|
| `avg_position = 0` | means "no ranking data," not rank zero |
| `impression_tier = "no_data"` | no real impression reading — dropped from the eligible population |
| `content_id`, `client_id` | pseudonymous IDs — grouping/splitting only, never features |
| `trend_direction`, `trend_pct` | the label's own source — predicted, never used as a feature |
| `ga4_*` before `ga4_data_start` | zero-filled placeholders in the warehouse table, not real zero-engagement days |

After exclusions, **28,795 of 30,000 pages (96.0%)** form the eligible population every metric in
this report is computed against. No client name, domain, URL, page title, or keyword appears
anywhere in this repository — every identifier is a pseudonymous hash (verified by grep across
`work/` before each commit).

## 3. Baseline

`stale_but_visible` (`w04_baseline_score.ipynb`): flags a page when `days_since_last_update ≥ 90`
**and** `impressions_90d ≥ 500`, scored by raw `impressions_90d` among flagged pages and zero
otherwise. No fitted weights — every flagged page carries a reason a reviewer can check by hand.
It flags 6,575 of 28,795 eligible pages (22.8%) and scores **Precision@50 = 0.440** — *below* the
0.564 base rate, because it ranks by raw visibility rather than anything decline-shaped, which the
project's own signal audit already predicted (staleness alone: MIXED verdict).

## 4. Model / analysis

**Target:** `is_declining` = (`trend_direction == "down"`) — an observed outcome column, not a
rule this analysis defined itself. **Method:** Logistic Regression, then Random Forest (per the
project's method-selection rule: readable first, stronger second), scored by predicted
probability and ranked for Precision@50.

**Final feature set** (post-leakage-removal, `w05_model.ipynb`): trailing-90/30-day engagement
and search columns (impressions, clicks, sessions, users, scroll events, AI-traffic share, CTR,
average position, content age, days since last update, word/char count, search volume,
competition, CPC), missingness flags computed *before* filling blanks (missingness follows
`content_type`, not randomness), and one-hot encoded `content_type` / `main_intent`.
`content_id` / `client_id` are excluded from the feature matrix.

**The leakage catch:** an early pass included `impressions_last_30d` and
`impressions_prev_30d`. Precision@50 hit a suspicious **1.000**. The attack test confirmed why:
`corr((last − prev) / prev, trend_pct) = 0.99999998` — those two columns are `trend_pct`'s own
source columns under different names, not predictive features. Both were removed, and a second
sanity check — injecting `trend_pct` directly — confirmed the leak-detector is actually sensitive
(score jumped back to 1.000), rather than trusting a single clean pass.

## 5. Evaluation

**Split:** `GroupKFold` by `client_id`, 5 folds, across the 31 distinct clients in the eligible
slice — every fold tests on clients the model never trained on. No time-based split exists: the
starter export is a single trailing-90-day snapshot, not a daily panel (a real limitation of this
data, named here rather than hidden).

| Method | Precision@50 |
|---|---|
| Base rate (random ordering) | 0.564 |
| Week-4 rule (`stale_but_visible`) | 0.440 |
| **Logistic Regression (client-grouped)** | **0.720** |
| Random Forest (client-grouped) | 0.520 |

**How much of "0.72" is the split?** Holding model, features, and K fixed and changing only the
split: a naive, non-grouped row KFold reports Precision@50 = **0.860** — because essentially
every client (30.6 of 31, on average) appears on both sides of every fold, so part of "0.860" is
the model recognizing a client it has already seen. The 0.14-point gap is reported here as a
finding, not smoothed over (`w06_validation_audit.ipynb`).

**Error analysis:** of the top 50 LR-ranked pages, 14 are false positives against
`trend_direction`, and 12 of those 14 are already "stable" rather than "up" — the model is
catching a genuine recent click drop on pages whose longer trend hasn't yet crossed the export's
own boundary into "down." Given the asymmetric cost from Section 1, this is the cheaper kind of
mistake: a wasted look, not a missed decline.

## 6. Interpretation

Averaged, standardized Logistic Regression coefficients put `clicks_last_30d` as by far the
strongest signal pushing toward "declining" (a recent drop), with `sessions_90d` and
`clicks_prev_30d` the strongest signals pushing the other way (a healthy longer-run baseline) — a
*recent dip against a longer baseline*, not any single raw count. This is a reassuring shape, not
a "suspiciously perfect" one. **Negative result, stated plainly:** Random Forest (0.520) does not
beat the base rate (0.564) at this K — added complexity did not buy more precision on this
feature set, and that is reported as a finding rather than tuned away.

## 7. Recommendation

The rule supplies a reason a human can check; the model supplies the order. Priority sequence
(first match wins), computed across all 28,795 eligible pages (`w07_action_playbook.ipynb`):

| # | Action | Pages | Why |
|---|---|---:|---|
| 1 | `refresh_priority_review` | 2,761 | rule and model agree — strongest, most auditable case |
| 2 | `investigate_quiet_risk` | 3,989 | model flags risk, rule doesn't — the rule's blind spot |
| 3 | `refresh_review_routine` | 3,740 | rule flags it, model doesn't confirm elevated risk |
| 4 | `verify_before_action` | 170 | ≥500% swing in the label's own source columns — confirm before trusting |
| 5 | `monitor_too_new` / `monitor_not_owned` | 478 / 1,366 | no-go: too little history, or syndicated/not owned |
| 6 | `monitor` | 16,291 | no elevated signal from either rule or model this cycle |

**Confidence and limits, stated explicitly:** this is decision support for a human reviewer, not
an auto-refresh trigger. It has not been tested forward in time or on a new client. Before acting
on any item: check for a seasonal/one-off event, drifted search intent, a client-side publishing
change, or (for navigational-intent pages) a structurally low CTR that isn't a content problem.
**Monitoring policy:** recompute Precision@50 for rule, model, and base rate every labeled period;
if the model falls below the frozen rule for two consecutive periods, revert the live queue to
the rule and escalate for a retraining review — proposed here since this single-snapshot dataset
has no second labeled period yet to test the policy against.

## 8. Reproducibility

Every number above traces back to a committed file:

| Artifact | Holds |
|---|---|
| `work/notebooks/w04_baseline_score.ipynb` | the rule, its eligible population, top-10 human review |
| `work/notebooks/w05_model.ipynb` | feature build, the leakage catch, LR / RF training |
| `work/notebooks/w06_validation_audit.ipynb` | grouped-vs-naive split comparison, leak-detector sanity check |
| `work/notebooks/w07_action_playbook.ipynb` | the reason-coded queue, no-go rules, retrospective audit |
| `work/outputs/w04_baseline_metrics.json`, `w05_model_metrics.json`, `w06_validation_audit.json`, `w07_playbook_metrics.json` | every metric in this report, as committed receipts |
| `work/figures/w07_precision_at_k.png`, `w07_action_mix.png` | the two charts embedded in the deployed paper |
| `work/notebooks/w03_data_contract.ipynb` | the 9.84M-row warehouse grain/field verification (Section 2) |

**Re-run:** clone the repo, `pip install -r requirements.txt`, open any
`work/notebooks/*.ipynb` (each has a Colab badge reading straight from this repository).
**Seeds:** `random_state=42` on every estimator that exposes one; `GroupKFold(n_splits=5)` is
deterministic given the same client grouping. The warehouse notebooks additionally need a
Hugging Face read token (`HF_TOKEN`, gated, instant approval) via Colab secrets — never pasted
into a cell.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — https://flyrank.ai
