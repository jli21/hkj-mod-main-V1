# HKJC Model — Session Findings (2026-04)

Comprehensive summary of feature engineering + architecture work on the HKJC horse racing model.

---

## TL;DR — Current best config

- **Model:** Offset (market as `base_margin`) + 5-seed bagging
- **Seeds:** `[7, 17, 42, 123, 256]`
- **Params:** `lr=0.01, depth=5, min_child_weight=20, subsample=0.7, colsample_bytree=0.7, gamma=0.2, reg_alpha=2.0, reg_lambda=10.0`, 400 rounds (deep_slow)
- **Features:** 19 edge + 4 race card + 2 market = **25 total** (see feature set below)
- **Kelly:** 0.02 for consistency / 0.03 for max return

### Results (2015-2026, compounded; 2026 partial)

**⚠️ Honest caveat first:** the bag_A seeds `[7, 17, 42, 123, 256]` that we use as production were retested against 4 other independent 5-seed sets. **Bag_A was the luckiest tail** — across 5 meta-sets the mean cum ROI at K=0.02 is **+107%**, with std 130%, min **+24%**, max +332% (bag_A). So production's bag_A numbers below represent the top of the distribution, not the expected case.

Headline numbers (bag_A, lucky tail):

| Kelly | Cum no_rebate | Cum with_rebate | Pos no_rebate | Pos with_rebate |
|---|---|---|---|---|
| 0.01 | +139.47% | +260.26% | 9/12 | 11/12 |
| **0.02** | **+332.53%** | +880.08% | **9/12** | **11/12** ← bag_A |
| 0.03 | +522.74% | +2027.26% | 7/12 | 10/12 |
| 0.05 | +644.80% | +5701.83% | 7/12 | 9/12 |

**Realistic production expectation (mean across 5 independent 5-seed bagging sets, K=0.02):**

| Stat | cum_no | cum_wr |
|---|---|---|
| MEAN | +107.18% | +366.46% |
| STD | 129.60% | 295.37% |
| **MIN** | **+23.99%** | **+178.64%** |
| MAX | +332.48% | +879.87% |

**All 5 bagging sets positive (min +24%).** This is the genuine win: bagging reliably eliminates catastrophic losing seeds (single-seed min was −19%), even if it doesn't reliably deliver the bag_A bonanza.

Out-of-sample: 2025 slight negative (~−2% at K02 averaged across meta-sets), 2026 partial positive across all 5 meta-sets (+1% to +7%).

### Running it

```bash
cd model/
python run_production.py
```

Generates `results/bankroll_viz.html` — interactive visualization with Kelly tabs + year tabs.

---

## Feature set (final)

### Market (2)
- `Implied_Prob` — market win probability (normalized per race)
- `log_implied_prob` — log of implied

### Race card (4)
- `Racecourse` — ST=1, HV=0
- `Track_Turf` — Turf=1, AWT=0
- `Draw` — starting stall (1-14)
- `Act_Wt` — weight carried (lbs)

### Edge (19)

**Class (1):** `class_change` = `prev_class − current_class`

**Weight (2):**
- `wt_z` — z-score of `Act_Wt` within current field
- `setup_weight_z` — z-score vs horse's own weight history

**Form (4):**
- `recent_form` — avg(1/place) over last 3 races (concave weighting, empirically better than linear)
- `form_vs_career` — recent_form − career_form
- `career_beat_odds` — expanding mean of (win − implied_prob) per horse
- `prev_win` — binary: won last race

**Trainer (3):**
- `trainer_track_spec` — trainer's track-specific WR minus overall WR
- `tt_gap` — trainer × track actual WR minus market-implied WR
- `tr_gap` — trainer overall actual WR minus market-implied WR

**Draw/Field interactions (2):**
- `draw_outside_ST` — binary: Draw ≥ 10 AND at ST
- `fav_field_size` — is_fav × field_size

**Barrier trial (2):**
- `bt_avg_early` — expanding avg front-running in trials
- `bt_last_behind` — time behind winner in last trial

**Sectional (4):**
- `trail_sect_closing` — avg closing speed vs race (trailing per horse)
- `trail_sect_peak_at` — avg position of peak speed (0=early, 1=late)
- `trail_sect_avg` — avg overall sectional speed vs race
- `trail_sect_best_gain` — avg biggest in-race position gain

**Body weight (1) — NEW this session:**
- `body_wt_mkt_residual` — time-respecting market-failure residual bucketed by `(body_wt_change × implied_prob)`. Formula: `(past_wins − past_implied) / (past_implied + k)` with k=50, per bucket.

---

## What we tried — chronologically

### 1. Feature pruning (prune2) ✓ WIN

Dropped the 2 lowest-importance features: `top3_count_5` (rank 22.1) and `field_form_cv` (rank 22.6). 

**Result: +69pp paired delta over baseline, robust across 3 seeds, LL unchanged.**

Key lesson: low-importance LEAF features can be safely pruned. Low-importance BACKBONE features (Racecourse, Track_Turf) cannot — they enable downstream interactions even when their own gain is low.

### 2. Draw re-engineering (`draw_bias_loc`) ❌ FAILED

Built time-respecting bucketed lift feature at `(Racecourse, Dist, Draw)`. Correctly encoded empirical draw bias (HV 1200m Draw 1 = +41% win rate advantage; ST 1000m reversed).

**Result: −21pp paired delta.** Market already prices draw bias. Duplicating it added noise.

**Key lesson:** Don't encode features the market already prices. Target market-failure residuals instead.

### 3. prev_mkt_residual (prev_win swap) ⚠️ NEUTRAL

Replaced binary `prev_win` with bucketed residual `(prev_won × prev_impl)`. Empirical analysis showed market under-reacts to repeat favourites (+2.1% edge) and overprices fluke longshot winners (−0.4%).

**Result:** Single-seed showed +20pp, but across 3 seeds mean was +9pp with high variance. Not clearly better than noise.

### 4. class_change re-engineering ❌ FAILED

Built market-failure residual bucketed by `(RaceClass_ord × class_change)`. Empirical analysis showed strong patterns: class rises +1.30% edge, class drops −0.61%, Class 2 rises the strongest at +21% residual.

**Result: −38pp paired delta across seeds.**

**Key lesson:** Don't re-encode the model's top-importance features. `class_change` is consistently rank #1 — XGBoost already finds optimal splits. Adding a derived version wastes tree capacity.

### 5. Sectional consolidation ❌ FAILED

Tried reducing 4 sectional features to 2 or 1 composite (`sect_kick = closing − avg`).

**Result: −22pp to −47pp paired delta.** All 4 sectionals carry complementary information — not redundant.

### 6. Form normalization ❌ FAILED

Swapped `recent_form = avg(1/place)` for `avg((field_size − place + 1)/field_size)` to properly account for field size.

**Result: −32pp paired delta.** The concave `1/place` encoding implicitly weights top finishes more heavily — turns out this is the right shape for horse-quality signal. Linear normalization is too democratic.

### 7. Dropping odds-dependent features ❌ FAILED

Tested LOO on each of: `fav_field_size`, `career_beat_odds`, `tt_gap`, `tr_gap`. All have moderate corr with market; rationale was reducing deployment fragility.

**Result: Every LOO hurt.** fav_field_size: −40pp, career_beat_odds: −30pp, tt_gap: −21pp, tr_gap: −21pp. All 4 removed: −33pp. Every odds-dependent feature earns its keep.

### 8. body_wt_mkt_residual ✓ **WIN**

**First robust feature addition.** Built from previously-unused `Declar. Horse Wt.` field in horses JSON.

Bucketed market-failure residual at `(body_wt_change_bucket × impl_bucket)`. Feature is nearly orthogonal to market (corr = +0.01).

Empirical insights:
- Favourites (impl > 0.15) with weight changes outperform market by +2-3% edge
- Horses after long rest (>60 days): weight GAIN is positive (+1.5% edge)
- Volatile-weight horses with big drops: +0.7% edge
- Prev winners with weight drops: +1.3% edge

**Result (8-seed test):**
- Baseline mean: +14.42%, min −33.3%, max +66.4%
- With body_wt_residual: mean **+48.74%**, min **−3.0%**, max +91.0%
- Mean paired Δ: **+34.32%**, 7/8 seeds positive
- **Downside compressed dramatically (−33% → −3%).**

Also tested raw body_wt_z + body_wt_change (2 features) and `body_wt_after_win` targeted binary — both hurt. Only the bucketed residual works.

### 9. 5-seed Offset bagging ✓ **WIN**

Train 5 Offset models on same data with different XGBoost seeds. Average predictions per race, renormalize.

**Result (3 meta-seed sets × 10 years):**
- Mean cum ROI: +70.98% (vs +48.74% single-seed)
- Min cum: **+41.38%** (vs −3.0% single-seed)
- Best meta-set: +103.65% cum, **8/10 positive years** (vs 5-6/10 single-seed)
- **Flipped 2018 AND 2019 from negative to positive** — both were structural losers

**Why:** XGBoost has random elements (column subsampling, row subsampling, split randomization). Different seeds → different trees → averaging cancels noise. LL stays flat (~0.24520) — bagging doesn't improve *prediction accuracy*, it reduces *prediction variance* that Kelly was amplifying.

### 10. Calibration + Softmax ensemble ❌ FAILED

Tested:
- Bagged Offset + isotonic calibration: **−141pp**
- Softmax (with log_implied_prob + isotonic): **−91pp**
- Ensemble (0.5 × bagged_offset_cal + 0.5 × softmax_cal): **−62pp**

**Result: bagged Offset alone is the architectural ceiling.**

- Isotonic on Offset hurts because market anchor (`base_margin`) already provides calibration by construction.
- Softmax underperforms because it has to learn the market anchor from scratch (as a feature) whereas Offset hard-wires it.
- Ensembling a good model with a bad one produces a mediocre model.

### 11. Kelly sweep on bagged ✓ **WIN**

With bagging reducing variance, Kelly can safely be 2-3x higher than the old 0.01 standard.

| Kelly | Cum ROI | Pos |
|---|---|---|
| 0.01 | +103.66% | 8/10 |
| 0.02 | +218.63% | 8/10 |
| 0.03 | +303.92% | 7/10 |
| 0.05 | +290.71% | 6/10 |

Kelly 0.03 is the new sweet spot; Kelly 0.02 the conservative pick.

### 12. Truncated training (last 5 / last 3 years) ❌ FAILED

Hypothesis: if the market is pricing in our edges over time, training only on
recent years should outperform full-history training. Tested at K=0.02, 3 seeds:

| Window | Mean cum no_rebate | Seeds |
|---|---|---|
| Full (2010→$t$−1) | **+103.34%** | +26.6%, −18.9%, +302.4% |
| Last 5 years | −71.37% | −69.2%, −72.7%, −72.2% |
| Last 3 years | −85.06% | −67.2%, −91.1%, −96.9% |

**Truncation catastrophically hurt overall.** BUT year-by-year showed
partial recency gains: last_3 beat full history in 2024 (+7.92% vs −4.46%),
2025 (+21% vs −7%), and 2026 (+3% vs −0.3%). So the edge-decay story has
partial support, but simple truncation is too blunt — the cost in
2015-2021 dwarfs recency gains in 2024-2026.

**Next attempt (untested):** recency-weighted training
($w = e^{-\lambda\,\text{years\_ago}}$) instead of throwing away data.

### 13. 5-seed variance study: "was bag_A lucky?" ✅ YES

Ran 5 INDEPENDENT 5-seed bagging meta-sets at K=0.02 over 12 years:

| Meta-set | cum_no | cum_wr | pos_no |
|---|---|---|---|
| bag_A (production) | **+332.48%** | +879.87% | 9/12 |
| bag_B | +101.18% | +354.02% | 9/12 |
| bag_C | +29.07% | +188.39% | 7/12 |
| bag_D | +23.99% | +178.64% | 6/12 |
| bag_E | +49.19% | +231.36% | 6/12 |
| **MEAN** | **+107.18%** | +366.46% | 7.4/12 |
| **STD** | 129.60% | 295.37% | — |
| **MIN** | +23.99% | +178.64% | 6/12 |

**Bag_A was the luckiest of 5 independent bagging attempts.** Mean realistic
production is ~+107% cum (not +332%). But:

- **All 5 meta-sets positive** (min +24%) — bagging reliably removes catastrophic runs.
- Single-seed comparison: MEAN +131%, MIN **−18.9%**, MAX +302%.
- So bagging's actual benefit is *downside compression* (min −19% → min +24%), NOT raising the expected mean.
- **LL is identical across all 5 meta-sets** (0.23708-0.23709) — confirms the variance-reduction mechanism.

Years robust across meta-sets:
- **2016, 2022, 2023, 2026**: positive in all 5 meta-sets.
- **2020**: negative in all 5 meta-sets (structural COVID-era bad year).
- **2015**: chaotic (−24% to +82% depending on meta-seed) — the main source of
  production variance.

---

## Year-by-year (bagged Offset, Kelly 0.01, no rebate)

| Year | Single seed=42 | 5-seed bagged (bag_A) |
|---|---|---|
| 2015 | +39.12% | +35.48% |
| 2016 | +24.71% | +20.95% |
| 2017 | −0.96% | −2.02% |
| 2018 | −3.46% | **+2.46%** |
| 2019 | −7.71% | **+5.24%** |
| 2020 | −9.27% | −5.34% |
| 2021 | +15.31% | +11.45% |
| 2022 | +4.74% | +7.19% |
| 2023 | +3.24% | +3.08% |
| 2024 | −0.65% | +0.90% |
| **Cum** | **+72.04%** | **+103.66%** |

---

## Methodology lessons

### 1. LL-flat / ROI-variable is the norm

Most feature tweaks leave log-loss essentially unchanged (Δ ≤ 0.0001) but move ROI by ±30-100pp. This is Kelly amplifying tiny probability shifts. Single-seed ROI differences at this scale are noise-dominated.

**Rule:** Rank feature changes by paired-delta across 3+ seeds, not single-seed ROI.

### 2. Market-failure residuals > raw features > derived re-encodings

Pattern that worked: take a feature with market-orthogonal signal (body weight), discretize by meaningful buckets (weight change × implied), compute time-respecting `(actual − implied)/ (implied + k)` per bucket. This targets exactly where market is wrong.

Pattern that failed: re-encoding features already in the model (class_change_residual, draw_bias_loc). Adds correlated info → overfitting surface.

### 3. Importance rank ≠ marginal value

XGBoost gain-importance measures *this feature's contribution to splits*. Doesn't measure *this feature's value as interaction substrate*. Low-rank CATEGORICALS like Racecourse are essential backbones for other features' interactions; removing them crashes the model.

### 4. Bagging is cheap variance reduction

5-seed bagging: 5× training cost, +22pp mean ROI, eliminates downside. Almost certainly the best $ per effort ratio in the session.

### 5. Offset > Softmax for HKJC

The Offset model's `base_margin = logit(Implied_Prob)` hard-anchors predictions to market. Softmax trying to learn this from features is fundamentally weaker. Don't spend cycles on Softmax architectures.

---

## Reproducibility

Pipeline:
1. `python data-processing/prep-data.py` — builds `data/processed/race_features.parquet` with all 25 production features (including `body_wt_mkt_residual`). Regenerate after any data refresh or feature-code change.
2. `python model/run_production.py` — loads the parquet, trains the bagged Offset model across 2015-2024 at Kelly ∈ {0.01, 0.02, 0.03, 0.05}, writes `model/results/bankroll_viz.html`.

## Memory files

Individual findings preserved in `~/.claude/projects/.../memory/`:
- `production_state.md` — current config
- `body_wt_residual_finding.md` — the body weight feature
- `architecture_5seed_bagging.md` — bagging details
- `architecture_kelly_sweep.md` — Kelly 0.02-0.03 finding
- `architecture_cal_ensemble_rejected.md` — what NOT to retry
- `feedback_dont_duplicate_market.md`, `feedback_dont_reencode_top_features.md`, `feedback_pruning_strategy.md` — methodology guardrails
